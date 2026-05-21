"""
api/main.py — FastAPI application for the LangGraph agent stack.

Primary HTTP surface:

* ``POST /run`` / ``POST /run/stream`` — Legacy routes; pack class from
  ``get_legacy_pack_cls()`` (``DEFAULT_PACK_ID`` registry default at startup).
* ``POST /packs/{pack_id}/run`` (+ stream) — Typed routes built from ``PackRegistry``
  at lifespan (e.g. ``research_analysis``, ``research_only``).
* ``GET /packs`` — Pack discovery (schemas, metadata).
* ``POST /research`` — Standalone ``ResearchAgent`` (not a domain pack).
* ``GET /health`` / ``GET /ready`` — Probes.

Architecture notes
------------------
* Application lifecycle is managed via a single ``lifespan`` context manager
  (FastAPI modern pattern — no deprecated ``@app.on_event`` decorators).
* All agent calls are inherently CPU/IO-bound and blocking.  Each endpoint
  offloads them to a ``ThreadPoolExecutor`` via ``asyncio.get_event_loop()
  .run_in_executor()`` so the event loop is never stalled.
* CORS origins are driven by ``settings`` — never hard-coded.
* Secrets are loaded exclusively from the environment / ``.env`` file via the
  ``Settings`` pydantic-settings model in ``core.config``.
* ``CONNECTOR_ENABLED`` + ``CONNECTOR_ID``: lifespan resolves a shared connector
  (``core/connectors.py``) and injects it into ``ResearchAnalysisPack`` via
  ``_pack_runtime_kwargs()``; other packs ignore it.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import hmac
import json
import logging
import threading
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated, Any, cast

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi import Path as FastAPIPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from langchain_core.language_models import BaseChatModel

from agents.analyst import AnalysisReport
from agents.base_agent import (
    AgentBudgetExceededError,
    AgentExecutionError,
    AgentTimeoutError,
    AgentValidationError,
)
from agents.researcher import ResearchAgent
from api.models import (
    ComponentHealth,
    HealthResponse,
    HistoryEntry,
    HistoryResponse,
    ResearchRequest,
    ResearchResponse,
    RunRequest,
    RunResponse,
)
from control_plane.enforce import (
    effective_budget_usd,
    effective_stream_timeout_seconds,
    validate_pack_body,
    validate_query_for_pack,
)
from core.config import Settings, get_settings
from core.connectors import resolve_connector
from core.graph import MultiAgentGraph
from core.memory import cleanup_checkpointer, create_run_history

# ---------------------------------------------------------------------------
# Logging — structured JSON via core.observability when available
# ---------------------------------------------------------------------------
from core.observability import (
    active_pipelines,
    configure_logging,
    create_metrics_app,
    http_request_duration_seconds,
    http_requests_total,
    init_tracing,
    metrics_path_label,
    requests_rejected_during_shutdown,
    server_shutting_down,
    set_request_id,
)
from core.security import (
    InputValidator,
    create_rate_limiter,
    ensure_request_body_within_limit,
    rate_limit_client_key,
    resolve_client_ip,
)

configure_logging(level=get_settings().log_level.value)
logger = logging.getLogger(__name__)

from pack_kernel.builtin_packs import register_builtin_packs  # noqa: E402
from pack_kernel.base_pack import normalize_pack_stream_event  # noqa: E402
from pack_kernel.registry import PackRegistry  # noqa: E402

register_builtin_packs()

# ---------------------------------------------------------------------------
# Module-level state (populated during lifespan startup)
# ---------------------------------------------------------------------------

_APP_VERSION = "0.5.0"
_start_time: float = 0.0
_executor: ThreadPoolExecutor | None = None
_shared_llm: BaseChatModel | None = None
_shared_checkpointer: Any | None = None
_shared_memory: Any = None
_active_pack_cls: Any = None  # resolved from PackRegistry at startup
_shared_connector: Any = None  # optional BaseConnector when CONNECTOR_ENABLED

# Security primitives — 60 requests per minute per IP is a conservative
# default suited for an LLM pipeline where each request may take several
# seconds.  The rate limiter is initialised in lifespan() so importing this
# module does NOT trigger a Redis connection (fixing side-effects-at-import).
_rate_limiter: Any = None
_input_validator = InputValidator(max_length=2000)
_shutting_down = threading.Event()


def _request_peer_host(request: Request) -> str | None:
    return request.client.host if request.client else None


def _request_client_ip(request: Request) -> str:
    settings = get_settings()
    return resolve_client_ip(
        _request_peer_host(request),
        request.headers,
        trust_proxy=settings.trust_proxy_headers,
        forwarded_allow_ips=settings.forwarded_allow_ips,
    )


def _rate_limit_key(request: Request) -> str:
    settings = get_settings()
    return rate_limit_client_key(
        _request_peer_host(request),
        request.headers,
        trust_proxy=settings.trust_proxy_headers,
        forwarded_allow_ips=settings.forwarded_allow_ips,
        api_key=settings.api_key,
    )


# ---------------------------------------------------------------------------
# Checkpointer health probe
# ---------------------------------------------------------------------------


def _check_checkpointer_health(settings: Settings) -> tuple[str, str]:
    """Probe the checkpointer backend for real connectivity.

    Returns:
        Tuple of ``(status, detail)`` — ``"ok"`` or ``"degraded"``.
    """
    backend = settings.memory_backend.value

    if backend == "redis" and settings.redis_url:
        try:
            import redis as redis_lib

            r = redis_lib.Redis.from_url(settings.redis_url, socket_timeout=2)
            r.ping()
            r.close()
            return ("ok", "redis reachable")
        except Exception as exc:
            return ("degraded", f"redis unreachable: {exc}")

    if backend == "postgres" and settings.postgres_url:
        try:
            import psycopg

            with psycopg.connect(settings.postgres_url, connect_timeout=2) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return ("ok", "postgres reachable")
        except Exception as exc:
            return ("degraded", f"postgres unreachable: {exc}")

    return ("ok", backend)


# ---------------------------------------------------------------------------
# LLM / checkpointer lazy initialisation
# ---------------------------------------------------------------------------


def _init_llm_and_checkpointer(settings: Settings) -> None:
    """Attempt to create the shared LLM and checkpointer.

    On failure the globals are set to ``None`` and a warning is logged.
    Called at startup and can be re-invoked to retry after a transient error.
    """
    global _shared_llm, _shared_checkpointer

    from core.llm import get_llm
    from core.memory import create_checkpointer

    try:
        _shared_llm = get_llm(settings.llm_config)
        _shared_checkpointer = create_checkpointer(settings)
        logger.info("LLM provider '%s' configured successfully", settings.llm_provider)
    except (ImportError, ValueError) as exc:
        logger.warning("LLM configuration warning: %s", exc)
        _shared_llm = None
        _shared_checkpointer = None


# ---------------------------------------------------------------------------
# Lifespan context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application startup and shutdown resources.

    Startup:
        * Records the process start time for uptime reporting.
        * Pre-warms a ``ThreadPoolExecutor`` used by all blocking agent calls.
        * Logs readiness with key configuration values (no secrets).

    Shutdown:
        * Gracefully shuts down the thread pool, waiting for in-flight tasks.
    """
    global _start_time, _executor, _shared_llm, _shared_checkpointer, _shared_memory
    global _rate_limiter, _active_pack_cls, _shared_connector

    _start_time = time.monotonic()

    _settings = get_settings()

    if _rate_limiter is None:
        _rate_limiter = create_rate_limiter(
            backend=_settings.rate_limit_backend,
            redis_url=_settings.redis_url,
        )

    if _settings.memory_backend.value == "postgres" and not _settings.postgres_url:
        raise RuntimeError(
            "POSTGRES_URL is required when MEMORY_BACKEND=postgres. "
            "Set the POSTGRES_URL environment variable."
        )
    if _settings.memory_backend.value == "redis" and not _settings.redis_url:
        raise RuntimeError(
            "REDIS_URL is required when MEMORY_BACKEND=redis. "
            "Set the REDIS_URL environment variable."
        )

    _executor = ThreadPoolExecutor(
        max_workers=_settings.thread_pool_max_workers,
        thread_name_prefix="agent-worker",
    )

    init_tracing()
    _init_llm_and_checkpointer(_settings)

    try:
        _active_pack_cls = PackRegistry.get(_settings.default_pack_id)
        logger.info(
            "Active domain pack resolved",
            extra={"pack_id": _settings.default_pack_id},
        )
    except KeyError as exc:
        raise RuntimeError(
            f"DEFAULT_PACK_ID '{_settings.default_pack_id}' is not registered. "
            "Check pack_kernel/__init__.py."
        ) from exc

    _shared_connector = resolve_connector(_settings)
    if _shared_connector is not None:
        logger.info(
            "Retrieval connector enabled",
            extra={"connector_id": _settings.connector_id},
        )

    # Wire per-pack routers from registry — guard against duplicate registration
    # which can occur in tests where the same module-level ``app`` object is
    # reused across multiple TestClient context managers (each one triggers this
    # lifespan afresh).
    _existing_prefixes = {
        getattr(r, "path", "").split("/{")[0] for r in app.routes if hasattr(r, "path")
    }
    for _pack_id in PackRegistry.list_packs():
        _expected_prefix = f"/packs/{_pack_id}/run"
        if _expected_prefix in _existing_prefixes:
            logger.debug(
                "Pack router already registered — skipping",
                extra={"pack_id": _pack_id},
            )
            continue
        _pack_cls = PackRegistry.get(_pack_id)
        _in_schema, _out_schema = PackRegistry.get_schemas(_pack_id)
        app.include_router(
            _build_pack_router(_pack_id, _pack_cls, _in_schema, _out_schema)
        )
        logger.info("Pack router registered", extra={"pack_id": _pack_id})

    _shared_memory = create_run_history(_settings)

    logger.info(
        "API server starting up",
        extra={
            "version": _APP_VERSION,
            "environment": _settings.environment,
            "host": _settings.api_host,
            "port": _settings.api_port,
            "llm_provider": _settings.llm_provider,
            "memory_backend": _settings.memory_backend.value,
        },
    )

    _shutting_down.clear()
    if server_shutting_down is not None:
        server_shutting_down.set(0)

    yield  # Application is live here

    logger.info("API server shutting down — draining in-flight requests")
    _shutting_down.set()
    if server_shutting_down is not None:
        server_shutting_down.set(1)
    if _executor is not None:
        _executor.shutdown(wait=True, cancel_futures=False)
    cleanup_checkpointer()
    if _shared_memory is not None:
        _shared_memory.close()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# Shared dependency accessors
# ---------------------------------------------------------------------------


_init_lock = threading.Lock()


def get_shared_llm() -> BaseChatModel | None:
    """Return the shared LLM, retrying init if the first attempt failed."""
    global _shared_llm
    if _shared_llm is None:
        with _init_lock:
            if _shared_llm is None:
                _init_llm_and_checkpointer(get_settings())
    return _shared_llm


def get_shared_checkpointer() -> Any | None:
    """Return the shared checkpointer, retrying init if the first attempt failed."""
    global _shared_checkpointer
    if _shared_checkpointer is None:
        with _init_lock:
            if _shared_checkpointer is None:
                _init_llm_and_checkpointer(get_settings())
    return _shared_checkpointer


def get_shared_memory() -> Any:
    return _shared_memory


def _pack_primary_text(body: Any) -> str:
    """Extract the main free-text field from a typed pack request body."""
    for field in (
        "query",
        "text",
        "company",
        "topic",
        "ticket_subject",
        "question",
        "role_title",
    ):
        if hasattr(body, field):
            value = getattr(body, field)
            if value:
                return str(value)
    return str(body)[:500]


def _validate_pack_body_fields(pack_id: str, body: Any) -> None:
    """Content-safety scan on all string fields (documents, tickets, resumes, etc.)."""
    try:
        validate_pack_body(body, pack_id, _input_validator)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


def _pack_has_structured_input(pack_cls: type) -> bool:
    """True when the pack class defines ``run_from_input`` (not a MagicMock artefact)."""
    return "run_from_input" in pack_cls.__dict__


def _pack_has_structured_stream(pack_cls: type) -> bool:
    """True when the pack class defines ``stream_events_from_input``."""
    return "stream_events_from_input" in pack_cls.__dict__


def _invoke_pack_run(pack_cls: type, pipeline: Any, body: Any) -> Any:
    """Call ``run_from_input`` when implemented on the pack class, else ``run(query)``."""
    if _pack_has_structured_input(pack_cls):
        return pipeline.run_from_input(body)
    return pipeline.run(_pack_primary_text(body))


async def _iter_pack_stream_events(
    pack_cls: type, pipeline: Any, body: Any
) -> AsyncIterator[Any]:
    """Stream pack events using structured input when the pack class supports it."""
    if _pack_has_structured_stream(pack_cls):
        events = pipeline.stream_events_from_input(body)
        async for event in cast(AsyncIterator[dict[str, Any]], events):
            yield normalize_pack_stream_event(event)
        return
    async for event in cast(
        AsyncIterator[dict[str, Any]],
        pipeline.stream_events(_pack_primary_text(body)),
    ):
        yield event


def _serialize_pack_result(
    result: Any, output_model: type, cost_usd: float | None
) -> Any:
    if hasattr(output_model, "from_analysis_report"):
        return output_model.from_analysis_report(result, cost_usd=cost_usd)
    if hasattr(output_model, "from_research_result"):
        return output_model.from_research_result(result, cost_usd=cost_usd)
    if hasattr(output_model, "from_summary_result"):
        return output_model.from_summary_result(result, cost_usd=cost_usd)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result


def _validate_pack_query(pack_id: str, raw_query: str) -> str:
    """Validate query text using pack policy constraints and global sanitizer."""
    try:
        return validate_query_for_pack(raw_query, pack_id, _input_validator)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


def _pack_runtime_kwargs(pack_cls: type) -> dict[str, Any]:
    """Extra constructor kwargs: policy budget and optional connector."""
    import inspect

    kwargs: dict[str, Any] = {}
    pack_id = getattr(pack_cls, "pack_id", None)
    if pack_id:
        budget = effective_budget_usd(pack_id, get_settings())
        if budget is not None:
            kwargs["budget_usd"] = budget
    if (
        _shared_connector is not None
        and "connector" in inspect.signature(pack_cls.__init__).parameters
    ):
        kwargs["connector"] = _shared_connector
    return kwargs


def get_legacy_pack_cls() -> type[Any]:
    """FastAPI dependency: pack class for legacy ``POST /run`` and ``POST /run/stream``."""
    return _active_pack_cls or MultiAgentGraph


# ---------------------------------------------------------------------------
# Shutdown guard helper (used by per-pack router endpoints)
# ---------------------------------------------------------------------------


def _guard_not_shutting_down() -> None:
    """Raise 503 if the server is in the process of shutting down."""
    if _shutting_down.is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is shutting down.",
        )


# ---------------------------------------------------------------------------
# API-key Depends helper (used by per-pack router endpoints)
# ---------------------------------------------------------------------------


def verify_api_key(request: Request) -> None:
    """Validate the shared Bearer secret when ``API_KEY`` is set.

    One global secret for all callers (no scopes or per-tenant keys). Returns
    ``None`` so callers use ``Depends(verify_api_key)`` only for side effect.
    Also enforced globally via ``auth_middleware``; this dependency documents
    the contract in OpenAPI for pack routes.
    """
    _api_key = get_settings().api_key
    if _api_key is None:
        return  # Auth disabled globally
    auth_header = request.headers.get("Authorization", "")
    token = (
        auth_header.removeprefix("Bearer ").strip()
        if auth_header.startswith("Bearer ")
        else ""
    )
    if not token or not hmac.compare_digest(token, _api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Per-pack router factory
# ---------------------------------------------------------------------------


def _build_pack_router(
    pack_id: str,
    pack_cls: type,  # type: ignore[type-arg]  — dynamic, mypy can't narrow here
    input_model: type,  # type: ignore[type-arg]
    output_model: type,  # type: ignore[type-arg]
) -> APIRouter:
    """Build a per-pack APIRouter with typed /run and /run/stream endpoints.

    Called during lifespan startup for each registered pack.
    The generated routes use pack-specific Pydantic models for request/response
    validation — inputs are rejected before any LLM call when invalid.

    Note: type annotations use bare ``type`` because the specific Pydantic
    subclass is only known at runtime (dynamic dispatch from PackRegistry).
    The ``# type: ignore`` comments on the function signature are intentional.

    Closure safety: ``pack_cls``, ``input_model``, and ``output_model`` are
    captured from the function parameters, not from a loop variable, so there
    is no late-binding hazard.  Each call to ``_build_pack_router`` creates a
    fresh scope with its own binding of these names.
    """
    router = APIRouter(prefix=f"/packs/{pack_id}", tags=[pack_id])

    # Define endpoint functions without decorator first so we can patch
    # __annotations__ with the real runtime types.  ``from __future__ import
    # annotations`` (active at module level) turns all annotations into strings —
    # ``body: input_model`` becomes the string ``"input_model"`` which FastAPI
    # cannot resolve.  Overriding __annotations__ with the actual classes before
    # router.add_api_route() is called ensures Pydantic receives the real model.

    async def run_pack(  # type: ignore[misc]
        body: input_model,  # type: ignore[valid-type]
        request: Request,
        response: Response,
        _auth: Annotated[None, Depends(verify_api_key)],
    ) -> Any:
        """Execute the pack pipeline synchronously."""
        _guard_not_shutting_down()
        run_id = str(uuid.uuid4())

        # Version pinning via request header
        requested_version = request.headers.get("X-Pack-Version") or None

        # Sticky session: if no explicit version pin, check session history
        if requested_version is None and _shared_memory is not None:
            session_id_for_sticky = getattr(body, "session_id", None) or None
            if session_id_for_sticky and hasattr(
                _shared_memory, "get_pack_version_for_session"
            ):
                requested_version = _shared_memory.get_pack_version_for_session(
                    session_id_for_sticky, pack_id
                )

        try:
            pack_cls_to_use = PackRegistry.get(pack_id, version=requested_version)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc

        # Determine the version string actually used (for the response header)
        used_version = next(
            (
                pv.version
                for pv in PackRegistry._get_versions(pack_id)
                if pv.pack_cls is pack_cls_to_use
            ),
            "unknown",
        )
        response.headers["X-Pack-Version-Used"] = used_version

        _validate_pack_body_fields(pack_id, body)
        raw_query = _pack_primary_text(body)
        try:
            query = _validate_pack_query(pack_id, raw_query)
        except AgentValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        def _execute() -> Any:
            with pack_cls_to_use(
                run_id=run_id,
                llm=get_shared_llm(),
                checkpointer=get_shared_checkpointer(),
                **_pack_runtime_kwargs(pack_cls_to_use),
            ) as pipeline:
                result = _invoke_pack_run(pack_cls_to_use, pipeline, body)
                cost_usd = getattr(pipeline, "cost_usd", None)

                if hasattr(result, "to_dict"):
                    result_payload = result.to_dict()
                elif hasattr(result, "model_dump"):
                    result_payload = result.model_dump()
                else:
                    result_payload = {}

                # Record run in history with pack_version metadata
                if _shared_memory is not None:
                    session_id_for_history = getattr(body, "session_id", None) or None
                    _shared_memory.save_run(
                        run_id=run_id,
                        query=query,
                        result=result_payload,
                        metadata={
                            "pack_id": pack_id,
                            "pack_version": used_version,
                            **(
                                {"session_id": session_id_for_history}
                                if session_id_for_history
                                else {}
                            ),
                        },
                    )

                return _serialize_pack_result(result, output_model, cost_usd)

        try:
            return await _run_in_executor(_execute)
        except AgentBudgetExceededError as exc:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED, detail=str(exc)
            ) from exc
        except AgentTimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)
            ) from exc
        except (AgentExecutionError, AgentValidationError) as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc

    # Patch __annotations__ so FastAPI / Pydantic see the real model classes,
    # not the deferred-evaluation string "input_model" produced by PEP 563.
    run_pack.__annotations__["body"] = input_model

    router.add_api_route(
        "/run",
        run_pack,
        methods=["POST"],
        summary=f"Run {pack_id} pipeline",
        response_model=output_model,
    )

    async def stream_pack(  # type: ignore[misc]
        body: input_model,  # type: ignore[valid-type]
        request: Request,
        _auth: Annotated[None, Depends(verify_api_key)],
    ) -> StreamingResponse:
        """Stream pack pipeline events as Server-Sent Events."""
        _guard_not_shutting_down()
        run_id = str(uuid.uuid4())

        requested_version = request.headers.get("X-Pack-Version") or None
        try:
            pack_cls_to_use = PackRegistry.get(pack_id, version=requested_version)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc

        used_version = next(
            (
                pv.version
                for pv in PackRegistry._get_versions(pack_id)
                if pv.pack_cls is pack_cls_to_use
            ),
            "unknown",
        )

        _validate_pack_body_fields(pack_id, body)
        raw_query = _pack_primary_text(body)
        try:
            _validate_pack_query(pack_id, raw_query)
        except AgentValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        async def _event_generator() -> AsyncGenerator[str, None]:
            pack = pack_cls_to_use(
                run_id=run_id,
                llm=get_shared_llm(),
                checkpointer=get_shared_checkpointer(),
                **_pack_runtime_kwargs(pack_cls_to_use),
            )
            try:
                async for event in _iter_pack_stream_events(
                    pack_cls_to_use, pack, body
                ):
                    yield f"data: {json.dumps(event, default=str)}\n\n"
            finally:
                pack.close()

        stream_timeout = effective_stream_timeout_seconds(pack_id, get_settings())

        async def _timed_event_generator() -> AsyncGenerator[str, None]:
            try:
                async with asyncio.timeout(stream_timeout):
                    async for chunk in _event_generator():
                        yield chunk
            except TimeoutError:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Stream timed out after {stream_timeout}s'})}\n\n"

        return StreamingResponse(
            _timed_event_generator(),
            media_type="text/event-stream",
            headers={
                "X-Pack-Version-Used": used_version,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Same annotation fix for the stream endpoint.
    stream_pack.__annotations__["body"] = input_model

    router.add_api_route(
        "/run/stream",
        stream_pack,
        methods=["POST"],
        summary=f"Stream {pack_id} pipeline",
    )

    return router


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

_is_production = get_settings().environment == "production"

app = FastAPI(
    title="LangGraph Agent Stack API",
    description=(
        "Production API exposing a multi-agent LangGraph pipeline. "
        "The pipeline sequences a ``ResearchAgent`` and an ``AnalystAgent`` "
        "to turn a free-text query into a structured ``AnalysisReport``."
    ),
    version=_APP_VERSION,
    docs_url=None if _is_production else "/docs",
    redoc_url=None if _is_production else "/redoc",
    lifespan=lifespan,
)

# Mount Prometheus metrics endpoint — exempt from auth and rate limiting
_metrics_app = create_metrics_app()
if _metrics_app is not None:
    app.mount("/metrics", _metrics_app)

# ---------------------------------------------------------------------------
# CORS — fail-closed: no wildcard unless explicitly set in CORS_ORIGINS
# ---------------------------------------------------------------------------

_cors_origins = get_settings().cors_origins

if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
elif not _is_production:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def add_security_headers(request: Request, call_next: Any) -> Any:
    """
    Attach security-relevant HTTP response headers to every reply.

    These headers harden browser-facing deployments against common web
    vulnerabilities (clickjacking, MIME-sniffing, information leakage).
    They are low-risk to add and impose no functional overhead.
    """
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"

    # CSP assoupli pour les pages de documentation (chargent JS/CSS depuis CDN)
    if request.url.path in ("/docs", "/redoc", "/openapi.json"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' cdn.jsdelivr.net; "
            "img-src 'self' data: fastapi.tiangolo.com;"
        )
    else:
        response.headers["Content-Security-Policy"] = "default-src 'self'"

    # HSTS is only meaningful over HTTPS — restrict to production to avoid
    # breaking local HTTP development and test environments.
    if get_settings().environment == "production":
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )
    # Remove the Server header to avoid advertising the runtime stack.
    if "server" in response.headers:
        del response.headers["server"]
    return response


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def auth_middleware(request: Request, call_next: Any) -> Any:
    """
    Optional Bearer-token authentication gate.

    When ``settings.api_key`` is set, every request to a non-exempt path must
    carry a matching ``Authorization: Bearer <token>`` header.  Exempt paths
    (health probes, interactive docs) are always allowed through.

    Disable auth entirely by leaving ``API_KEY`` unset in the environment.
    """
    _exempt = {
        "/",
        "/health",
        "/ready",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
    }
    if request.url.path in _exempt:
        return await call_next(request)

    _api_key = get_settings().api_key
    if _api_key is None:
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")
    token = (
        auth_header.removeprefix("Bearer ").strip()
        if auth_header.startswith("Bearer ")
        else ""
    )
    if not token or not hmac.compare_digest(token, _api_key):
        logger.warning(
            "Auth failed",
            extra={
                "path": request.url.path,
                "client": _request_client_ip(request),
            },
        )
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing Bearer token."},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Rate-limiting middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next: Any) -> Any:
    """
    Enforce a sliding-window rate limit on all incoming requests.

    Keys are scoped per Bearer token when ``API_KEY`` auth is enabled, else
    per client IP.  Behind a trusted reverse proxy, set ``TRUST_PROXY_HEADERS``
    and ``FORWARDED_ALLOW_IPS`` so ``X-Forwarded-For`` is used instead of the
    load-balancer address.

    The ``/health``, ``/ready``, and ``/metrics`` endpoints are excluded so
    Kubernetes probes are never blocked.
    """
    if request.url.path in {"/health", "/ready", "/metrics"}:
        return await call_next(request)

    limit_key = _rate_limit_key(request)
    if _rate_limiter is not None and not _rate_limiter.is_allowed(limit_key):
        logger.warning(
            "Rate limit exceeded",
            extra={
                "client": _request_client_ip(request),
                "limit_key": limit_key,
                "path": request.url.path,
            },
        )
        return Response(
            content='{"detail":"Rate limit exceeded. Please slow down."}',
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            media_type="application/json",
            headers={
                "Retry-After": str(int(getattr(_rate_limiter, "window_seconds", 60)))
            },
        )

    return await call_next(request)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    """
    Structured access log for every HTTP request.

    Logs method, path, status code, and wall-clock latency so that each
    request is traceable in aggregated log systems without extra tooling.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    set_request_id(request_id)
    start = time.monotonic()

    logger.info(
        "Request received",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client": _request_client_ip(request),
        },
    )

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    elapsed_s = time.monotonic() - start
    elapsed_ms = elapsed_s * 1000
    logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(elapsed_ms, 2),
        },
    )

    path_label = metrics_path_label(request.scope)
    if http_requests_total is not None:
        http_requests_total.labels(
            method=request.method,
            path=path_label,
            status_code=str(response.status_code),
        ).inc()
    if http_request_duration_seconds is not None:
        http_request_duration_seconds.labels(path=path_label).observe(elapsed_s)

    return response


_DRAIN_EXEMPT_PATHS = {"/health", "/ready", "/metrics"}


@app.middleware("http")
async def drain_middleware(request: Request, call_next: Any) -> Any:
    """Reject new requests with 503 when the server is shutting down.

    ``/health``, ``/ready``, and ``/metrics`` are exempt so Kubernetes
    probes continue to work during the drain period.
    """
    if _shutting_down.is_set() and request.url.path not in _DRAIN_EXEMPT_PATHS:
        if requests_rejected_during_shutdown is not None:
            requests_rejected_during_shutdown.inc()
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Server is shutting down."},
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Request body size limit — reject oversized payloads before JSON parsing
# ---------------------------------------------------------------------------


@app.middleware("http")
async def body_size_limit_middleware(request: Request, call_next: Any) -> Any:
    """Reject HTTP bodies above ``MAX_REQUEST_BODY_BYTES`` with 413.

    Checks ``Content-Length`` when present and performs a stream-bounded read
    otherwise so chunked uploads cannot bypass the cap.
    """
    settings = get_settings()
    bounded_request, error = await ensure_request_body_within_limit(
        request,
        settings.max_request_body_bytes,
    )
    if error is not None:
        logger.warning(
            "Request body too large",
            extra={
                "path": request.url.path,
                "client": _request_client_ip(request),
                "max_bytes": settings.max_request_body_bytes,
            },
        )
        return JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={"detail": error},
        )
    return await call_next(bounded_request)


# ---------------------------------------------------------------------------
# Helper: run a blocking callable in the thread pool
# ---------------------------------------------------------------------------


async def _run_in_executor(fn: Any, *args: Any) -> Any:
    """
    Execute a blocking callable in the application thread pool.

    Increments / decrements the ``active_pipelines`` Prometheus gauge so
    operators can observe in-flight pipeline concurrency.

    Args:
        fn: The synchronous callable to execute.
        *args: Positional arguments forwarded to ``fn``.

    Returns:
        The return value of ``fn(*args)``.
    """
    if _executor is None:
        raise RuntimeError("Application not started — call during lifespan only")
    if active_pipelines is not None:
        active_pipelines.inc()
    try:
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(
            _executor, functools.partial(ctx.run, fn, *args)
        )
    finally:
        if active_pipelines is not None:
            active_pipelines.dec()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/",
    include_in_schema=False,
    summary="Root redirect",
)
async def root() -> RedirectResponse:
    """Redirect browser traffic from ``/`` to the interactive API documentation."""
    return RedirectResponse(url="/docs", status_code=status.HTTP_302_FOUND)


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Operations"],
    summary="Health check",
    response_description="Service health status and uptime information.",
)
async def health(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """
    Return the current health status of the service.

    Performs a deep health check of LLM, memory, and checkpointer components.
    Returns ``"degraded"`` if any component is unavailable but the service
    itself is reachable.

    Returns:
        ``HealthResponse`` with status, version, uptime, component health,
        and environment.
    """
    components: dict[str, ComponentHealth] = {}

    # LLM health is an initialisation check only — we deliberately do NOT
    # call the provider on every probe to avoid latency, token cost, and
    # rate-limit pressure.  A stale API key or provider outage will surface
    # on the first real request, not on the liveness probe.
    if _shared_llm is not None:
        components["llm"] = ComponentHealth(
            status="ok", detail=f"{settings.llm_provider} (initialised)"
        )
    else:
        components["llm"] = ComponentHealth(
            status="degraded", detail="LLM not initialised"
        )

    if _shared_memory is not None:
        mem_status, mem_detail = _shared_memory.health_check()
        components["memory"] = ComponentHealth(
            status="ok" if mem_status == "ok" else "degraded",
            detail=mem_detail,
        )
    else:
        components["memory"] = ComponentHealth(
            status="degraded", detail="Memory store not initialised"
        )

    if _shared_checkpointer is not None:
        chk_status, chk_detail = _check_checkpointer_health(settings)
        components["checkpointer"] = ComponentHealth(
            status="ok" if chk_status == "ok" else "degraded",
            detail=chk_detail,
        )
    else:
        components["checkpointer"] = ComponentHealth(
            status="degraded", detail="Checkpointer not initialised"
        )

    overall = "ok" if all(c.status == "ok" for c in components.values()) else "degraded"

    return HealthResponse(
        status=overall,
        version=_APP_VERSION,
        uptime_seconds=round(time.monotonic() - _start_time, 3),
        environment=settings.environment,
        components=components,
    )


@app.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    tags=["Operations"],
    summary="Readiness probe",
    response_description="Returns 200 when the service is ready to accept traffic.",
)
async def ready() -> dict[str, str]:
    """Readiness probe for Kubernetes.

    Returns 200 only when LLM and checkpointer are initialised.
    Returns 503 if the service is not yet ready or is shutting down.
    """
    if _shutting_down.is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is shutting down.",
        )
    if get_shared_llm() is None or get_shared_checkpointer() is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready: LLM or checkpointer not initialised.",
        )
    return {"status": "ready"}


@app.post(
    "/run",
    response_model=RunResponse,
    status_code=status.HTTP_200_OK,
    tags=["Pipeline"],
    summary="Run the full Research + Analysis pipeline",
    response_description="Structured AnalysisReport produced by the full agent pipeline.",
)
async def run_pipeline(
    body: RunRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    pack_cls: Annotated[type[Any], Depends(get_legacy_pack_cls)],
) -> RunResponse:
    """
    Execute the complete multi-agent pipeline for a given query.

    The pipeline sequences two LangGraph agents:

    1. ``ResearchAgent`` — expands the query, retrieves information snippets,
       validates quality, and produces a ``ResearchResult``.
    2. ``AnalystAgent``  — consumes the research findings, extracts insights,
       identifies patterns, and produces an ``AnalysisReport``.

    The underlying agent calls are blocking and may take several seconds
    depending on the LLM response time.  The endpoint offloads them to a
    thread pool to keep the async event loop unblocked.

    Args:
        body: Request body containing the ``query`` string.

    Returns:
        A ``RunResponse`` containing the executive summary, key insights,
        patterns, implications, confidence score, and traceability metadata.

    Raises:
        422 Unprocessable Entity: When the request body fails validation.
        400 Bad Request: When the query is empty after stripping whitespace.
        402 Payment Required: When the run exceeds its configured USD cost budget.
        500 Internal Server Error: When the agent pipeline encounters an
            unrecoverable error.
        504 Gateway Timeout: When the agent exceeds its configured step budget.

    """
    if _shutting_down.is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is shutting down.",
        )

    try:
        query = _validate_pack_query(settings.default_pack_id, body.query)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty.",
        )

    if get_shared_llm() is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider is not configured. Check server logs.",
        )

    session_id = body.session_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    logger.info(
        "POST /run — pipeline started",
        extra={
            "run_id": run_id,
            "session_id": session_id,
            "query_preview": query[:120],
        },
    )

    def _execute() -> RunResponse:
        with pack_cls(
            run_id=run_id,
            llm=_shared_llm,
            checkpointer=_shared_checkpointer,
            **_pack_runtime_kwargs(pack_cls),
        ) as pipeline:
            report = pipeline.run(query)

            cost_usd = getattr(pipeline, "cost_usd", None)

            if _shared_memory is not None:
                _shared_memory.save_run(
                    run_id=run_id,
                    query=query,
                    result=report.to_dict() if hasattr(report, "to_dict") else {},
                    metadata={"session_id": session_id, "agent": "MultiAgentGraph"},
                )

            return RunResponse.from_analysis_report(
                report, session_id=session_id, cost_usd=cost_usd
            )

    try:
        response = await _run_in_executor(_execute)
    except AgentValidationError as exc:
        logger.warning(
            "POST /run — validation error",
            extra={"run_id": run_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except AgentTimeoutError as exc:
        logger.error(
            "POST /run — pipeline timeout",
            extra={"run_id": run_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The agent pipeline exceeded its step budget. Try a simpler query.",
        ) from exc
    except AgentBudgetExceededError as exc:
        logger.warning(
            "POST /run — budget exceeded",
            extra={"run_id": run_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Run cost budget exceeded. Increase PACK_DEFAULT_BUDGET_USD or pass a higher budget.",
        ) from exc
    except AgentExecutionError as exc:
        logger.error(
            "POST /run — pipeline execution error",
            extra={"run_id": run_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The agent pipeline encountered an internal error.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "POST /run — unexpected error",
            extra={"run_id": run_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        ) from exc

    logger.info(
        "POST /run — pipeline completed",
        extra={
            "run_id": run_id,
            "session_id": session_id,
            "confidence": response.confidence,
            "insights_count": len(response.key_insights),
        },
    )
    return response


# ---------------------------------------------------------------------------
# SSE streaming helpers
# ---------------------------------------------------------------------------


async def _stream_pipeline(
    query: str,
    session_id: str,
    run_id: str,
    pack_cls: type[Any],
) -> AsyncGenerator[str, None]:
    """Async generator that streams the pipeline execution as SSE events.

    Streams via ``pack_cls``'s ``stream_events()`` (typically
    ``ResearchAnalysisPack`` / ``MultiAgentGraph`` alias).

    Event types emitted:

    * ``status``          — Progress message.
    * ``phase_started``   — A pipeline phase has begun executing.
    * ``phase_completed`` — A pipeline phase has finished.
    * ``token``           — An LLM token chunk (real-time streaming).
    * ``done``            — Final result with traceability metadata.
    * ``error``           — Terminal error event.

    Args:
        query: Validated user query string.
        session_id: Session identifier for memory persistence.
        run_id: Unique identifier for this pipeline run.

    Yields:
        SSE-formatted strings.
    """
    if active_pipelines is not None:
        active_pipelines.inc()
    try:
        yield f"data: {json.dumps({'type': 'status', 'message': 'Starting pipeline...'})}\n\n"

        llm = get_shared_llm()
        checkpointer = get_shared_checkpointer()

        pipeline = pack_cls(
            run_id=run_id,
            llm=llm,
            checkpointer=checkpointer,
            **_pack_runtime_kwargs(pack_cls),
        )
        report = None

        try:
            async for event in pipeline.stream_events(query):
                kind = event["type"]

                if kind in ("phase_started", "phase_completed", "token"):
                    yield f"data: {json.dumps(event)}\n\n"

                elif kind == "pipeline_completed":
                    report_raw = event.get("report")
                    if isinstance(report_raw, dict):
                        report = AnalysisReport(**report_raw)
                    else:
                        report = report_raw
        finally:
            pipeline.close()

        if report is None:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Pipeline completed without a report.'})}\n\n"
            return

        logger.info(
            "POST /run/stream — pipeline completed",
            extra={
                "run_id": run_id,
                "session_id": session_id,
                "confidence": report.confidence,
            },
        )

        if _shared_memory is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                _executor,
                functools.partial(
                    _shared_memory.save_run,
                    run_id=run_id,
                    query=query,
                    result=report.to_dict(),
                    metadata={"session_id": session_id, "agent": "stream_pipeline"},
                ),
            )

        done_payload = {
            "type": "done",
            "run_id": run_id,
            "session_id": session_id,
            "executive_summary": report.executive_summary,
            "key_insights": report.key_insights,
            "patterns": report.patterns,
            "implications": report.implications,
            "confidence": report.confidence,
            "research_summary": report.research_summary,
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    except AgentTimeoutError as exc:
        logger.error(
            "POST /run/stream — pipeline timeout",
            extra={"run_id": run_id, "error": str(exc)},
        )
        yield f"data: {json.dumps({'type': 'error', 'message': 'The pipeline timed out. Try a simpler query.'})}\n\n"
    except (AgentExecutionError, AgentValidationError) as exc:
        logger.error(
            "POST /run/stream — pipeline error",
            extra={"run_id": run_id, "error": str(exc)},
        )
        yield f"data: {json.dumps({'type': 'error', 'message': 'The pipeline encountered an error.'})}\n\n"
    except Exception as exc:
        logger.exception(
            "POST /run/stream — unexpected error",
            extra={"run_id": run_id, "error": str(exc)},
        )
        yield f"data: {json.dumps({'type': 'error', 'message': 'An unexpected error occurred.'})}\n\n"
    finally:
        if active_pipelines is not None:
            active_pipelines.dec()


@app.post(
    "/run/stream",
    status_code=status.HTTP_200_OK,
    tags=["Pipeline"],
    summary="Stream the full Research + Analysis pipeline as Server-Sent Events",
    response_description=(
        "A text/event-stream response emitting status, phase_started, "
        "phase_completed, token, done, and error SSE events in real time "
        "as the pipeline progresses."
    ),
)
async def run_stream(
    body: RunRequest,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    pack_cls: Annotated[type[Any], Depends(get_legacy_pack_cls)],
) -> StreamingResponse:
    """
    Execute the complete multi-agent pipeline and stream progress as SSE.

    The pipeline sequences two LangGraph agents:

    1. ``ResearchAgent``  — expands the query and produces a ``ResearchResult``.
    2. ``AnalystAgent``   — consumes the research and produces an ``AnalysisReport``.

    Events are streamed from ``pack_cls`` (same class as ``POST /run``), using
    its ``stream_events()`` and LangGraph's ``astream_events`` API under the hood.

    SSE event types
    ---------------
    * ``status``          — ``{"type": "status", "message": "…"}``
    * ``phase_started``   — ``{"type": "phase_started", "phase": "research"}``
    * ``phase_completed`` — ``{"type": "phase_completed", "phase": "research"}``
    * ``token``           — ``{"type": "token", "content": "…", "node": "analysis_node"}``
    * ``done``            — ``{"type": "done", "run_id": "…", "session_id": "…", "confidence": 0.87}``
    * ``error``           — ``{"type": "error", "message": "…"}``

    Args:
        body: Request body containing the ``query`` string and optional ``session_id``.
        request: The raw FastAPI ``Request`` (used for client metadata).

    Returns:
        A ``StreamingResponse`` with ``media_type="text/event-stream"``.

    Raises:
        422 Unprocessable Entity: When the request body fails schema validation.
        400 Bad Request: When the query is empty after stripping whitespace.
    """
    if _shutting_down.is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is shutting down.",
        )

    try:
        query = _validate_pack_query(settings.default_pack_id, body.query)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty.",
        )

    if get_shared_llm() is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider is not configured. Check server logs.",
        )

    session_id = body.session_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    stream_timeout = effective_stream_timeout_seconds(
        settings.default_pack_id, settings
    )

    logger.info(
        "POST /run/stream — pipeline started",
        extra={
            "run_id": run_id,
            "session_id": session_id,
            "query_preview": query[:120],
        },
    )

    async def _guarded_stream() -> AsyncGenerator[str, None]:
        try:
            async with asyncio.timeout(stream_timeout):
                async for event in _stream_pipeline(
                    query, session_id, run_id, pack_cls
                ):
                    yield event
        except TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Stream timed out after {stream_timeout}s'})}\n\n"

    return StreamingResponse(
        _guarded_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post(
    "/research",
    response_model=ResearchResponse,
    status_code=status.HTTP_200_OK,
    tags=["Pipeline"],
    summary="Run the Research-only pipeline",
    response_description="Structured ResearchResult produced by the ResearchAgent.",
)
async def run_research(
    body: ResearchRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ResearchResponse:
    """
    Execute only the research phase of the pipeline.

    The ``ResearchAgent`` expands the user query into focused sub-queries,
    retrieves information snippets, validates their quality (optionally
    looping for a second retrieval pass), and returns a structured
    ``ResearchResult`` with a summary, raw findings, and source references.

    Use this endpoint when you want research output without the downstream
    analysis step — for example, to feed custom post-processing logic or to
    inspect intermediate pipeline results.

    Args:
        body: Request body containing the ``query`` string.

    Returns:
        A ``ResearchResponse`` containing the summary, findings list, sources,
        and confidence score.

    Raises:
        422 Unprocessable Entity: When the request body fails validation.
        400 Bad Request: When the query is empty after stripping whitespace.
        402 Payment Required: When the run exceeds its configured USD cost budget.
        500 Internal Server Error: When the research agent fails.
        504 Gateway Timeout: When the agent exceeds its configured step budget.
    """
    if _shutting_down.is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is shutting down.",
        )

    try:
        query = _input_validator.validate(body.query)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty.",
        )

    if get_shared_llm() is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider is not configured. Check server logs.",
        )

    session_id = body.session_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    logger.info(
        "POST /research — started",
        extra={
            "run_id": run_id,
            "session_id": session_id,
            "query_preview": query[:120],
        },
    )

    def _execute() -> ResearchResponse:
        agent = ResearchAgent(
            thread_id=run_id,
            llm=_shared_llm,
            checkpointer=_shared_checkpointer,
        )
        result = agent.run_structured(query)
        cost_usd = getattr(agent, "cost_usd", None)
        # cost_usd captured here while agent is still in scope; do not move

        if _shared_memory is not None:
            _shared_memory.save_run(
                run_id=run_id,
                query=query,
                result=result.to_dict(),
                metadata={"session_id": session_id, "agent": "ResearchAgent"},
            )

        return ResearchResponse.from_research_result(
            result, session_id=session_id, cost_usd=cost_usd
        )

    try:
        response = await _run_in_executor(_execute)
    except AgentValidationError as exc:
        logger.warning(
            "POST /research — validation error",
            extra={"run_id": run_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except AgentTimeoutError as exc:
        logger.error(
            "POST /research — timeout",
            extra={"run_id": run_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The research agent exceeded its step budget. Try a simpler query.",
        ) from exc
    except AgentBudgetExceededError as exc:
        logger.warning(
            "POST /research — budget exceeded",
            extra={"run_id": run_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Run cost budget exceeded. Increase PACK_DEFAULT_BUDGET_USD or pass a higher budget.",
        ) from exc
    except AgentExecutionError as exc:
        logger.error(
            "POST /research — execution error",
            extra={"run_id": run_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The research agent encountered an internal error.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "POST /research — unexpected error",
            extra={"run_id": run_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        ) from exc

    logger.info(
        "POST /research — completed",
        extra={
            "run_id": run_id,
            "session_id": session_id,
            "confidence": response.confidence,
            "findings_count": len(response.findings),
        },
    )
    return response


@app.get(
    "/sessions/{session_id}/history",
    response_model=HistoryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Sessions"],
    summary="Retrieve run history for a session",
    response_description="Ordered list of run records associated with the given session ID.",
)
async def get_session_history(
    session_id: Annotated[
        str,
        FastAPIPath(
            min_length=1,
            max_length=200,
            description="Session identifier",
        ),
    ],
) -> HistoryResponse:
    """
    Return all run records associated with ``session_id``.

    Filters by ``session_id`` directly in SQL via ``json_extract`` so only
    matching rows are loaded.  Results are ordered newest-first.

    Args:
        session_id: The session identifier to look up (URL path parameter).

    Returns:
        A ``HistoryResponse`` with the matching entries and a total count.
    """
    mem = _shared_memory
    if mem is None:
        return HistoryResponse(session_id=session_id, entries=[], total=0)
    runs = await _run_in_executor(mem.list_runs_by_session, session_id)
    entries = [
        HistoryEntry(
            run_id=r["run_id"],
            query=r["query"],
            result_summary=str(r.get("result", {}) or "")[:200],
            created_at=r.get("created_at", ""),
            metadata=r.get("metadata", {}),
        )
        for r in runs
    ]

    logger.info(
        "GET /sessions/%s/history — %d entries returned",
        session_id,
        len(entries),
    )
    return HistoryResponse(session_id=session_id, entries=entries, total=len(entries))


@app.get(
    "/packs",
    summary="List registered domain packs",
    response_model=list[dict],  # type: ignore[type-arg]
    tags=["packs"],
)
async def list_packs() -> list[dict[str, Any]]:
    """Return all registered domain packs with their input/output JSON schemas.

    Useful for service discovery and generating client SDKs.
    """
    return PackRegistry.list_packs_with_metadata()


@app.get(
    "/packs/{pack_id}/versions",
    summary="List versions of a registered pack",
    tags=["packs"],
)
async def list_pack_versions(pack_id: str) -> list[dict[str, Any]]:
    """Return all registered versions for a pack with their current weights."""
    try:
        versions = PackRegistry._get_versions(pack_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Pack '{pack_id}' not found."
        )
    return [{"version": pv.version, "weight": pv.weight} for pv in versions]


@app.patch(
    "/packs/{pack_id}/versions/{version}/weight",
    summary="Update traffic-split weight for a pack version",
    tags=["packs"],
)
async def update_pack_version_weight(
    pack_id: str,
    version: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Set the traffic-split weight for a specific registered pack version."""
    weight = body.get("weight")
    if weight is None or not isinstance(weight, (int, float)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'weight' field (number) is required.",
        )
    try:
        PackRegistry.set_weights(pack_id, {version: float(weight)})
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return {"pack_id": pack_id, "version": version, "weight": float(weight)}
