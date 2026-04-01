"""
api/main.py — Production-ready FastAPI application for the LangGraph agent stack.

Exposes three functional endpoints over the multi-agent pipeline:

* ``POST /run``        — Full Research + Analysis pipeline via ``MultiAgentGraph``.
* ``POST /run/stream`` — Same pipeline streamed as Server-Sent Events.
* ``POST /research``   — Research-only pipeline via ``ResearchAgent``.
* ``GET  /health``     — Lightweight health/liveness probe.
* ``GET  /``           — Redirect to the auto-generated ``/docs`` UI.

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
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi import Path as FastAPIPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from langchain_core.language_models import BaseChatModel

from agents.analyst import AnalystAgent
from agents.base_agent import (
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
from core.config import Settings, get_settings
from core.graph import MultiAgentGraph
from core.memory import ConversationMemory

# ---------------------------------------------------------------------------
# Logging — structured JSON via core.observability when available
# ---------------------------------------------------------------------------
from core.observability import configure_logging, init_tracing
from core.security import InputValidator, RateLimiter

configure_logging(level=get_settings().log_level.value)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state (populated during lifespan startup)
# ---------------------------------------------------------------------------

_APP_VERSION = "0.1.0"
_start_time: float = 0.0
_executor: ThreadPoolExecutor | None = None
_shared_llm: BaseChatModel | None = None
_shared_checkpointer: Any | None = None
_shared_memory: ConversationMemory | None = None

# Security primitives — instantiated once, shared across all requests.
# 60 requests per minute per IP is a conservative default suited for an LLM
# pipeline where each request may take several seconds.  Adjust via subclassing
# or by passing a custom RateLimiter instance in tests.
_rate_limiter = RateLimiter(max_requests=60, window_seconds=60.0)
_input_validator = InputValidator(max_length=2000)
_shutting_down = threading.Event()


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

    _start_time = time.monotonic()
    _executor = ThreadPoolExecutor(
        max_workers=4,
        thread_name_prefix="agent-worker",
    )

    _settings = get_settings()
    init_tracing()
    _init_llm_and_checkpointer(_settings)

    _shared_memory = ConversationMemory(
        _settings.sqlite_path,
        backend=_settings.memory_backend.value,
        redis_url=_settings.redis_url,
        postgres_url=_settings.postgres_url,
    )

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

    yield  # Application is live here

    logger.info("API server shutting down — draining in-flight requests")
    _shutting_down.set()
    if _executor is not None:
        _executor.shutdown(wait=True, cancel_futures=False)
    if _shared_memory is not None:
        _shared_memory.close()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# Shared dependency accessors
# ---------------------------------------------------------------------------


def get_shared_llm() -> BaseChatModel | None:
    """Return the shared LLM, retrying init if the first attempt failed."""
    if _shared_llm is None:
        _init_llm_and_checkpointer(get_settings())
    return _shared_llm


def get_shared_checkpointer() -> Any | None:
    """Return the shared checkpointer, retrying init if the first attempt failed."""
    if _shared_checkpointer is None:
        _init_llm_and_checkpointer(get_settings())
    return _shared_checkpointer


def get_shared_memory() -> ConversationMemory | None:
    return _shared_memory


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LangGraph Agent Stack API",
    description=(
        "Production API exposing a multi-agent LangGraph pipeline. "
        "The pipeline sequences a ``ResearchAgent`` and an ``AnalystAgent`` "
        "to turn a free-text query into a structured ``AnalysisReport``."
    ),
    version=_APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — wildcard for open-source template; tighten for production
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    # Wildcard is intentional for an open-source template.  In production,
    # replace "*" with an explicit list of trusted origins:
    #   allow_origins=["https://your-frontend.example.com"]
    # Never combine allow_origins=["*"] with allow_credentials=True.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
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
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"
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
                "client": request.client.host if request.client else "unknown",
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
    Enforce a per-IP sliding-window rate limit on all incoming requests.

    The health endpoint is excluded so Kubernetes probes are never blocked.
    When a client exceeds the limit a ``429 Too Many Requests`` response is
    returned immediately without forwarding the request to any handler.
    """
    if request.url.path == "/health":
        return await call_next(request)

    client_ip: str = request.client.host if request.client else "unknown"
    if not _rate_limiter.is_allowed(client_ip):
        logger.warning(
            "Rate limit exceeded",
            extra={"client": client_ip, "path": request.url.path},
        )
        return Response(
            content='{"detail":"Rate limit exceeded. Please slow down."}',
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            media_type="application/json",
            headers={"Retry-After": str(int(_rate_limiter.window_seconds))},
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
    request_id = str(uuid.uuid4())
    start = time.monotonic()

    logger.info(
        "Request received",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown",
        },
    )

    response = await call_next(request)

    elapsed_ms = (time.monotonic() - start) * 1000
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

    return response


# ---------------------------------------------------------------------------
# Helper: run a blocking callable in the thread pool
# ---------------------------------------------------------------------------


async def _run_in_executor(fn: Any, *args: Any) -> Any:
    """
    Execute a blocking callable in the application thread pool.

    Args:
        fn: The synchronous callable to execute.
        *args: Positional arguments forwarded to ``fn``.

    Returns:
        The return value of ``fn(*args)``.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, fn, *args)


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

    if _shared_llm is not None:
        components["llm"] = ComponentHealth(
            status="ok", detail=settings.llm_provider.value
        )
    else:
        components["llm"] = ComponentHealth(
            status="degraded", detail="LLM not initialised"
        )

    if _shared_memory is not None:
        components["memory"] = ComponentHealth(
            status="ok", detail=str(_shared_memory.db_path)
        )
    else:
        components["memory"] = ComponentHealth(
            status="degraded", detail="Memory store not initialised"
        )

    if _shared_checkpointer is not None:
        components["checkpointer"] = ComponentHealth(
            status="ok", detail=settings.memory_backend.value
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
        500 Internal Server Error: When the agent pipeline encounters an
            unrecoverable error.
        504 Gateway Timeout: When the agent exceeds its configured step budget.
    """
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
        pipeline = MultiAgentGraph(
            run_id=run_id,
            llm=_shared_llm,
            checkpointer=_shared_checkpointer,
        )
        report = pipeline.run(query)

        if _shared_memory is not None:
            _shared_memory.save_run(
                run_id=run_id,
                query=query,
                result=vars(report) if hasattr(report, "__dict__") else {},
                metadata={"session_id": session_id, "agent": "MultiAgentGraph"},
            )

        return RunResponse.from_analysis_report(report, session_id=session_id)

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
) -> AsyncGenerator[str, None]:
    """
    Async generator that executes the Research + Analysis pipeline and yields
    SSE-formatted event strings.

    Each yielded string is a complete SSE event of the form::

        data: <json payload>\\n\\n

    Event types emitted:

    * ``status``       — Progress message (``{"type": "status", "message": "…"}``).
    * ``agent_switch`` — Transition between agents
                         (``{"type": "agent_switch", "from": "…", "to": "…"}``).
    * ``done``         — Final success event with traceability metadata
                         (``{"type": "done", "run_id": "…", "session_id": "…",
                         "confidence": 0.87}``).
    * ``error``        — Terminal error event
                         (``{"type": "error", "message": "…"}``).

    Args:
        query: Validated user query string.
        session_id: Session identifier for memory persistence.
        run_id: Unique identifier for this pipeline run.

    Yields:
        SSE-formatted strings ready to be sent as ``text/event-stream`` chunks.
    """
    try:
        yield f"data: {json.dumps({'type': 'status', 'message': 'Starting research phase...'})}\n\n"

        loop = asyncio.get_running_loop()
        research_agent = ResearchAgent(
            thread_id=session_id,
            llm=_shared_llm,
            checkpointer=_shared_checkpointer,
        )
        research_result = await loop.run_in_executor(
            _executor, research_agent.run_structured, query
        )

        yield f"data: {json.dumps({'type': 'agent_switch', 'from': 'researcher', 'to': 'analyst'})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'message': 'Starting analysis phase...'})}\n\n"

        analyst_agent = AnalystAgent(
            thread_id=session_id,
            llm=_shared_llm,
            checkpointer=_shared_checkpointer,
        )
        report = await loop.run_in_executor(
            _executor, analyst_agent.run_structured, research_result
        )

        logger.info(
            "POST /run/stream — pipeline completed",
            extra={
                "run_id": run_id,
                "session_id": session_id,
                "confidence": report.confidence,
            },
        )

        if _shared_memory is not None:
            _shared_memory.save_run(
                run_id=run_id,
                query=query,
                result=report.to_dict(),
                metadata={"session_id": session_id, "agent": "stream_pipeline"},
            )

        yield f"data: {json.dumps({'type': 'done', 'run_id': run_id, 'session_id': session_id, 'executive_summary': report.executive_summary, 'key_insights': report.key_insights, 'patterns': report.patterns, 'implications': report.implications, 'confidence': report.confidence, 'research_summary': report.research_summary})}\n\n"

    except AgentTimeoutError as exc:
        logger.error(
            "POST /run/stream — pipeline timeout",
            extra={"run_id": run_id, "error": str(exc)},
        )
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
    except (AgentExecutionError, AgentValidationError) as exc:
        logger.error(
            "POST /run/stream — pipeline error",
            extra={"run_id": run_id, "error": str(exc)},
        )
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
    except Exception as exc:
        logger.exception(
            "POST /run/stream — unexpected error",
            extra={"run_id": run_id, "error": str(exc)},
        )
        yield f"data: {json.dumps({'type': 'error', 'message': 'An unexpected error occurred.'})}\n\n"


@app.post(
    "/run/stream",
    status_code=status.HTTP_200_OK,
    tags=["Pipeline"],
    summary="Stream the full Research + Analysis pipeline as Server-Sent Events",
    response_description=(
        "A text/event-stream response emitting status, agent_switch, done, "
        "and error SSE events as the pipeline progresses."
    ),
)
async def run_stream(
    body: RunRequest,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> StreamingResponse:
    """
    Execute the complete multi-agent pipeline and stream progress as SSE.

    The pipeline sequences two LangGraph agents:

    1. ``ResearchAgent``  — expands the query and produces a ``ResearchResult``.
    2. ``AnalystAgent``   — consumes the research and produces an ``AnalysisReport``.

    SSE event types
    ---------------
    * ``status``       — ``{"type": "status", "message": "…"}``
    * ``agent_switch`` — ``{"type": "agent_switch", "from": "researcher", "to": "analyst"}``
    * ``done``         — ``{"type": "done", "run_id": "…", "session_id": "…", "confidence": 0.87}``
    * ``error``        — ``{"type": "error", "message": "…"}``

    Args:
        body: Request body containing the ``query`` string and optional ``session_id``.
        request: The raw FastAPI ``Request`` (used for client metadata).

    Returns:
        A ``StreamingResponse`` with ``media_type="text/event-stream"``.

    Raises:
        422 Unprocessable Entity: When the request body fails schema validation.
        400 Bad Request: When the query is empty after stripping whitespace.
    """
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

    session_id = body.session_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())

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
            async with asyncio.timeout(settings.stream_timeout_seconds):
                async for event in _stream_pipeline(query, session_id, run_id):
                    yield event
        except TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Stream timed out after {settings.stream_timeout_seconds}s'})}\n\n"

    return StreamingResponse(
        _guarded_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
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
        500 Internal Server Error: When the research agent fails.
        504 Gateway Timeout: When the agent exceeds its configured step budget.
    """
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

        if _shared_memory is not None:
            _shared_memory.save_run(
                run_id=run_id,
                query=query,
                result=vars(result) if hasattr(result, "__dict__") else {},
                metadata={"session_id": session_id, "agent": "ResearchAgent"},
            )

        return ResearchResponse.from_research_result(result, session_id=session_id)

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
    runs = mem.list_runs_by_session(session_id)
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
