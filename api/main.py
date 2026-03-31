"""
api/main.py — Production-ready FastAPI application for the LangGraph agent stack.

Exposes three functional endpoints over the multi-agent pipeline:

* ``POST /run``       — Full Research + Analysis pipeline via ``MultiAgentGraph``.
* ``POST /research``  — Research-only pipeline via ``ResearchAgent``.
* ``GET  /health``    — Lightweight health/liveness probe.
* ``GET  /``          — Redirect to the auto-generated ``/docs`` UI.

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
import logging
import time
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.models import (
    HealthResponse,
    ResearchRequest,
    ResearchResponse,
    RunRequest,
    RunResponse,
)
from agents.base_agent import AgentExecutionError, AgentTimeoutError, AgentValidationError
from core.config import settings
from core.graph import MultiAgentGraph
from core.security import InputValidator, RateLimiter, validate_api_key_format
from agents.researcher import ResearchAgent

# ---------------------------------------------------------------------------
# Logging — structured JSON-friendly via ``extra`` dicts
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=settings.log_level.value,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state (populated during lifespan startup)
# ---------------------------------------------------------------------------

_APP_VERSION = "0.1.0"
_start_time: float = 0.0
_executor: ThreadPoolExecutor | None = None

# Security primitives — instantiated once, shared across all requests.
# 60 requests per minute per IP is a conservative default suited for an LLM
# pipeline where each request may take several seconds.  Adjust via subclassing
# or by passing a custom RateLimiter instance in tests.
_rate_limiter = RateLimiter(max_requests=60, window_seconds=60.0)
_input_validator = InputValidator(max_length=2000)


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
    global _start_time, _executor

    _start_time = time.monotonic()
    _executor = ThreadPoolExecutor(
        max_workers=4,
        thread_name_prefix="agent-worker",
    )

    # Validate the Anthropic API key format early so a misconfigured key
    # (e.g. an unset placeholder) produces a clear startup error rather than
    # a cryptic Anthropic SDK exception on the first real request.
    if not validate_api_key_format(settings.anthropic_api_key):
        logger.warning(
            "ANTHROPIC_API_KEY does not match the expected 'sk-ant-...' format — "
            "verify the key is set correctly before sending requests."
        )

    logger.info(
        "API server starting up",
        extra={
            "version": _APP_VERSION,
            "environment": settings.environment,
            "host": settings.api_host,
            "port": settings.api_port,
            "model": settings.model_name,
            "memory_backend": settings.memory_backend.value,
        },
    )

    yield  # Application is live here

    logger.info("API server shutting down — draining thread pool")
    if _executor is not None:
        _executor.shutdown(wait=True)
    logger.info("Shutdown complete")


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
# CORS — origins from settings; never hard-coded
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
async def health() -> HealthResponse:
    """
    Return the current health status of the service.

    This endpoint is designed to be polled by load balancers and container
    orchestration platforms (Kubernetes liveness / readiness probes).  It
    performs no I/O and responds immediately.

    Returns:
        ``HealthResponse`` with status, version, uptime, and environment.
    """
    return HealthResponse(
        status="ok",
        version=_APP_VERSION,
        uptime_seconds=round(time.monotonic() - _start_time, 3),
        environment=settings.environment,
    )


@app.post(
    "/run",
    response_model=RunResponse,
    status_code=status.HTTP_200_OK,
    tags=["Pipeline"],
    summary="Run the full Research + Analysis pipeline",
    response_description="Structured AnalysisReport produced by the full agent pipeline.",
)
async def run_pipeline(body: RunRequest) -> RunResponse:
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

    run_id = str(uuid.uuid4())
    logger.info(
        "POST /run — pipeline started",
        extra={"run_id": run_id, "query_preview": query[:120]},
    )

    def _execute() -> RunResponse:
        pipeline = MultiAgentGraph(run_id=run_id)
        report = pipeline.run(query)
        return RunResponse.from_analysis_report(report)

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
            "confidence": response.confidence,
            "insights_count": len(response.key_insights),
        },
    )
    return response


@app.post(
    "/research",
    response_model=ResearchResponse,
    status_code=status.HTTP_200_OK,
    tags=["Pipeline"],
    summary="Run the Research-only pipeline",
    response_description="Structured ResearchResult produced by the ResearchAgent.",
)
async def run_research(body: ResearchRequest) -> ResearchResponse:
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

    run_id = str(uuid.uuid4())
    logger.info(
        "POST /research — started",
        extra={"run_id": run_id, "query_preview": query[:120]},
    )

    def _execute() -> ResearchResponse:
        agent = ResearchAgent(thread_id=run_id)
        result = agent.run_structured(query)
        return ResearchResponse.from_research_result(result)

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
            "confidence": response.confidence,
            "findings_count": len(response.findings),
        },
    )
    return response
