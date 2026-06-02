"""api/middleware.py — All HTTP middlewares for the FastAPI application.

Registered in api/app.py.  Order matters: FastAPI executes middlewares in
reverse registration order (last-registered = outermost = first to execute).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

import api.state as state
from core.config import get_settings
from core.observability import (
    http_request_duration_seconds,
    http_requests_total,
    metrics_path_label,
    requests_rejected_during_shutdown,
)
from core.security import ensure_request_body_within_limit

logger = logging.getLogger(__name__)

_DRAIN_EXEMPT_PATHS = frozenset({"/health", "/ready", "/metrics"})
_AUTH_EXEMPT_PATHS = frozenset(
    {
        "/",
        "/health",
        "/ready",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
    }
)
_RATE_LIMIT_EXEMPT_PATHS = frozenset({"/health", "/ready", "/metrics"})


async def add_security_headers(request: Request, call_next: Any) -> Any:
    """Attach security-relevant HTTP response headers to every reply."""
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"

    if request.url.path in ("/docs", "/redoc", "/openapi.json"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' cdn.jsdelivr.net; "
            "img-src 'self' data: fastapi.tiangolo.com;"
        )
    else:
        response.headers["Content-Security-Policy"] = "default-src 'self'"

    if get_settings().environment == "production":
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )
    if "server" in response.headers:
        del response.headers["server"]
    return response


async def auth_middleware(request: Request, call_next: Any) -> Any:
    """Optional Bearer-token authentication gate.

    When API_KEY is set, every request to a non-exempt path must carry a
    matching Authorization: Bearer <token> header.
    """
    if request.url.path in _AUTH_EXEMPT_PATHS:
        return await call_next(request)

    import hmac

    api_key = get_settings().api_key
    if api_key is None:
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")
    token = (
        auth_header.removeprefix("Bearer ").strip()
        if auth_header.startswith("Bearer ")
        else ""
    )
    if not token or not hmac.compare_digest(token, api_key):
        from api.dependencies import _request_client_ip

        logger.warning(
            "Auth failed",
            extra={"path": request.url.path, "client": _request_client_ip(request)},
        )
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing Bearer token."},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


async def rate_limit_middleware(request: Request, call_next: Any) -> Any:
    """Enforce a sliding-window rate limit on all incoming requests."""
    if request.url.path in _RATE_LIMIT_EXEMPT_PATHS:
        return await call_next(request)

    from api.dependencies import _rate_limit_key

    limit_key = _rate_limit_key(request)
    if state.rate_limiter is not None and not state.rate_limiter.is_allowed(limit_key):
        from api.dependencies import _request_client_ip

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
                "Retry-After": str(
                    int(getattr(state.rate_limiter, "window_seconds", 60))
                )
            },
        )
    return await call_next(request)


async def log_requests(request: Request, call_next: Any) -> Any:
    """Structured access log for every HTTP request."""
    from core.observability import set_request_id

    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    set_request_id(request_id)
    start = time.monotonic()

    from api.dependencies import _request_client_ip

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
    logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(elapsed_s * 1000, 2),
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


async def drain_middleware(request: Request, call_next: Any) -> Any:
    """Reject new requests with 503 when the server is shutting down."""
    if state.shutting_down.is_set() and request.url.path not in _DRAIN_EXEMPT_PATHS:
        if requests_rejected_during_shutdown is not None:
            requests_rejected_during_shutdown.inc()
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Server is shutting down."},
        )
    return await call_next(request)


async def body_size_limit_middleware(request: Request, call_next: Any) -> Any:
    """Reject HTTP bodies above MAX_REQUEST_BODY_BYTES with 413."""
    from api.dependencies import _request_client_ip

    settings = get_settings()
    bounded_request, error = await ensure_request_body_within_limit(
        request, settings.max_request_body_bytes
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
