"""
agents/llm_retry.py — Retry predicates for transient LLM / HTTP errors.

Collects optional SDK exception types (Anthropic, OpenAI, httpx, Google) without
requiring every provider extra at import time.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# HTTP statuses that should be retried when surfaced as APIStatusError-like objects.
_TRANSIENT_HTTP_STATUS_CODES: frozenset[int] = frozenset(
    {408, 429, 500, 502, 503, 504, 529}
)

# HTTP statuses that indicate a credential problem (never retryable,
# never silently degradable — the caller must surface these).
_AUTH_HTTP_STATUS_CODES: frozenset[int] = frozenset({401, 403})


@lru_cache(maxsize=1)
def retryable_llm_exception_types() -> tuple[type[BaseException], ...]:
    """Return exception classes that indicate a transient LLM call failure."""
    types: set[type[BaseException]] = {
        TimeoutError,
        ConnectionError,
    }

    try:
        import httpx

        types.update(
            (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.NetworkError,
                httpx.ReadTimeout,
                httpx.WriteTimeout,
            )
        )
    except ImportError:
        pass

    try:
        import anthropic

        types.update(
            (
                anthropic.RateLimitError,
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            )
        )
    except ImportError:
        pass

    try:
        import openai

        types.update(
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            )
        )
    except ImportError:
        pass

    try:
        from google.api_core import exceptions as google_exceptions

        types.update(
            (
                google_exceptions.ServiceUnavailable,
                google_exceptions.DeadlineExceeded,
                google_exceptions.TooManyRequests,
                google_exceptions.InternalServerError,
            )
        )
    except ImportError:
        pass

    return tuple(types)


def _http_status_code(exc: BaseException) -> int | None:
    """Extract an HTTP status code from SDK exception shapes."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    if response is not None:
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status
    return None


@lru_cache(maxsize=1)
def auth_llm_exception_types() -> tuple[type[BaseException], ...]:
    """Return exception classes that indicate an authentication / permission failure."""
    types: set[type[BaseException]] = set()

    try:
        import anthropic

        types.update((anthropic.AuthenticationError, anthropic.PermissionDeniedError))
    except ImportError:
        pass

    try:
        import openai

        types.update((openai.AuthenticationError, openai.PermissionDeniedError))
    except ImportError:
        pass

    try:
        from google.api_core import exceptions as google_exceptions

        types.update(
            (google_exceptions.Unauthenticated, google_exceptions.PermissionDenied)
        )
    except ImportError:
        pass

    return tuple(types)


def is_auth_llm_error(exc: BaseException) -> bool:
    """Return True when ``exc`` represents an authentication / API-key failure.

    Matches typed provider SDK exceptions (Anthropic, OpenAI, Google) as well
    as any exception carrying an HTTP 401/403 status code.
    """
    if isinstance(exc, auth_llm_exception_types()):
        return True
    status = _http_status_code(exc)
    return status is not None and status in _AUTH_HTTP_STATUS_CODES


def find_auth_cause(exc: BaseException, max_depth: int = 10) -> BaseException | None:
    """Walk the ``__cause__`` / ``__context__`` chain looking for an auth failure.

    Packs (including third-party plugins) may wrap a provider 401/403 into a
    generic ``AgentExecutionError``; this lets the API layer still surface an
    actionable HTTP 502 instead of a generic 500.
    """
    current: BaseException | None = exc
    for _ in range(max_depth):
        if current is None:
            return None
        if is_auth_llm_error(current):
            return current
        current = current.__cause__ or current.__context__
    return None


def is_retryable_llm_error(exc: BaseException) -> bool:
    """Return True when ``exc`` represents a transient LLM / upstream failure."""
    if isinstance(exc, retryable_llm_exception_types()):
        return True
    status = _http_status_code(exc)
    return status is not None and status in _TRANSIENT_HTTP_STATUS_CODES


def retry_if_transient_llm_error(exc: BaseException) -> bool:
    """Tenacity ``retry_if_exception`` predicate (excludes budget errors)."""
    from core.cost import BudgetExceededError

    if isinstance(exc, BudgetExceededError):
        return False
    return is_retryable_llm_error(exc)


def record_retry_attempt(provider: str, outcome: str) -> None:
    """Increment ``llm_retry_attempts_total{provider, outcome}`` (no-op without Prometheus).

    Args:
        provider: Low-cardinality LLM provider name (e.g. ``"anthropic"``).
        outcome: ``"success"`` when a retry is performed after a transient
            error, ``"exhausted"`` when the retry budget is consumed without
            recovery.
    """
    from core.observability import llm_retry_attempts_total

    if llm_retry_attempts_total is not None:
        llm_retry_attempts_total.labels(provider=provider, outcome=outcome).inc()


def record_retry_exhausted(provider: str) -> None:
    """Record that all retries were exhausted for ``provider``."""
    record_retry_attempt(provider, "exhausted")


def before_sleep_log_transient_error(
    logger: object, provider: str | None = None
) -> Callable[..., None]:
    """Build a tenacity ``before_sleep`` callback that logs retry attempts.

    When ``provider`` is given, each performed retry also increments
    ``llm_retry_attempts_total{provider, outcome="success"}``.
    """

    def _before_sleep(retry_state: object) -> None:
        if provider is not None:
            record_retry_attempt(provider, "success")
        outcome = getattr(retry_state, "outcome", None)
        exc = outcome.exception() if outcome is not None else None
        attempt = getattr(retry_state, "attempt_number", 0)
        stop = getattr(retry_state, "retry_object", None)
        stop_after = getattr(getattr(stop, "stop", None), "max_attempt_number", None)
        max_attempts = stop_after if stop_after is not None else "?"
        getattr(logger, "warning")(
            "LLM call failed (attempt %s/%s), retrying: %s",
            attempt,
            max_attempts,
            exc,
            extra={"error": str(exc) if exc is not None else ""},
        )

    return _before_sleep
