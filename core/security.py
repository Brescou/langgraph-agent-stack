"""
core/security.py — Centralised security utilities for the LangGraph agent stack.

This module provides four focused primitives that harden the application against
the most common attack vectors at the API layer and in structured logging:

``InputValidator``
    Validates and sanitises free-text queries.  Enforces a maximum byte length,
    rejects null bytes, and detects a configurable set of dangerous patterns
    (prompt injection, SSRF-style payloads, template injection, path traversal).

``RateLimiter``
    In-memory sliding-window rate limiter keyed by client IP.  Designed to be
    instantiated once at module level and called from FastAPI middleware.

``sanitize_log_data``
    Recursively masks sensitive values in log ``extra`` dicts so that API keys,
    tokens, and passwords are never emitted to log sinks in plaintext.

``validate_api_key_format``
    Checks that a string matches the Anthropic API key format (``sk-ant-``
    prefix) before it is handed to the SDK, catching common misconfiguration
    errors early.

All public functions and classes carry complete type hints and docstrings.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Any

# ---------------------------------------------------------------------------
# InputValidator
# ---------------------------------------------------------------------------

# Patterns that signal prompt-injection or server-side injection attempts.
# Each entry is a compiled regex.  Matching is case-insensitive.
_DANGEROUS_PATTERNS: list[re.Pattern[str]] = [
    # Prompt injection / jailbreak markers
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:acting\s+as|a\s+)", re.IGNORECASE),
    re.compile(r"</?(system|assistant|user|human|prompt)\s*/?>", re.IGNORECASE),
    # Server-Side Template Injection
    re.compile(r"\{\{.*?\}\}", re.IGNORECASE | re.DOTALL),
    re.compile(r"\{%.*?%\}", re.IGNORECASE | re.DOTALL),
    # SSRF / internal endpoint probing
    re.compile(
        r"https?://(?:169\.254\.169\.254|metadata\.google\.internal|localhost|127\.\d+\.\d+\.\d+|::1|0\.0\.0\.0)",
        re.IGNORECASE,
    ),
    # Path traversal
    re.compile(r"(?:\.\.[\\/]){2,}", re.IGNORECASE),
    # Null bytes
    re.compile(r"\x00"),
]

# Maximum query length in characters (mirrors the Pydantic model constraint;
# validated here as a defence-in-depth layer with a clear error message).
_DEFAULT_MAX_LENGTH: int = 2000


class InputValidator:
    """
    Validate and sanitise free-text query strings before they reach the LLM.

    The validator applies three checks in order:
    1. Length enforcement — rejects inputs that exceed ``max_length`` characters.
    2. Dangerous-pattern detection — rejects inputs that match any entry in the
       ``_DANGEROUS_PATTERNS`` list (prompt injection, SSRF, template injection,
       path traversal, null bytes).
    3. Sanitisation — strips leading/trailing whitespace and collapses runs of
       three or more consecutive newlines to two, preventing log-injection via
       newline flooding.

    Attributes:
        max_length: Maximum allowed character count for a query.

    Example::

        validator = InputValidator(max_length=2000)
        clean = validator.validate("Tell me about quantum computing")
        # Returns the sanitised string or raises ValueError
    """

    def __init__(self, max_length: int = _DEFAULT_MAX_LENGTH) -> None:
        """
        Initialise the validator.

        Args:
            max_length: Maximum number of characters allowed in a single query.
                        Must be a positive integer.

        Raises:
            ValueError: If ``max_length`` is not a positive integer.
        """
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length!r}")
        self.max_length = max_length

    def validate(self, query: str) -> str:
        """
        Validate and sanitise ``query``.

        Args:
            query: The raw input string from the API caller.

        Returns:
            The sanitised query string (whitespace-normalised).

        Raises:
            ValueError: When the query exceeds ``max_length`` or matches a
                        dangerous pattern.  The message is safe to surface to
                        API callers (it does not reveal internal implementation
                        details).
        """
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        if len(query) > self.max_length:
            raise ValueError(
                f"Query exceeds maximum length of {self.max_length} characters "
                f"(received {len(query)} characters)."
            )

        for pattern in _DANGEROUS_PATTERNS:
            if pattern.search(query):
                # Do not echo the matched content back to the caller.
                raise ValueError(
                    "Query contains disallowed content and cannot be processed."
                )

        # Sanitise: strip surrounding whitespace, collapse excessive newlines
        sanitised = query.strip()
        sanitised = re.sub(r"\n{3,}", "\n\n", sanitised)
        return sanitised


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """
    Simple in-memory sliding-window rate limiter keyed by client IP.

    Uses a ``deque`` per IP to track the timestamps of recent requests within
    the rolling window.  Timestamps older than ``window_seconds`` are pruned on
    every check, so memory usage stays bounded even under sustained load from a
    single IP.

    Thread-safe via a per-instance ``threading.Lock``.

    Attributes:
        max_requests: Maximum number of requests allowed per ``window_seconds``.
        window_seconds: Length of the sliding window in seconds.

    Example::

        limiter = RateLimiter(max_requests=60, window_seconds=60)

        @app.middleware("http")
        async def rate_limit(request: Request, call_next):
            ip = request.client.host if request.client else "unknown"
            if not limiter.is_allowed(ip):
                raise HTTPException(status_code=429, detail="Rate limit exceeded.")
            return await call_next(request)
    """

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: float = 60.0,
    ) -> None:
        """
        Initialise the rate limiter.

        Args:
            max_requests: Maximum requests permitted per IP within the window.
                          Must be >= 1.
            window_seconds: Sliding window duration in seconds.  Must be > 0.

        Raises:
            ValueError: If ``max_requests`` < 1 or ``window_seconds`` <= 0.
        """
        if max_requests < 1:
            raise ValueError(f"max_requests must be >= 1, got {max_requests!r}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds!r}")

        self.max_requests = max_requests
        self.window_seconds = window_seconds

        # ip -> deque of request timestamps (float, monotonic)
        self._buckets: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, ip: str) -> bool:
        """
        Check whether a new request from ``ip`` is within the rate limit.

        Records the current timestamp when the request is allowed.  Does not
        record anything when the request is denied.

        Args:
            ip: The client IP address string used as the rate-limit key.
                Pass ``"unknown"`` when the IP cannot be determined; all
                unknown callers share a single bucket.

        Returns:
            ``True`` if the request is allowed, ``False`` if it is rate-limited.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            bucket = self._buckets.setdefault(ip, deque())

            # Prune timestamps outside the sliding window
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= self.max_requests:
                return False

            bucket.append(now)
            return True

    def remaining(self, ip: str) -> int:
        """
        Return the number of requests remaining for ``ip`` in the current window.

        This is a read-only query — it does not record a new timestamp.

        Args:
            ip: The client IP address string.

        Returns:
            Number of requests remaining (0 when at the limit).
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            bucket = self._buckets.get(ip, deque())
            active = sum(1 for ts in bucket if ts >= cutoff)
            return max(0, self.max_requests - active)


# ---------------------------------------------------------------------------
# sanitize_log_data
# ---------------------------------------------------------------------------

# Key fragments (case-insensitive) whose values should be masked in log output.
_SENSITIVE_KEY_FRAGMENTS: frozenset[str] = frozenset(
    {"key", "token", "secret", "password", "passwd", "pwd", "credential", "auth"}
)

_MASK = "***REDACTED***"


def sanitize_log_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of ``data`` with sensitive values masked.

    Recursively traverses nested dicts.  Any key whose lowercased name
    contains one of the fragments in ``_SENSITIVE_KEY_FRAGMENTS`` has its
    value replaced with ``"***REDACTED***"``.

    This function is a pure transformation — it never mutates the input dict.

    Args:
        data: A dictionary of log ``extra`` fields (or any string-keyed dict).

    Returns:
        A new dictionary with identical structure but sensitive values masked.

    Example::

        raw = {"user": "alice", "api_key": "sk-ant-...", "query": "hello"}
        safe = sanitize_log_data(raw)
        # {"user": "alice", "api_key": "***REDACTED***", "query": "hello"}
    """
    sanitised: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            sanitised[k] = sanitize_log_data(v)
        elif _is_sensitive_key(k):
            sanitised[k] = _MASK
        else:
            sanitised[k] = v
    return sanitised


def _is_sensitive_key(key: str) -> bool:
    """Return True if ``key`` (case-insensitive) contains a sensitive fragment."""
    lower = key.lower()
    return any(fragment in lower for fragment in _SENSITIVE_KEY_FRAGMENTS)


# ---------------------------------------------------------------------------
# validate_api_key_format
# ---------------------------------------------------------------------------

# Anthropic API keys begin with "sk-ant-" followed by alphanumeric characters,
# hyphens, and underscores.  The minimum length after the prefix is 10 chars.
_ANTHROPIC_KEY_PATTERN: re.Pattern[str] = re.compile(r"^sk-ant-[A-Za-z0-9\-_]{10,}$")


def validate_api_key_format(key: str) -> bool:
    """
    Check whether ``key`` matches the expected Anthropic API key format.

    The expected format is: ``sk-ant-`` followed by at least 10 alphanumeric
    characters, hyphens, or underscores.

    This is a structural check only — it does not verify the key against the
    Anthropic API.  Use it at startup to catch copy-paste errors and unset
    placeholder values before the first real API call is attempted.

    Args:
        key: The API key string to check.

    Returns:
        ``True`` if ``key`` matches the expected format, ``False`` otherwise.

    Example::

        if not validate_api_key_format(settings.anthropic_api_key):
            raise RuntimeError("ANTHROPIC_API_KEY does not match expected format.")
    """
    if not isinstance(key, str):
        return False
    return bool(_ANTHROPIC_KEY_PATTERN.match(key))
