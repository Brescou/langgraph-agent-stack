"""
core/security.py — Centralised security utilities for the LangGraph agent stack.

This module provides four focused primitives that harden the application against
the most common attack vectors at the API layer and in structured logging:

``InputValidator``
    Validates and sanitises free-text queries.  Enforces a maximum character
    length, rejects null bytes, and normalises whitespace.  Prompt-injection
    mitigation belongs in ``domain_packs/common/prompt_safety.py`` (delimiter
    wrapping), not regex blocking of user text.

``validate_outbound_url``
    SSRF guard for URLs passed to ``httpx`` (connector fetches).  Blocks
    loopback, link-local, metadata, and private IP targets — including hostnames
    whose DNS records resolve to those ranges.

``RateLimiter``
    In-memory sliding-window rate limiter keyed by client IP.  Designed to be
    instantiated once at module level and called from FastAPI middleware.

``sanitize_log_data``
    Recursively masks sensitive values in log ``extra`` dicts so that API keys,
    tokens, and passwords are never emitted to log sinks in plaintext.

``validate_api_key_format``
    Checks that a string matches the expected API key format for the given
    LLM provider before it is handed to the SDK, catching common
    misconfiguration errors early.  Supports Anthropic, OpenAI, and a
    generic fallback for other providers.

``ensure_request_body_within_limit``
    Reads (or rejects) HTTP request bodies before route handlers parse JSON.
    Checks ``Content-Length`` and performs a stream-bounded read for chunked
    uploads so oversized payloads are rejected with 413 without buffering
    multi-megabyte bodies in memory.

All public functions and classes carry complete type hints and docstrings.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import re
import socket
import threading
import time
from collections import deque
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Literal, Protocol, cast, runtime_checkable
from urllib.parse import urlparse

from starlette.requests import Request

logger = logging.getLogger(__name__)

# Default cap for inbound JSON API bodies (defence against multi-MB payloads).
DEFAULT_MAX_REQUEST_BODY_BYTES: int = 1_048_576  # 1 MiB

_BODY_METHODS: frozenset[str] = frozenset({"POST", "PUT", "PATCH"})

# ---------------------------------------------------------------------------
# InputValidator
# ---------------------------------------------------------------------------

# Maximum query length in characters (mirrors the Pydantic model constraint;
# validated here as a defence-in-depth layer with a clear error message).
_DEFAULT_MAX_LENGTH: int = 2000

# Outbound HTTP(S) targets that must never be fetched by connectors.
_BLOCKED_OUTBOUND_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "metadata.google.internal",
        "metadata.google",
    }
)


class InputValidator:
    """
    Validate and sanitise free-text query strings before they reach the LLM.

    The validator applies three checks in order:
    1. Length enforcement — rejects inputs that exceed ``max_length`` characters.
    2. Null-byte rejection — rejects embedded ``\\x00`` (Pydantic also blocks
       these on model fields; kept here for defence in depth on raw strings).
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
            ValueError: When the query exceeds ``max_length``, contains a null
                        byte, or is not a string.  Messages are safe to surface
                        to API callers.
        """
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        if len(query) > self.max_length:
            raise ValueError(
                f"Query exceeds maximum length of {self.max_length} characters "
                f"(received {len(query)} characters)."
            )

        return self.check_content_safety(query, max_length=self.max_length)

    def check_content_safety(self, text: str, *, max_length: int | None = None) -> str:
        """Enforce length and null-byte rules; normalise whitespace.

        Used for per-field validation on typed pack bodies where each field may
        have a different Pydantic ``max_length`` than the global query cap.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")

        limit = self.max_length if max_length is None else max_length
        if limit is not None and len(text) > limit:
            raise ValueError(
                f"Input exceeds maximum length of {limit} characters "
                f"(received {len(text)} characters)."
            )

        if "\x00" in text:
            raise ValueError("Input contains disallowed null bytes.")

        sanitised = text.strip()
        sanitised = re.sub(r"\n{3,}", "\n\n", sanitised)
        return sanitised


def _is_blocked_outbound_ip(
    addr: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    """Return True when *addr* must not be used as an outbound fetch target."""
    return (
        addr.is_loopback
        or addr.is_private
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _resolve_outbound_host_ips(
    host: str,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    """Resolve *host* to IP addresses for SSRF validation."""
    try:
        infos = socket.getaddrinfo(
            host,
            None,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError(
            f"Outbound URL hostname could not be resolved: {host!r}"
        ) from exc

    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    seen: set[str] = set()
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip_str = str(sockaddr[0])
        if ip_str in seen:
            continue
        seen.add(ip_str)
        try:
            addresses.append(ipaddress.ip_address(ip_str))
        except ValueError:
            continue

    if not addresses:
        raise ValueError(f"Outbound URL hostname could not be resolved: {host!r}")

    return tuple(addresses)


def validate_outbound_url(url: str) -> str:
    """Reject outbound HTTP(S) URLs that target internal or metadata endpoints.

    Call this on every URL immediately before ``httpx`` fetches — not on free-text
    LLM queries, which are never used as fetch targets.

    Args:
        url: Fully resolved URL string (scheme, host, path).

    Returns:
        The input ``url`` unchanged when allowed.

    Raises:
        ValueError: When the scheme is not ``http``/``https``, the host is missing,
                    or the host resolves to a blocked target.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Outbound URL scheme not allowed: {parsed.scheme!r}")

    host = parsed.hostname
    if not host:
        raise ValueError("Outbound URL must include a hostname.")

    host_lower = host.lower().rstrip(".")
    if host_lower in _BLOCKED_OUTBOUND_HOSTNAMES:
        raise ValueError(f"Outbound URL target not allowed: {host!r}")

    bare = host_lower.strip("[]")
    try:
        addresses = (ipaddress.ip_address(bare),)
    except ValueError:
        addresses = _resolve_outbound_host_ips(host_lower)

    for addr in addresses:
        if _is_blocked_outbound_ip(addr):
            raise ValueError(f"Outbound URL target not allowed: {host!r}")

    return url


# ---------------------------------------------------------------------------
# Request body size limit
# ---------------------------------------------------------------------------


def _parse_content_length(content_length: str | None) -> int | None:
    """Return declared body size, or ``None`` when the header is absent."""
    if content_length is None or content_length.strip() == "":
        return None
    try:
        return int(content_length)
    except ValueError:
        raise ValueError("Invalid Content-Length header.") from None


async def ensure_request_body_within_limit(
    request: Request,
    max_bytes: int,
) -> tuple[Request | None, str | None]:
    """Bound inbound body size before FastAPI/Pydantic JSON parsing.

    Checks ``Content-Length`` first, then reads the body with a streaming cap
    for chunked uploads.  Returns a new :class:`Request` whose body can be read
    again by downstream handlers.

    Args:
        request: Incoming Starlette request.
        max_bytes: Maximum allowed body size in bytes.

    Returns:
        ``(request, None)`` when within limits, or ``(None, detail)`` when the
        body must be rejected (caller should return HTTP 413).
    """
    if request.method not in _BODY_METHODS:
        return request, None

    try:
        declared = _parse_content_length(request.headers.get("content-length"))
    except ValueError as exc:
        return None, str(exc)

    if declared is not None:
        if declared > max_bytes:
            return None, (f"Request body too large. Maximum size is {max_bytes} bytes.")
        # Trust Content-Length — downstream ASGI stack reads the body once.
        return request, None

    body = b""
    async for chunk in request.stream():
        body += chunk
        if len(body) > max_bytes:
            return None, (f"Request body too large. Maximum size is {max_bytes} bytes.")

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(request.scope, receive), None


# ---------------------------------------------------------------------------
# Client IP / rate-limit key resolution (proxy-aware)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _forwarded_allow_networks(
    spec: str,
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Parse a comma-separated list of IPs/CIDRs for trusted reverse proxies."""
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for raw in spec.split(","):
        entry = raw.strip()
        if not entry:
            continue
        if "/" not in entry:
            entry = f"{entry}/32" if ":" not in entry else f"{entry}/128"
        networks.append(ipaddress.ip_network(entry, strict=False))
    return tuple(networks)


def _peer_is_trusted(peer_host: str | None, forwarded_allow_ips: str) -> bool:
    """Return True when the direct TCP peer is a trusted reverse proxy."""
    if not peer_host or not forwarded_allow_ips.strip():
        return False
    try:
        peer = ipaddress.ip_address(peer_host)
    except ValueError:
        return False
    return any(
        peer in network for network in _forwarded_allow_networks(forwarded_allow_ips)
    )


def _first_xff_ip(xff: str) -> str | None:
    """Return the left-most (client) IP from an X-Forwarded-For header."""
    if not xff.strip():
        return None
    candidate = xff.split(",")[0].strip()
    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        return None
    return candidate


def _forwarded_header_ip(forwarded: str) -> str | None:
    """Extract the client IP from an RFC 7239 Forwarded header (first ``for=``)."""
    for part in forwarded.split(","):
        for directive in part.split(";"):
            directive = directive.strip()
            if not directive.lower().startswith("for="):
                continue
            value = directive[4:].strip().strip('"')
            if value.lower() == "unknown":
                continue
            if value.startswith("[") and "]" in value:
                value = value[1 : value.index("]")]
            try:
                ipaddress.ip_address(value)
            except ValueError:
                continue
            return value
    return None


def resolve_client_ip(
    peer_host: str | None,
    headers: Mapping[str, str],
    *,
    trust_proxy: bool,
    forwarded_allow_ips: str,
) -> str:
    """Return the best-effort client IP for logging and rate limiting.

    ``X-Forwarded-For`` / ``Forwarded`` are honoured only when ``trust_proxy``
    is enabled **and** the direct peer matches ``forwarded_allow_ips``.
    """
    if not peer_host:
        return "unknown"

    if trust_proxy and _peer_is_trusted(peer_host, forwarded_allow_ips):
        for key in ("x-forwarded-for", "X-Forwarded-For"):
            if key in headers:
                xff_ip = _first_xff_ip(headers[key])
                if xff_ip:
                    return xff_ip
        for key in ("forwarded", "Forwarded"):
            if key in headers:
                fwd_ip = _forwarded_header_ip(headers[key])
                if fwd_ip:
                    return fwd_ip

    return peer_host


def rate_limit_client_key(
    peer_host: str | None,
    headers: Mapping[str, str],
    *,
    trust_proxy: bool,
    forwarded_allow_ips: str,
    api_key: str | None,
    rate_limit_per_token: bool = False,
) -> str:
    """Build a sliding-window bucket key for the rate limiter.

    By default the limit is scoped per client IP (using proxy-aware resolution
    when configured). Per-token buckets apply only when ``rate_limit_per_token``
    is True — i.e. when multiple distinct Bearer secrets can be presented
    (multi-tenant). With the built-in single shared ``API_KEY``, all callers
    share the same secret so token hashing would collapse to one global bucket;
    IP scoping is used instead.
    """
    if rate_limit_per_token and api_key:
        auth = headers.get("authorization") or headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
            if token:
                digest = hashlib.sha256(token.encode()).hexdigest()[:16]
                return f"token:{digest}"

    client_ip = resolve_client_ip(
        peer_host,
        headers,
        trust_proxy=trust_proxy,
        forwarded_allow_ips=forwarded_allow_ips,
    )
    return f"ip:{client_ip}"


# ---------------------------------------------------------------------------
# RateLimiter — interface + implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class RateLimiterBackend(Protocol):
    """Protocol for rate-limiter backends (in-memory or distributed)."""

    def is_allowed(self, ip: str) -> bool:
        """Return True if the request from *ip* is within the rate limit."""
        ...

    def remaining(self, ip: str) -> int:
        """Return the number of requests remaining for *ip*."""
        ...


class RateLimiter:
    """
    Simple in-memory sliding-window rate limiter keyed by client identifier.

    Uses a ``deque`` per key (IP, token fingerprint, etc.) to track the timestamps
    of recent requests within the rolling window.  Timestamps older than
    ``window_seconds`` are pruned on every check, so memory usage stays bounded
    even under sustained load from a single client.

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
        self._call_count: int = 0
        self._EVICT_INTERVAL: int = 100

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
            self._call_count += 1
            if self._call_count % self._EVICT_INTERVAL == 0:
                self._evict_stale(cutoff)
            return True

    def _evict_stale(self, cutoff: float) -> None:
        """Remove IP buckets whose newest timestamp is older than ``cutoff``.

        Must be called while ``self._lock`` is held.
        """
        stale = [ip for ip, dq in self._buckets.items() if not dq or dq[-1] < cutoff]
        for ip in stale:
            del self._buckets[ip]

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


InMemoryRateLimiter = RateLimiter


class RedisRateLimiter:
    """Distributed sliding-window rate limiter backed by Redis.

    Uses a sorted set per IP with timestamps as scores.  The Lua script
    atomically prunes expired entries, checks the count, and adds the new
    timestamp — all in a single round-trip.

    Args:
        redis_url: Redis connection string.
        max_requests: Maximum requests per window per IP.
        window_seconds: Length of the sliding window in seconds.
    """

    _LUA_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local max_req = tonumber(ARGV[3])
local cutoff = now - window

redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
local count = redis.call('ZCARD', key)

if count >= max_req then
    return -1
end

redis.call('ZADD', key, now, now .. '-' .. math.random(1000000))
redis.call('EXPIRE', key, math.ceil(window) + 1)
return max_req - count - 1
"""

    def __init__(
        self,
        redis_url: str,
        max_requests: int = 60,
        window_seconds: float = 60.0,
    ) -> None:
        try:
            import redis as redis_lib

            self._redis = redis_lib.Redis.from_url(redis_url, decode_responses=True)
            self._script = self._redis.register_script(self._LUA_SCRIPT)
        except ImportError as exc:
            raise ImportError(
                "redis package required for RATE_LIMIT_BACKEND=redis. "
                "Install with: uv sync --extra redis"
            ) from exc

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._prefix = "ratelimit:"

    def is_allowed(self, ip: str) -> bool:
        """Check the rate limit for *ip* using Redis.

        Fail-open: if Redis is unreachable the request is allowed and a
        warning is logged.  Rate limiting is a non-critical function — an
        outage should never block legitimate traffic.
        """
        try:
            now = time.time()
            key = f"{self._prefix}{ip}"
            result = self._script(
                keys=[key],
                args=[now, self.window_seconds, self.max_requests],
            )
            if result is None:
                return True  # fail-open
            return int(cast(int | str, result)) >= 0  # -1 = bloqué, >= 0 = autorisé
        except Exception as exc:
            logger.warning(
                "Redis rate limiter unreachable — failing open (request allowed)",
                extra={"ip": ip, "error": str(exc)},
            )
            return True

    def remaining(self, ip: str) -> int:
        """Return remaining requests for *ip* in the current window."""
        now = time.time()
        cutoff = now - self.window_seconds
        key = f"{self._prefix}{ip}"
        try:
            self._redis.zremrangebyscore(key, "-inf", cutoff)
            count = cast(int, self._redis.zcard(key))
            return max(0, self.max_requests - count)
        except Exception:
            return self.max_requests


def create_rate_limiter(
    backend: Literal["memory", "redis"] = "memory",
    redis_url: str = "",
    max_requests: int = 60,
    window_seconds: float = 60.0,
) -> RateLimiterBackend:
    """Factory: build a rate limiter matching the configured backend.

    Args:
        backend: ``"memory"`` (default, per-process) or ``"redis"``
            (shared across replicas).
        redis_url: Required when ``backend="redis"``.
        max_requests: Requests allowed per window per IP.
        window_seconds: Sliding window duration.

    Returns:
        A rate-limiter instance satisfying ``RateLimiterBackend``.
    """
    if backend == "redis":
        if not redis_url:
            logger.warning(
                "RATE_LIMIT_BACKEND=redis but REDIS_URL is not set — "
                "falling back to in-memory rate limiter."
            )
            return RateLimiter(max_requests=max_requests, window_seconds=window_seconds)
        return RedisRateLimiter(
            redis_url=redis_url,
            max_requests=max_requests,
            window_seconds=window_seconds,
        )
    return RateLimiter(max_requests=max_requests, window_seconds=window_seconds)


# ---------------------------------------------------------------------------
# sanitize_log_data
# ---------------------------------------------------------------------------

_SENSITIVE_RE: re.Pattern[str] = re.compile(
    r"(?<![a-z0-9])("
    + "|".join(
        re.escape(p)
        for p in (
            "key",
            "token",
            "secret",
            "password",
            "passwd",
            "pwd",
            "credential",
            "authorization",
            "auth_token",
        )
    )
    + r")(?![a-z0-9])"
)

_MASK = "***REDACTED***"


def sanitize_log_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of ``data`` with sensitive values masked.

    Recursively traverses nested dicts and lists.  Any key whose lowercased
    name matches a sensitive word (checked via ``_SENSITIVE_RE``) has its
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
        if _is_sensitive_key(k):
            sanitised[k] = _MASK
        elif isinstance(v, dict):
            sanitised[k] = sanitize_log_data(v)
        elif isinstance(v, list):
            sanitised[k] = [
                sanitize_log_data(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            sanitised[k] = v
    return sanitised


def _is_sensitive_key(key: str) -> bool:
    """Return True if ``key`` (case-insensitive) contains a sensitive word."""
    return bool(_SENSITIVE_RE.search(key.lower()))


# ---------------------------------------------------------------------------
# validate_api_key_format
# ---------------------------------------------------------------------------

_API_KEY_PATTERNS: dict[str, re.Pattern[str]] = {
    "anthropic": re.compile(r"^sk-ant-[A-Za-z0-9\-_]{10,}$"),
    "openai": re.compile(r"^sk-[A-Za-z0-9\-_]{20,}$"),
    "azure": re.compile(r"^[A-Fa-f0-9]{32}$"),
}

_GENERIC_KEY_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z0-9\-_]{8,}$")


def validate_api_key_format(key: str, provider: str = "anthropic") -> bool:
    """
    Check whether ``key`` matches the expected API key format for ``provider``.

    Supported providers and their expected formats:

    * ``anthropic`` — ``sk-ant-`` prefix + 10+ alphanumeric chars.
    * ``openai``    — ``sk-`` prefix + 20+ alphanumeric chars.
    * ``azure``     — 32-character hexadecimal string.
    * Other providers fall back to a generic check (8+ alphanumeric chars).

    This is a structural check only — it does not verify the key against
    the provider's API.  Use it at startup to catch copy-paste errors and
    unset placeholder values before the first real API call is attempted.

    Args:
        key: The API key string to check.
        provider: LLM provider name (e.g. ``"anthropic"``, ``"openai"``).

    Returns:
        ``True`` if ``key`` matches the expected format, ``False`` otherwise.

    Example::

        if not validate_api_key_format(settings.anthropic_api_key, "anthropic"):
            raise RuntimeError("API key does not match expected format.")
    """
    if not isinstance(key, str):
        return False
    pattern = _API_KEY_PATTERNS.get(provider, _GENERIC_KEY_PATTERN)
    return bool(pattern.match(key))
