"""
tests/test_middleware.py — Tests for api/middleware.py and rate-limit key helpers.

Covers the gaps left by tests/test_api.py (auth, 413, drain are tested there):
security response headers, X-Request-ID echo, the per-endpoint rate-limit key
helpers in api/dependencies.py, per-endpoint bucket isolation, and the strict
session_id path pattern.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.routing import Route

from tests.legacy_pack_override import override_legacy_pack_cls

# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------


def test_security_headers_on_regular_response(test_client: TestClient) -> None:
    """Every response carries the hardened security headers (dev environment)."""
    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Content-Security-Policy"] == "default-src 'self'"
    # The Server banner must be stripped, and HSTS only applies in production.
    assert "server" not in response.headers
    assert "Strict-Transport-Security" not in response.headers


def test_security_headers_relaxed_csp_on_docs(test_client: TestClient) -> None:
    """/docs gets the CDN-allowing CSP needed by Swagger UI."""
    response = test_client.get("/docs")

    csp = response.headers["Content-Security-Policy"]
    assert "cdn.jsdelivr.net" in csp
    assert csp.startswith("default-src 'self';")


# ---------------------------------------------------------------------------
# X-Request-ID
# ---------------------------------------------------------------------------


def test_request_id_echoed_when_supplied(test_client: TestClient) -> None:
    """A client-supplied X-Request-ID is propagated back unchanged."""
    response = test_client.get("/health", headers={"X-Request-ID": "req-12345"})

    assert response.headers["X-Request-ID"] == "req-12345"


# ---------------------------------------------------------------------------
# Rate-limit key helpers (api/dependencies.py)
# ---------------------------------------------------------------------------


def _build_request(path: str, *, route_template: str | None = None) -> Request:
    scope: dict[str, object] = {
        "type": "http",
        "asgi": {"spec_version": "2.3", "version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "scheme": "http",
        "query_string": b"",
        "headers": [],
        "client": ("203.0.113.7", 50000),
        "server": ("testserver", 80),
    }
    if route_template is not None:
        scope["route"] = Route(route_template, endpoint=lambda: None)
    return Request(scope)


def _no_proxy_settings():
    from core.config import Settings

    return Settings(
        llm_provider="anthropic",
        anthropic_api_key="sk-ant-test123456789012345",
        trust_proxy_headers=False,
        api_key=None,
    )


def test_rate_limit_route_path_prefers_route_template() -> None:
    """The low-cardinality route template is used when routing has resolved."""
    from api.dependencies import _rate_limit_route_path

    request = _build_request(
        "/sessions/abc-123/history",
        route_template="/sessions/{session_id}/history",
    )
    assert _rate_limit_route_path(request) == "/sessions/{session_id}/history"


def test_rate_limit_route_path_falls_back_to_raw_path() -> None:
    """Without a resolved route, the raw request path is used."""
    from api.dependencies import _rate_limit_route_path

    request = _build_request("/run")
    assert _rate_limit_route_path(request) == "/run"


def test_rate_limit_endpoint_key_includes_client_and_path() -> None:
    """The endpoint key is the client key suffixed with the endpoint path."""
    from api.dependencies import _rate_limit_endpoint_key, _rate_limit_key

    request = _build_request("/run")
    with patch("api.dependencies.get_settings", return_value=_no_proxy_settings()):
        client_key = _rate_limit_key(request)
        endpoint_key = _rate_limit_endpoint_key(request)

    assert endpoint_key == f"{client_key}:/run"
    assert "203.0.113.7" in client_key


def test_rate_limit_endpoint_keys_differ_between_endpoints() -> None:
    """The same client gets distinct buckets for distinct endpoints."""
    from api.dependencies import _rate_limit_endpoint_key

    with patch("api.dependencies.get_settings", return_value=_no_proxy_settings()):
        key_run = _rate_limit_endpoint_key(_build_request("/run"))
        key_research = _rate_limit_endpoint_key(_build_request("/research"))

    assert key_run != key_research
    assert key_run.endswith(":/run")
    assert key_research.endswith(":/research")


# ---------------------------------------------------------------------------
# Per-endpoint rate-limit isolation (integration through the middleware)
# ---------------------------------------------------------------------------


def test_rate_limit_buckets_isolated_per_endpoint() -> None:
    """Exhausting the limit on /run must not block /research for the same client."""
    from core.security import RateLimiter

    tight_limiter = RateLimiter(max_requests=2, window_seconds=60.0)

    mock_graph_instance = MagicMock()
    mock_graph_instance.run.return_value = MagicMock(
        query="q",
        executive_summary="s",
        key_insights=[],
        patterns=[],
        implications=[],
        confidence=0.5,
        research_summary="r",
        metadata={},
    )
    mock_graph_instance.__enter__ = MagicMock(return_value=mock_graph_instance)
    mock_graph_instance.__exit__ = MagicMock(return_value=False)
    mock_graph_cls = MagicMock(return_value=mock_graph_instance)

    mock_agent = MagicMock()
    mock_agent.run_structured.return_value = MagicMock(
        query="q",
        summary="s",
        findings=[],
        sources=[],
        confidence=0.5,
        metadata={},
    )

    with (
        override_legacy_pack_cls(mock_graph_cls),
        patch("api.endpoints.pipeline.ResearchAgent", return_value=mock_agent),
        patch("api.state.rate_limiter", tight_limiter),
        patch("api.state.get_shared_llm", return_value=MagicMock(spec=True)),
        patch("api.state.get_shared_checkpointer", return_value=MagicMock()),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            payload = {"query": "What is quantum computing?"}

            # Exhaust the /run bucket: 2 allowed, 3rd rejected.
            assert client.post("/run", json=payload).status_code == 200
            assert client.post("/run", json=payload).status_code == 200
            blocked = client.post("/run", json=payload)
            assert blocked.status_code == 429
            assert "Retry-After" in blocked.headers

            # /research has its own bucket and must still be allowed.
            assert client.post("/research", json=payload).status_code == 200


# ---------------------------------------------------------------------------
# Session ID path pattern (api/endpoints/sessions.py)
# ---------------------------------------------------------------------------


def test_session_history_rejects_invalid_session_id(test_client: TestClient) -> None:
    """session_id outside ^[a-zA-Z0-9_-]+$ is rejected with 422."""
    response = test_client.get("/sessions/foo!bar/history")

    assert response.status_code == 422


def test_session_history_accepts_valid_session_id(test_client: TestClient) -> None:
    """A well-formed session_id passes path validation."""
    response = test_client.get("/sessions/valid_session-123/history")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "valid_session-123"
    assert body["entries"] == []
    assert body["total"] == 0
