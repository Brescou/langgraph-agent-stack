"""
tests/test_metrics.py — Tests for Prometheus metrics, request ID propagation,
CORS configuration, HSTS header, and startup validation.

Covers features introduced in the quality sprint v4 (blocs 2, 4, 5, 6, 7).
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_prometheus_available = False
try:
    import prometheus_client  # noqa: F401

    _prometheus_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# /metrics endpoint (only when prometheus-client is installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _prometheus_available,
    reason="prometheus-client not installed",
)
def test_metrics_endpoint_returns_200(test_client: TestClient) -> None:
    """GET /metrics returns 200 with Prometheus text format."""
    response = test_client.get("/metrics")
    assert response.status_code == 200


@pytest.mark.skipif(
    not _prometheus_available,
    reason="prometheus-client not installed",
)
def test_metrics_exempt_from_auth() -> None:
    """GET /metrics bypasses authentication when API_KEY is set."""
    from core.security import RateLimiter

    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)
    mock_llm = MagicMock(spec=True)
    mock_checkpointer = MagicMock()

    with (
        patch("api.state.rate_limiter", permissive),
        patch("api.state.get_shared_llm", return_value=mock_llm),
        patch("api.state.get_shared_checkpointer", return_value=mock_checkpointer),
        patch.dict(os.environ, {"API_KEY": "test-secret-key-12345"}),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/metrics")
            assert response.status_code == 200


@pytest.mark.skipif(
    not _prometheus_available,
    reason="prometheus-client not installed",
)
def test_http_metrics_use_route_template_not_concrete_path(
    test_client: TestClient,
) -> None:
    """Prometheus ``path`` labels must use route templates, not URL paths with IDs."""
    session_id = "cardinality-test-session-uuid-12345"
    test_client.get(f"/sessions/{session_id}/history")
    metrics = test_client.get("/metrics")
    assert metrics.status_code == 200
    body = metrics.text
    assert 'path="/sessions/{session_id}/history"' in body
    assert session_id not in body


@pytest.mark.skipif(
    not _prometheus_available,
    reason="prometheus-client not installed",
)
def test_pack_run_emits_pack_metrics(
    test_client: TestClient,
    mock_analysis_report: MagicMock,
) -> None:
    """POST /packs/research_analysis/run increments pack_runs_total with labels."""
    from unittest.mock import patch

    from domain_packs.research.research_analysis.pack import ResearchAnalysisPack

    def _noop_init(self, **kwargs):  # type: ignore[override]
        pass

    with (
        patch.object(ResearchAnalysisPack, "__init__", _noop_init),
        patch.object(ResearchAnalysisPack, "run", return_value=mock_analysis_report),
        patch.object(ResearchAnalysisPack, "close", return_value=None),
    ):
        response = test_client.post(
            "/packs/research_analysis/run",
            json={"query": "What is a microservice?"},
        )

    assert response.status_code == 200
    used_version = response.headers.get("x-pack-version-used", "unknown")
    metrics = test_client.get("/metrics")
    assert metrics.status_code == 200
    body = metrics.text
    assert (
        f'pack_runs_total{{outcome="success",pack_id="research_analysis",version="{used_version}"}}'
        in body
        or f'pack_runs_total{{pack_id="research_analysis",version="{used_version}",outcome="success"}}'
        in body
    )


@pytest.mark.skipif(
    not _prometheus_available,
    reason="prometheus-client not installed",
)
def test_pack_run_metrics_two_versions(
    test_client: TestClient,
    mock_analysis_report: MagicMock,
) -> None:
    """Canary: two X-Pack-Version values produce distinct pack_runs_total series."""
    from unittest.mock import patch

    from domain_packs.research.research_analysis.pack import ResearchAnalysisPack
    from pack_kernel.registry import PackRegistry

    def _noop_init(self, **kwargs):  # type: ignore[override]
        pass

    class _V2Pack(ResearchAnalysisPack):
        pack_id = "research_analysis"
        version = "2.0"
        name = ResearchAnalysisPack.name
        description = ResearchAnalysisPack.description

    PackRegistry.register(_V2Pack)
    try:
        with (
            patch.object(ResearchAnalysisPack, "__init__", _noop_init),
            patch.object(_V2Pack, "__init__", _noop_init),
            patch.object(ResearchAnalysisPack, "run", return_value=mock_analysis_report),
            patch.object(_V2Pack, "run", return_value=mock_analysis_report),
            patch.object(ResearchAnalysisPack, "close", return_value=None),
            patch.object(_V2Pack, "close", return_value=None),
        ):
            r1 = test_client.post(
                "/packs/research_analysis/run",
                json={"query": "version one"},
                headers={"X-Pack-Version": "1.0"},
            )
            r2 = test_client.post(
                "/packs/research_analysis/run",
                json={"query": "version two"},
                headers={"X-Pack-Version": "2.0"},
            )

        assert r1.status_code == 200
        assert r2.status_code == 200
        metrics = test_client.get("/metrics").text
        assert 'version="1.0"' in metrics
        assert 'version="2.0"' in metrics
        assert 'pack_id="research_analysis"' in metrics
    finally:
        versions = PackRegistry._get_versions("research_analysis")
        PackRegistry._registry["research_analysis"] = [
            pv for pv in versions if pv.version != "2.0"
        ]


# ---------------------------------------------------------------------------
# X-Request-ID propagation
# ---------------------------------------------------------------------------


def test_request_id_propagated_in_response(test_client: TestClient) -> None:
    """Responses include an X-Request-ID header."""
    response = test_client.get("/health")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


def test_request_id_echoed_when_provided(test_client: TestClient) -> None:
    """When the client sends X-Request-ID, the same value is returned."""
    custom_id = "my-trace-id-42"
    response = test_client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.headers["X-Request-ID"] == custom_id


def test_request_id_generated_when_absent(test_client: TestClient) -> None:
    """When no X-Request-ID is sent, the server generates a UUID."""
    response = test_client.get("/health")
    request_id = response.headers.get("X-Request-ID", "")
    assert len(request_id) == 36  # UUID format


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------


def test_cors_wildcard_disables_credentials(test_client: TestClient) -> None:
    """When CORS_ORIGINS is empty, credentials must not be allowed."""
    response = test_client.options(
        "/run",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    allow_creds = response.headers.get("Access-Control-Allow-Credentials", "")
    assert allow_creds != "true"


# ---------------------------------------------------------------------------
# HSTS header
# ---------------------------------------------------------------------------


def test_hsts_absent_in_development(test_client: TestClient) -> None:
    """HSTS should NOT be present in development mode."""
    response = test_client.get("/health")
    assert "Strict-Transport-Security" not in response.headers


def test_hsts_present_in_production() -> None:
    """HSTS must be set when ENVIRONMENT=production."""
    from core.config import get_settings
    from core.security import RateLimiter

    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)
    mock_llm = MagicMock(spec=True)
    mock_checkpointer = MagicMock()

    original_env = os.environ.get("ENVIRONMENT", "development")
    original_api_key = os.environ.get("API_KEY")
    os.environ["ENVIRONMENT"] = "production"
    os.environ["API_KEY"] = "test-production-api-key"
    get_settings.cache_clear()
    try:
        with (
            patch("api.state.rate_limiter", permissive),
            patch("api.state.get_shared_llm", return_value=mock_llm),
            patch("api.state.get_shared_checkpointer", return_value=mock_checkpointer),
        ):
            from api.main import app

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/health")
                assert "Strict-Transport-Security" in response.headers
                assert (
                    "max-age=31536000" in response.headers["Strict-Transport-Security"]
                )
    finally:
        os.environ["ENVIRONMENT"] = original_env
        if original_api_key is None:
            os.environ.pop("API_KEY", None)
        else:
            os.environ["API_KEY"] = original_api_key
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Startup validation (BLOC 7)
# ---------------------------------------------------------------------------


def test_startup_fails_with_postgres_backend_and_no_url() -> None:
    """Settings validator rejects MEMORY_BACKEND=postgres without POSTGRES_URL."""
    from pydantic import ValidationError

    from core.config import Settings, get_settings

    get_settings.cache_clear()
    original_backend = os.environ.get("MEMORY_BACKEND", "sqlite")
    try:
        with patch.dict(
            os.environ,
            {"MEMORY_BACKEND": "postgres"},
        ):
            os.environ.pop("POSTGRES_URL", None)
            get_settings.cache_clear()

            with pytest.raises(ValidationError, match="POSTGRES_URL must be set"):
                Settings()
    finally:
        os.environ["MEMORY_BACKEND"] = original_backend
        get_settings.cache_clear()
