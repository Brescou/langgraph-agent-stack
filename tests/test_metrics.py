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
        patch("api.main._rate_limiter", permissive),
        patch("api.main.get_shared_llm", return_value=mock_llm),
        patch("api.main.get_shared_checkpointer", return_value=mock_checkpointer),
        patch.dict(os.environ, {"API_KEY": "test-secret-key-12345"}),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/metrics")
            assert response.status_code == 200


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
    os.environ["ENVIRONMENT"] = "production"
    get_settings.cache_clear()
    try:
        with (
            patch("api.main._rate_limiter", permissive),
            patch("api.main.get_shared_llm", return_value=mock_llm),
            patch("api.main.get_shared_checkpointer", return_value=mock_checkpointer),
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
