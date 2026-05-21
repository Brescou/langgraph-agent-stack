"""
tests/test_api.py — Functional tests for the FastAPI endpoints.

All agent calls are mocked; no real Anthropic API requests are made.
Each test is fully isolated — the ``test_client`` fixture is function-scoped.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from agents.base_agent import (
    AgentBudgetExceededError,
    AgentExecutionError,
    AgentTimeoutError,
    AgentValidationError,
)
from agents.models import AnalysisReport, ResearchResult
from tests.legacy_pack_override import override_legacy_pack_cls

# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_check(test_client: TestClient) -> None:
    """GET /health must return 200 with the expected response fields."""
    response = test_client.get("/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] in ("ok", "degraded")
    assert "version" in body
    assert "uptime_seconds" in body
    assert isinstance(body["uptime_seconds"], float)
    assert "environment" in body
    assert "components" in body


def test_health_check_with_llm_initialised() -> None:
    """GET /health returns component.llm.status=ok when the LLM is initialised.

    Regression test: LLMProvider is a Literal (str), not an Enum.
    Calling .value on it would raise AttributeError.
    """
    import api.main as api_module
    from core.security import RateLimiter

    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)
    mock_llm = MagicMock(spec=True)
    mock_checkpointer = MagicMock()
    mock_memory = MagicMock()
    mock_memory.db_path = ":memory:"
    mock_memory.health_check.return_value = ("ok", ":memory:")

    with (
        patch("api.main._rate_limiter", permissive),
        patch("api.main.get_shared_llm", return_value=mock_llm),
        patch("api.main.get_shared_checkpointer", return_value=mock_checkpointer),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            saved = (
                api_module._shared_llm,
                api_module._shared_checkpointer,
                api_module._shared_memory,
            )
            api_module._shared_llm = mock_llm
            api_module._shared_checkpointer = mock_checkpointer
            api_module._shared_memory = mock_memory
            try:
                response = client.get("/health")
            finally:
                (
                    api_module._shared_llm,
                    api_module._shared_checkpointer,
                    api_module._shared_memory,
                ) = saved

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["components"]["llm"]["status"] == "ok"
    assert body["components"]["llm"]["detail"] == "anthropic (initialised)"


# ---------------------------------------------------------------------------
# GET /ready
# ---------------------------------------------------------------------------


def test_ready_returns_200_when_initialised(test_client: TestClient) -> None:
    """GET /ready must return 200 when LLM and checkpointer are initialised."""
    response = test_client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"


def test_ready_returns_503_when_llm_not_initialised() -> None:
    """GET /ready must return 503 when the LLM is not initialised."""
    from core.security import RateLimiter

    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)

    with (
        patch("api.main._rate_limiter", permissive),
        patch("api.main.get_shared_llm", return_value=None),
        patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/ready")

    assert response.status_code == 503
    assert "not ready" in response.json()["detail"].lower()


def test_ready_returns_503_when_shutting_down() -> None:
    """GET /ready must return 503 when the server is shutting down."""
    import api.main as api_module
    from core.security import RateLimiter

    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)

    with (
        patch("api.main._rate_limiter", permissive),
        patch("api.main.get_shared_llm", return_value=MagicMock(spec=True)),
        patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            api_module._shutting_down.set()
            try:
                response = client.get("/ready")
            finally:
                api_module._shutting_down.clear()

    assert response.status_code == 503
    assert "shutting down" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# POST /run
# ---------------------------------------------------------------------------


def test_run_success(test_client: TestClient, mock_analysis_report: MagicMock) -> None:
    """POST /run with a valid query must return 200 and a RunResponse payload."""
    response = test_client.post("/run", json={"query": "What is quantum computing?"})

    assert response.status_code == 200

    body = response.json()
    assert "query" in body
    assert "executive_summary" in body
    assert "key_insights" in body
    assert isinstance(body["key_insights"], list)
    assert "patterns" in body
    assert isinstance(body["patterns"], list)
    assert "implications" in body
    assert isinstance(body["implications"], list)
    assert "confidence" in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert "research_summary" in body
    assert "metadata" in body


def test_run_empty_query(test_client: TestClient) -> None:
    """POST /run with an empty query string must return 422 (Pydantic validation)."""
    response = test_client.post("/run", json={"query": ""})

    assert response.status_code == 422


def test_run_query_too_long(test_client: TestClient) -> None:
    """POST /run with a query exceeding 2000 characters must return 422."""
    long_query = "a" * 2001
    response = test_client.post("/run", json={"query": long_query})

    assert response.status_code == 422


def test_run_agent_error(test_client: TestClient) -> None:
    """POST /run must return 500 when MultiAgentGraph.run() raises AgentExecutionError."""
    mock_graph_instance = MagicMock()
    mock_graph_instance.run.side_effect = AgentExecutionError("Pipeline failed")
    mock_graph_instance.__enter__ = MagicMock(return_value=mock_graph_instance)
    mock_graph_instance.__exit__ = MagicMock(return_value=False)
    mock_graph_cls = MagicMock(return_value=mock_graph_instance)

    with override_legacy_pack_cls(mock_graph_cls):
        response = test_client.post(
            "/run", json={"query": "What is quantum computing?"}
        )

    assert response.status_code == 500
    assert "detail" in response.json()


def test_run_timeout_error(test_client: TestClient) -> None:
    """POST /run must return 504 when MultiAgentGraph.run() raises AgentTimeoutError."""
    mock_graph_instance = MagicMock()
    mock_graph_instance.run.side_effect = AgentTimeoutError("Step budget exceeded")
    mock_graph_instance.__enter__ = MagicMock(return_value=mock_graph_instance)
    mock_graph_instance.__exit__ = MagicMock(return_value=False)
    mock_graph_cls = MagicMock(return_value=mock_graph_instance)

    with override_legacy_pack_cls(mock_graph_cls):
        response = test_client.post(
            "/run", json={"query": "What is quantum computing?"}
        )

    assert response.status_code == 504
    assert "detail" in response.json()


# ---------------------------------------------------------------------------
# POST /research
# ---------------------------------------------------------------------------


def test_research_success(
    test_client: TestClient, mock_research_result: MagicMock
) -> None:
    """POST /research with a valid query must return 200 and a ResearchResponse payload."""
    response = test_client.post(
        "/research", json={"query": "Explain the CAP theorem in distributed systems."}
    )

    assert response.status_code == 200

    body = response.json()
    assert "query" in body
    assert "summary" in body
    assert "findings" in body
    assert isinstance(body["findings"], list)
    assert "sources" in body
    assert isinstance(body["sources"], list)
    assert "confidence" in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert "metadata" in body


def test_research_rejects_null_byte(test_client: TestClient) -> None:
    """POST /research must reject queries containing embedded null bytes."""
    response = test_client.post("/research", json={"query": "hello\x00world"})

    assert response.status_code == 400
    body = response.json()
    assert "detail" in body


def test_research_allows_prompt_injection_research_query(
    test_client: TestClient,
) -> None:
    """Queries discussing prompt injection must not be blocked by InputValidator."""
    query = (
        "Explain what 'ignore previous instructions' means in "
        "prompt injection research."
    )
    response = test_client.post("/research", json={"query": query})

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limiting() -> None:
    """
    Exceeding the rate limit on POST /run must eventually return 429.

    A dedicated client with a tight limit (max 3 requests) is created for
    this test so it does not interfere with other tests' fixture state.
    """
    from core.security import RateLimiter

    tight_limiter = RateLimiter(max_requests=3, window_seconds=60.0)

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

    with (
        override_legacy_pack_cls(mock_graph_cls),
        patch("api.main._rate_limiter", tight_limiter),
        patch("api.main.get_shared_llm", return_value=MagicMock(spec=True)),
        patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            statuses = []
            for _ in range(10):
                r = client.post("/run", json={"query": "What is quantum computing?"})
                statuses.append(r.status_code)

    assert 429 in statuses, "Expected at least one 429 Too Many Requests response"


def test_rate_limiting_on_research() -> None:
    """Exceeding the rate limit on POST /research must eventually return 429."""
    from core.security import RateLimiter

    tight_limiter = RateLimiter(max_requests=3, window_seconds=60.0)
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
        patch("api.main.ResearchAgent", return_value=mock_agent),
        patch("api.main._rate_limiter", tight_limiter),
        patch("api.main.get_shared_llm", return_value=MagicMock(spec=True)),
        patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            statuses = []
            for _ in range(10):
                r = client.post(
                    "/research", json={"query": "What is quantum computing?"}
                )
                statuses.append(r.status_code)

    assert 429 in statuses, "Expected at least one 429 Too Many Requests response"


def test_rate_limiting_isolated_per_xff_client() -> None:
    """With trusted proxy settings, each X-Forwarded-For client gets its own bucket."""
    from core.config import Settings
    from core.security import RateLimiter

    tight_limiter = RateLimiter(max_requests=2, window_seconds=60.0)
    proxy_settings = Settings(
        llm_provider="anthropic",
        anthropic_api_key="sk-ant-test123456789012345",
        trust_proxy_headers=True,
        forwarded_allow_ips="10.0.0.0/8",
        api_key=None,
    )

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

    with (
        override_legacy_pack_cls(mock_graph_cls),
        patch("api.main._rate_limiter", tight_limiter),
        patch("api.main.get_settings", return_value=proxy_settings),
        patch("api.main._request_peer_host", return_value="10.0.0.1"),
        patch("api.main.get_shared_llm", return_value=MagicMock(spec=True)),
        patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            for _ in range(2):
                assert (
                    client.post(
                        "/run",
                        json={"query": "q1"},
                        headers={"X-Forwarded-For": "198.51.100.10"},
                    ).status_code
                    == 200
                )
            blocked = client.post(
                "/run",
                json={"query": "q2"},
                headers={"X-Forwarded-For": "198.51.100.10"},
            )
            other = client.post(
                "/run",
                json={"query": "q3"},
                headers={"X-Forwarded-For": "198.51.100.99"},
            )

    assert blocked.status_code == 429
    assert other.status_code == 200


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------


def test_security_headers(test_client: TestClient) -> None:
    """Every response must include the mandatory security headers."""
    response = test_client.get("/health")

    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert response.headers.get("content-security-policy") == "default-src 'self'"
    assert response.headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert response.headers.get("cache-control") == "no-store"


# ---------------------------------------------------------------------------
# POST /run/stream
# ---------------------------------------------------------------------------


def test_run_stream_returns_sse(test_client: TestClient) -> None:
    """POST /run/stream should return 200 with text/event-stream and valid SSE events."""
    response = test_client.post("/run/stream", json={"query": "test query"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    assert "data: " in response.text


def test_run_stream_empty_query_returns_400(test_client: TestClient) -> None:
    """POST /run/stream with a whitespace-only query should return 400."""
    response = test_client.post("/run/stream", json={"query": "   "})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# GET /sessions/{session_id}/history
# ---------------------------------------------------------------------------


def test_get_session_history_returns_history_response(test_client: TestClient) -> None:
    """GET /sessions/{id}/history should return 200 with a HistoryResponse."""
    response = test_client.get("/sessions/test-session-abc/history")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "entries" in data
    assert "total" in data
    assert data["session_id"] == "test-session-abc"
    assert isinstance(data["entries"], list)


def test_get_session_history_unknown_session_returns_empty(
    test_client: TestClient,
) -> None:
    """GET /sessions/{id}/history for unknown session returns empty entries."""
    response = test_client.get("/sessions/nonexistent-session-xyz/history")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0


def test_get_session_history_with_populated_data() -> None:
    """GET /sessions/{id}/history returns entries when runs exist for the session."""
    from core.security import RateLimiter

    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)

    with (
        patch("api.main._rate_limiter", permissive),
        patch("api.main.get_shared_llm", return_value=MagicMock(spec=True)),
        patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            import api.main as api_module

            mem = api_module._shared_memory
            if mem is not None:
                mem.save_run(
                    run_id="test-run-populated-001",
                    query="What is AI?",
                    result={"summary": "AI is intelligence demonstrated by machines."},
                    metadata={"session_id": "populated-session-xyz"},
                )
                mem.save_run(
                    run_id="test-run-populated-002",
                    query="What is ML?",
                    result={"summary": "ML is a subset of AI."},
                    metadata={"session_id": "populated-session-xyz"},
                )

            response = client.get("/sessions/populated-session-xyz/history")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 2
    assert len(data["entries"]) >= 2
    assert data["entries"][0]["run_id"] in (
        "test-run-populated-001",
        "test-run-populated-002",
    )


# ---------------------------------------------------------------------------
# Request body size limit
# ---------------------------------------------------------------------------


def test_post_run_rejects_oversized_body(test_client: TestClient) -> None:
    """Oversized JSON bodies must return 413 before route handlers run."""
    from core.config import get_settings

    get_settings.cache_clear()
    with patch.dict(os.environ, {"MAX_REQUEST_BODY_BYTES": "1024"}, clear=False):
        get_settings.cache_clear()
        response = test_client.post(
            "/run",
            content=b"x" * 1025,
            headers={"Content-Type": "application/json"},
        )
    get_settings.cache_clear()
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


def test_post_run_rejects_oversized_content_length_header(
    test_client: TestClient,
) -> None:
    from core.config import get_settings

    get_settings.cache_clear()
    with patch.dict(os.environ, {"MAX_REQUEST_BODY_BYTES": "1024"}, clear=False):
        get_settings.cache_clear()
        response = test_client.post(
            "/run",
            content=b"{}",
            headers={
                "Content-Type": "application/json",
                "Content-Length": "999999",
            },
        )
    get_settings.cache_clear()
    assert response.status_code == 413


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


def _auth_client_ctx(
    mock_analysis_report: MagicMock,
    api_key: str,
):
    """Context manager: TestClient with API_KEY configured via env + cache_clear."""
    from core.config import get_settings as _gs
    from core.security import RateLimiter

    @contextmanager
    def _ctx():
        permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)
        mock_graph_instance = MagicMock()
        mock_graph_instance.run.return_value = mock_analysis_report
        mock_graph_instance.__enter__ = MagicMock(return_value=mock_graph_instance)
        mock_graph_instance.__exit__ = MagicMock(return_value=False)
        mock_graph_cls = MagicMock(return_value=mock_graph_instance)
        mock_agent_cls = MagicMock(return_value=MagicMock())

        env_overlay = {"API_KEY": api_key}
        _gs.cache_clear()
        try:
            with (
                patch.dict(os.environ, env_overlay, clear=False),
                override_legacy_pack_cls(mock_graph_cls),
                patch("api.main.ResearchAgent", mock_agent_cls),
                patch("api.main._rate_limiter", permissive),
                patch("api.main.get_shared_llm", return_value=MagicMock(spec=True)),
                patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
            ):
                from api.main import app

                with TestClient(app, raise_server_exceptions=False) as client:
                    yield client
        finally:
            _gs.cache_clear()

    return _ctx()


def test_auth_missing_token_returns_401(
    mock_analysis_report: MagicMock,
) -> None:
    """POST /run without Authorization header returns 401 when API_KEY is set."""
    with _auth_client_ctx(mock_analysis_report, "secret-token") as client:
        response = client.post("/run", json={"query": "test"})

    assert response.status_code == 401
    assert "detail" in response.json()


def test_auth_wrong_token_returns_401(
    mock_analysis_report: MagicMock,
) -> None:
    """POST /run with wrong Bearer token returns 401."""
    with _auth_client_ctx(mock_analysis_report, "secret-token") as client:
        response = client.post(
            "/run",
            json={"query": "test"},
            headers={"Authorization": "Bearer wrong-token"},
        )

    assert response.status_code == 401


def test_auth_correct_token_passes(
    mock_analysis_report: MagicMock,
) -> None:
    """POST /run with correct Bearer token returns 200."""
    with _auth_client_ctx(mock_analysis_report, "secret-token") as client:
        response = client.post(
            "/run",
            json={"query": "test"},
            headers={"Authorization": "Bearer secret-token"},
        )

    assert response.status_code == 200


def test_auth_exempt_health_path(
    mock_analysis_report: MagicMock,
) -> None:
    """GET /health must be accessible without any auth token even when API_KEY is set."""
    with _auth_client_ctx(mock_analysis_report, "secret-token") as client:
        response = client.get("/health")

    assert response.status_code == 200


def test_auth_exempt_docs_path(
    mock_analysis_report: MagicMock,
) -> None:
    """GET /docs must be accessible without auth token when API_KEY is set."""
    with _auth_client_ctx(mock_analysis_report, "secret-token") as client:
        response = client.get("/docs")

    assert response.status_code == 200


def test_auth_exempt_openapi_path(
    mock_analysis_report: MagicMock,
) -> None:
    """GET /openapi.json must be accessible without auth token when API_KEY is set."""
    with _auth_client_ctx(mock_analysis_report, "secret-token") as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200


def test_auth_disabled_when_no_api_key(test_client: TestClient) -> None:
    """All requests pass through when API_KEY is not configured."""
    # test_client fixture has no API_KEY set — /run should return 200 (mocked)
    response = test_client.post("/run", json={"query": "What is quantum computing?"})
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# SSE event parsing
# ---------------------------------------------------------------------------


def test_run_stream_events_are_valid_json(test_client: TestClient) -> None:
    """Each SSE data line in /run/stream must be valid JSON with a 'type' field."""
    response = test_client.post("/run/stream", json={"query": "test query"})
    assert response.status_code == 200

    events = []
    for line in response.text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            payload = json.loads(line[6:])
            assert "type" in payload
            events.append(payload)

    event_types = [e["type"] for e in events]
    assert "status" in event_types


# ---------------------------------------------------------------------------
# Validation / error-path coverage
# ---------------------------------------------------------------------------


def test_run_validation_error_returns_400(test_client: TestClient) -> None:
    """POST /run must return 400 when pipeline raises AgentValidationError."""
    inst = MagicMock()
    inst.run.side_effect = AgentValidationError("Bad query")
    inst.__enter__ = MagicMock(return_value=inst)
    inst.__exit__ = MagicMock(return_value=False)
    mock_cls = MagicMock(return_value=inst)

    with override_legacy_pack_cls(mock_cls):
        response = test_client.post("/run", json={"query": "What is AI?"})

    assert response.status_code == 400


def test_root_redirects_to_docs(test_client: TestClient) -> None:
    """GET / should redirect to /docs."""
    response = test_client.get("/", follow_redirects=False)
    assert response.status_code in (301, 302, 307, 308)


def test_research_agent_error_returns_500(test_client: TestClient) -> None:
    """POST /research must return 500 when ResearchAgent raises AgentExecutionError."""
    with patch("api.main.ResearchAgent") as mock_cls:
        inst = MagicMock()
        inst.run_structured.side_effect = AgentExecutionError("Research failed")
        mock_cls.return_value = inst
        response = test_client.post(
            "/research", json={"query": "Explain distributed systems."}
        )

    assert response.status_code == 500
    assert "detail" in response.json()


def test_run_returns_503_when_llm_not_configured() -> None:
    """POST /run returns 503 when LLM provider is not available."""
    from core.security import RateLimiter

    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)
    with (
        patch("api.main._rate_limiter", permissive),
        patch("api.main.get_shared_llm", return_value=None),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/run", json={"query": "test"})

    assert response.status_code == 503


# ---------------------------------------------------------------------------
# SSE done event and timeout tests
# ---------------------------------------------------------------------------


def test_run_stream_done_event_has_required_fields(test_client: TestClient) -> None:
    """The SSE 'done' event must contain run_id, session_id, confidence, executive_summary."""
    response = test_client.post("/run/stream", json={"query": "test query"})
    assert response.status_code == 200

    done_events = []
    for line in response.text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            payload = json.loads(line[6:])
            if payload.get("type") == "done":
                done_events.append(payload)

    assert len(done_events) == 1, "Expected exactly one 'done' event"
    done = done_events[0]
    assert "run_id" in done
    assert "session_id" in done
    assert "confidence" in done
    assert "executive_summary" in done


def test_run_stream_timeout_returns_error_event() -> None:
    """When stream_events raises AgentTimeoutError, SSE emits an error event."""
    from core.security import RateLimiter

    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)

    async def _error_stream(query):
        raise AgentTimeoutError("Step budget exceeded")
        yield  # pragma: no cover

    mock_graph_instance = MagicMock()
    mock_graph_instance.stream_events = _error_stream
    mock_graph_cls = MagicMock(return_value=mock_graph_instance)

    with (
        override_legacy_pack_cls(mock_graph_cls),
        patch("api.main._rate_limiter", permissive),
        patch("api.main.get_shared_llm", return_value=MagicMock(spec=True)),
        patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/run/stream", json={"query": "test query"})

    assert response.status_code == 200

    error_events = []
    for line in response.text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            payload = json.loads(line[6:])
            if payload.get("type") == "error":
                error_events.append(payload)

    assert len(error_events) >= 1, "Expected at least one 'error' event"
    assert "timed out" in error_events[0]["message"].lower()


# ---------------------------------------------------------------------------
# Shutdown guard — 503 on all endpoints
# ---------------------------------------------------------------------------


def _shutdown_client_ctx():
    """Context manager: TestClient with _shutting_down set after lifespan init."""
    from core.security import RateLimiter

    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)

    @contextmanager
    def _ctx():
        with (
            patch("api.main._rate_limiter", permissive),
            patch("api.main.get_shared_llm", return_value=MagicMock(spec=True)),
            patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
        ):
            import api.main as api_module
            from api.main import app

            with TestClient(app, raise_server_exceptions=False) as client:
                api_module._shutting_down.set()
                try:
                    yield client
                finally:
                    api_module._shutting_down.clear()

    return _ctx()


def test_run_returns_503_when_shutting_down() -> None:
    """POST /run returns 503 when _shutting_down is set."""
    with _shutdown_client_ctx() as client:
        response = client.post("/run", json={"query": "test"})
    assert response.status_code == 503


def test_stream_returns_503_when_shutting_down() -> None:
    """POST /run/stream returns 503 when _shutting_down is set."""
    with _shutdown_client_ctx() as client:
        response = client.post("/run/stream", json={"query": "test"})
    assert response.status_code == 503


def test_research_returns_503_when_shutting_down() -> None:
    """POST /research returns 503 when _shutting_down is set."""
    with _shutdown_client_ctx() as client:
        response = client.post("/research", json={"query": "test"})
    assert response.status_code == 503


def test_drain_middleware_allows_health_during_shutdown() -> None:
    """GET /health should still work when the server is shutting down."""
    with _shutdown_client_ctx() as client:
        response = client.get("/health")
    assert response.status_code == 200


def test_drain_middleware_allows_ready_during_shutdown() -> None:
    """GET /ready should still work when the server is shutting down."""
    with _shutdown_client_ctx() as client:
        response = client.get("/ready")
    assert response.status_code in (200, 503)  # depends on init state


# ---------------------------------------------------------------------------
# SSE error-path and gauge coverage
# ---------------------------------------------------------------------------


def test_run_stream_agent_execution_error_emits_error_event() -> None:
    """When stream_events raises AgentExecutionError, SSE emits an error event."""
    from core.security import RateLimiter

    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)

    async def _error_stream(query):
        raise AgentExecutionError("Research node failed")
        yield  # pragma: no cover

    mock_graph_instance = MagicMock()
    mock_graph_instance.stream_events = _error_stream
    mock_graph_cls = MagicMock(return_value=mock_graph_instance)

    with (
        override_legacy_pack_cls(mock_graph_cls),
        patch("api.main._rate_limiter", permissive),
        patch("api.main.get_shared_llm", return_value=MagicMock(spec=True)),
        patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/run/stream", json={"query": "test query"})

    assert response.status_code == 200

    error_events = [
        json.loads(line[6:])
        for line in response.text.strip().split("\n")
        if line.strip().startswith("data: ")
        and json.loads(line.strip()[6:]).get("type") == "error"
    ]
    assert len(error_events) >= 1, "Expected at least one 'error' SSE event"
    assert (
        "error" in error_events[0]["message"].lower()
        or "pipeline" in error_events[0]["message"].lower()
    )


def test_run_stream_active_pipelines_gauge_decremented_on_error() -> None:
    """active_pipelines gauge must be decremented in the finally block even on error."""
    from core.security import RateLimiter

    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)

    async def _error_stream(query):
        raise AgentExecutionError("forced failure")
        yield  # pragma: no cover

    mock_gauge = MagicMock()
    mock_graph_instance = MagicMock()
    mock_graph_instance.stream_events = _error_stream
    mock_graph_cls = MagicMock(return_value=mock_graph_instance)

    with (
        override_legacy_pack_cls(mock_graph_cls),
        patch("api.main._rate_limiter", permissive),
        patch("api.main.get_shared_llm", return_value=MagicMock(spec=True)),
        patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
        patch("api.main.active_pipelines", mock_gauge),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            client.post("/run/stream", json={"query": "test query"})

    mock_gauge.inc.assert_called_once()
    mock_gauge.dec.assert_called_once()


def test_run_stream_active_pipelines_gauge_decremented_on_success() -> None:
    """active_pipelines gauge must be decremented after a successful stream."""
    from core.security import RateLimiter

    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)

    mock_report = MagicMock(
        executive_summary="s",
        key_insights=[],
        patterns=[],
        implications=[],
        confidence=0.9,
        research_summary="r",
        to_dict=MagicMock(return_value={}),
    )

    async def _success_stream(query):
        yield {"type": "phase_started", "phase": "research"}
        yield {"type": "phase_completed", "phase": "research"}
        yield {"type": "phase_started", "phase": "analysis"}
        yield {"type": "phase_completed", "phase": "analysis"}
        yield {"type": "pipeline_completed", "report": mock_report}

    mock_gauge = MagicMock()
    mock_graph_instance = MagicMock()
    mock_graph_instance.stream_events = _success_stream
    mock_graph_cls = MagicMock(return_value=mock_graph_instance)

    with (
        override_legacy_pack_cls(mock_graph_cls),
        patch("api.main._rate_limiter", permissive),
        patch("api.main.get_shared_llm", return_value=MagicMock(spec=True)),
        patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
        patch("api.main.active_pipelines", mock_gauge),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            client.post("/run/stream", json={"query": "test query"})

    mock_gauge.inc.assert_called_once()
    mock_gauge.dec.assert_called_once()


# ---------------------------------------------------------------------------
# Cost tracking — cost_usd in response and budget exceeded (402)
# ---------------------------------------------------------------------------


def test_run_response_includes_cost_usd() -> None:
    """POST /run must include cost_usd in the response when the pack exposes it."""
    from core.security import RateLimiter

    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)

    mock_report = MagicMock(
        query="What is quantum computing?",
        executive_summary="summary",
        key_insights=[],
        patterns=[],
        implications=[],
        confidence=0.9,
        research_summary="research",
        metadata={},
    )
    mock_report.to_dict = MagicMock(return_value={})

    mock_graph_instance = MagicMock()
    mock_graph_instance.run.return_value = mock_report
    # Expose cost_usd on the pipeline instance so _execute() picks it up
    mock_graph_instance.cost_usd = 0.05
    mock_graph_instance.__enter__ = MagicMock(return_value=mock_graph_instance)
    mock_graph_instance.__exit__ = MagicMock(return_value=False)
    mock_graph_cls = MagicMock(return_value=mock_graph_instance)

    with (
        override_legacy_pack_cls(mock_graph_cls),
        patch("api.main._rate_limiter", permissive),
        patch("api.main.get_shared_llm", return_value=MagicMock(spec=True)),
        patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/run", json={"query": "What is quantum computing?"})

    assert response.status_code == 200
    body = response.json()
    assert "cost_usd" in body
    assert body["cost_usd"] == 0.05


def test_budget_exceeded_returns_402() -> None:
    """POST /run must return 402 when the pipeline raises AgentBudgetExceededError."""
    from core.security import RateLimiter

    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)

    mock_graph_instance = MagicMock()
    mock_graph_instance.run.side_effect = AgentBudgetExceededError("Budget exceeded")
    mock_graph_instance.__enter__ = MagicMock(return_value=mock_graph_instance)
    mock_graph_instance.__exit__ = MagicMock(return_value=False)
    mock_graph_cls = MagicMock(return_value=mock_graph_instance)

    with (
        override_legacy_pack_cls(mock_graph_cls),
        patch("api.main._rate_limiter", permissive),
        patch("api.main.get_shared_llm", return_value=MagicMock(spec=True)),
        patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
    ):
        from api.main import app as _app

        with TestClient(_app, raise_server_exceptions=False) as client:
            response = client.post("/run", json={"query": "What is quantum computing?"})

    assert response.status_code == 402
    assert "detail" in response.json()


def test_research_budget_exceeded_returns_402() -> None:
    """POST /research must return 402 when ResearchAgent raises AgentBudgetExceededError."""
    from core.security import RateLimiter

    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)

    with patch("api.main.ResearchAgent") as mock_cls:
        inst = MagicMock()
        inst.run_structured.side_effect = AgentBudgetExceededError("Budget exceeded")
        mock_cls.return_value = inst

        with (
            patch("api.main._rate_limiter", permissive),
            patch("api.main.get_shared_llm", return_value=MagicMock(spec=True)),
            patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
        ):
            from api.main import app as _app

            with TestClient(_app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/research", json={"query": "Explain distributed systems."}
                )

    assert response.status_code == 402
    assert "detail" in response.json()


# ---------------------------------------------------------------------------
# GET /packs — service discovery
# ---------------------------------------------------------------------------


def test_list_packs_returns_registered_packs(test_client: TestClient) -> None:
    """GET /packs must return 200 with at least the research_analysis pack."""
    response = test_client.get("/packs")

    assert response.status_code == 200
    packs = response.json()
    assert isinstance(packs, list)
    assert len(packs) >= 1

    pack_ids = [p["pack_id"] for p in packs]
    assert "research_analysis" in pack_ids
    assert "research_only" in pack_ids
    assert "summariser" in pack_ids
    assert "analysis_only" in pack_ids

    ra_pack = next(p for p in packs if p["pack_id"] == "research_analysis")
    assert "name" in ra_pack
    assert "description" in ra_pack
    assert "input_schema" in ra_pack
    assert "output_schema" in ra_pack
    assert isinstance(ra_pack["input_schema"], dict)
    assert isinstance(ra_pack["output_schema"], dict)


# ---------------------------------------------------------------------------
# POST /packs/research_analysis/run — per-pack dynamic endpoint
# ---------------------------------------------------------------------------


def test_pack_run_endpoint_exists_for_research_analysis(
    test_client: TestClient,
    mock_analysis_report: MagicMock,
) -> None:
    """POST /packs/research_analysis/run must return 200 with a valid output.

    The dynamic router is wired at lifespan startup by ``test_client``.
    The route closure captures the real ResearchAnalysisPack class — we
    patch both __init__ (no-op) and run() so no real LLM or LangGraph call
    is made.
    """
    from domain_packs.research_analysis.pack import ResearchAnalysisPack

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
    body = response.json()
    assert "executive_summary" in body


def test_pack_run_rejects_empty_query(test_client: TestClient) -> None:
    """POST /packs/research_analysis/run with empty query must return 422.

    Validation is enforced by the ResearchAnalysisInput Pydantic model
    (min_length=1 on the query field) before any LLM call is made.
    """
    response = test_client.post(
        "/packs/research_analysis/run",
        json={"query": ""},
    )

    assert response.status_code == 422


def test_pack_run_endpoint_exists_for_research_only(
    test_client: TestClient,
    mock_research_result: ResearchResult,
) -> None:
    """POST /packs/research_only/run must return 200 with research output fields."""
    from domain_packs.research_only.pack import ResearchOnlyPack

    def _noop_init(self, **kwargs):  # type: ignore[override]
        pass

    with (
        patch.object(ResearchOnlyPack, "__init__", _noop_init),
        patch.object(ResearchOnlyPack, "run", return_value=mock_research_result),
        patch.object(ResearchOnlyPack, "close", return_value=None),
    ):
        response = test_client.post(
            "/packs/research_only/run",
            json={"query": "What is a microservice?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert "summary" in body
    assert "findings" in body


def test_pack_run_endpoint_exists_for_summariser(test_client: TestClient) -> None:
    """POST /packs/summariser/run must return 200 with bullet summary fields."""
    from domain_packs.summariser.pack import SummariserPack
    from domain_packs.summariser.schemas import SummaryOutput

    mock_output = SummaryOutput(original_length=42, bullets=["a", "b", "c"])

    def _noop_init(self, **kwargs):  # type: ignore[override]
        pass

    with (
        patch.object(SummariserPack, "__init__", _noop_init),
        patch.object(SummariserPack, "run_from_input", return_value=mock_output),
        patch.object(SummariserPack, "close", return_value=None),
    ):
        response = test_client.post(
            "/packs/summariser/run",
            json={"text": "Long article about microservices.", "bullet_count": 3},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["bullets"] == ["a", "b", "c"]
    assert body["original_length"] == 42


def test_pack_run_endpoint_exists_for_analysis_only(
    test_client: TestClient,
    mock_analysis_report: AnalysisReport,
) -> None:
    """POST /packs/analysis_only/run must return 200 with analysis output fields."""
    from domain_packs.analysis_only.pack import AnalysisOnlyPack

    def _noop_init(self, **kwargs):  # type: ignore[override]
        pass

    with (
        patch.object(AnalysisOnlyPack, "__init__", _noop_init),
        patch.object(
            AnalysisOnlyPack, "run_from_input", return_value=mock_analysis_report
        ),
        patch.object(AnalysisOnlyPack, "close", return_value=None),
    ):
        response = test_client.post(
            "/packs/analysis_only/run",
            json={
                "query": "What is a microservice?",
                "summary": "Prior research summary.",
                "findings": ["Services are small."],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert "executive_summary" in body
    assert "key_insights" in body


# ---------------------------------------------------------------------------
# Legacy endpoints unchanged
# ---------------------------------------------------------------------------


def test_legacy_run_endpoint_still_works(test_client: TestClient) -> None:
    """POST /run must still return 200 after per-pack router wiring is added."""
    response = test_client.post("/run", json={"query": "What is quantum computing?"})

    assert response.status_code == 200
    body = response.json()
    assert "executive_summary" in body
    assert "key_insights" in body
    assert "confidence" in body


# ---------------------------------------------------------------------------
# X-Pack-Version headers
# ---------------------------------------------------------------------------


def test_pack_run_returns_x_pack_version_used_header(
    test_client: TestClient,
    mock_analysis_report: MagicMock,
) -> None:
    """POST /packs/research_analysis/run must return x-pack-version-used header."""
    from domain_packs.research_analysis.pack import ResearchAnalysisPack

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
    assert "x-pack-version-used" in response.headers


def test_pack_run_with_x_pack_version_header(
    test_client: TestClient,
    mock_analysis_report: MagicMock,
) -> None:
    """POST /packs/research_analysis/run with X-Pack-Version: 1.0 returns x-pack-version-used: 1.0."""
    from domain_packs.research_analysis.pack import ResearchAnalysisPack

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
            headers={"X-Pack-Version": "1.0"},
        )

    assert response.status_code == 200
    assert response.headers.get("x-pack-version-used") == "1.0"


def test_pack_run_unknown_version_returns_404(test_client: TestClient) -> None:
    """POST /packs/research_analysis/run with X-Pack-Version: 99.0 must return 404."""
    response = test_client.post(
        "/packs/research_analysis/run",
        json={"query": "What is a microservice?"},
        headers={"X-Pack-Version": "99.0"},
    )

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /packs/{pack_id}/versions
# ---------------------------------------------------------------------------


def test_list_pack_versions_returns_list(test_client: TestClient) -> None:
    """GET /packs/research_analysis/versions must return list with version and weight keys."""
    response = test_client.get("/packs/research_analysis/versions")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "version" in data[0]
    assert "weight" in data[0]


def test_list_pack_versions_unknown_pack_returns_404(test_client: TestClient) -> None:
    """GET /packs/nonexistent/versions must return 404."""
    response = test_client.get("/packs/nonexistent/versions")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /packs/{pack_id}/versions/{version}/weight
# ---------------------------------------------------------------------------


def test_update_pack_version_weight_succeeds(test_client: TestClient) -> None:
    """PATCH /packs/research_analysis/versions/1.0/weight with valid body returns 200."""
    response = test_client.patch(
        "/packs/research_analysis/versions/1.0/weight",
        json={"weight": 0.5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["pack_id"] == "research_analysis"
    assert body["version"] == "1.0"
    assert body["weight"] == 0.5


def test_update_pack_version_weight_unknown_pack_returns_404(
    test_client: TestClient,
) -> None:
    """PATCH /packs/nonexistent/versions/1.0/weight must return 404."""
    response = test_client.patch(
        "/packs/nonexistent/versions/1.0/weight",
        json={"weight": 0.5},
    )

    assert response.status_code == 404


def test_update_pack_version_weight_invalid_body_returns_422(
    test_client: TestClient,
) -> None:
    """PATCH /packs/research_analysis/versions/1.0/weight with missing weight returns 422."""
    response = test_client.patch(
        "/packs/research_analysis/versions/1.0/weight",
        json={"bad": 0.5},
    )

    assert response.status_code == 422
