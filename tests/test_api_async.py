"""
tests/test_api_async.py — Async tests for SSE streaming endpoints.

Uses httpx.AsyncClient with ASGITransport to exercise the real async
generator path (asyncio.timeout, async for, event loop) that the
synchronous TestClient cannot cover.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agents.base_agent import AgentExecutionError, AgentTimeoutError
from tests.legacy_pack_override import override_legacy_pack_cls


def _reset_module_state() -> None:
    """Reset ``api.main`` module-level state that leaks between tests.

    Previous tests may have triggered the lifespan shutdown handler, which
    sets ``_shutting_down`` and calls ``_executor.shutdown()``.  Since
    ``httpx.AsyncClient`` with ``ASGITransport`` does NOT trigger the
    lifespan, these values must be reset manually.
    """
    import api.main as api_module

    api_module._shutting_down.clear()
    api_module._executor = None  # run_in_executor(None, …) uses default executor
    api_module._shared_memory = None  # previous lifespan may have closed the DB


def _make_mock_app():
    """Return (app, patches) with all agents mocked — mirrors conftest.test_client."""
    from core.security import RateLimiter

    _reset_module_state()

    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)

    mock_report = MagicMock(
        executive_summary="summary",
        key_insights=["insight"],
        patterns=["pattern"],
        implications=["implication"],
        confidence=0.9,
        research_summary="research",
        to_dict=MagicMock(return_value={}),
    )

    async def _success_stream(query):
        yield {"type": "phase_started", "phase": "research"}
        yield {"type": "phase_completed", "phase": "research"}
        yield {"type": "phase_started", "phase": "analysis"}
        yield {"type": "phase_completed", "phase": "analysis"}
        yield {"type": "pipeline_completed", "report": mock_report}

    mock_graph_instance = MagicMock()
    mock_graph_instance.run.return_value = mock_report
    mock_graph_instance.stream_events = _success_stream
    mock_graph_instance.close.return_value = None
    mock_graph_cls = MagicMock(return_value=mock_graph_instance)

    patches = [
        patch("api.main._rate_limiter", permissive),
        patch("api.main.get_shared_llm", return_value=MagicMock(spec=True)),
        patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
    ]
    for p in patches:
        p.start()

    from api.main import app, get_legacy_pack_cls

    app.dependency_overrides[get_legacy_pack_cls] = lambda: mock_graph_cls

    return app, patches, mock_graph_cls


@pytest.fixture()
async def async_client():
    """Yield an httpx.AsyncClient wired to the FastAPI app with mocked agents."""
    app, patches, _cls = _make_mock_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    for p in patches:
        p.stop()
    from api.main import app, get_legacy_pack_cls

    app.dependency_overrides.pop(get_legacy_pack_cls, None)


# ---------------------------------------------------------------------------
# Happy-path streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_returns_event_stream_content_type(
    async_client: httpx.AsyncClient,
) -> None:
    """POST /run/stream should return content-type text/event-stream."""
    response = await async_client.post("/run/stream", json={"query": "test"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_stream_emits_status_and_done_events(
    async_client: httpx.AsyncClient,
) -> None:
    """SSE stream should emit status, phase_completed, and done events."""
    response = await async_client.post("/run/stream", json={"query": "test"})
    assert response.status_code == 200

    events = []
    for line in response.text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    types = [e["type"] for e in events]
    assert "status" in types
    assert "phase_completed" in types
    assert "done" in types


@pytest.mark.asyncio
async def test_stream_done_event_has_required_fields(
    async_client: httpx.AsyncClient,
) -> None:
    """The done event must contain run_id, session_id, confidence, executive_summary."""
    response = await async_client.post("/run/stream", json={"query": "test"})
    done = [
        json.loads(line.strip()[6:])
        for line in response.text.strip().split("\n")
        if line.strip().startswith("data: ")
        and json.loads(line.strip()[6:]).get("type") == "done"
    ]
    assert len(done) == 1
    assert all(
        k in done[0]
        for k in ("run_id", "session_id", "confidence", "executive_summary")
    )


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_timeout_error_emits_error_event() -> None:
    """AgentTimeoutError during pipeline should emit an SSE error event."""
    from core.security import RateLimiter

    _reset_module_state()
    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)

    async def _error_stream(query):
        raise AgentTimeoutError("budget exceeded")
        yield  # pragma: no cover

    mock_graph = MagicMock()
    mock_graph.stream_events = _error_stream
    mock_graph.close.return_value = None
    mock_graph_cls = MagicMock(return_value=mock_graph)

    with (
        override_legacy_pack_cls(mock_graph_cls),
        patch("api.main._rate_limiter", permissive),
        patch("api.main.get_shared_llm", return_value=MagicMock(spec=True)),
        patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
    ):
        from api.main import app

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post("/run/stream", json={"query": "test"})

    errors = [
        json.loads(line.strip()[6:])
        for line in response.text.strip().split("\n")
        if line.strip().startswith("data: ")
        and json.loads(line.strip()[6:]).get("type") == "error"
    ]
    assert len(errors) >= 1
    assert "timed out" in errors[0]["message"].lower()


@pytest.mark.asyncio
async def test_stream_execution_error_emits_error_event() -> None:
    """AgentExecutionError during pipeline should emit an SSE error event."""
    from core.security import RateLimiter

    _reset_module_state()
    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)

    async def _error_stream(query):
        raise AgentExecutionError("node failed")
        yield  # pragma: no cover

    mock_graph = MagicMock()
    mock_graph.stream_events = _error_stream
    mock_graph.close.return_value = None
    mock_graph_cls = MagicMock(return_value=mock_graph)

    with (
        override_legacy_pack_cls(mock_graph_cls),
        patch("api.main._rate_limiter", permissive),
        patch("api.main.get_shared_llm", return_value=MagicMock(spec=True)),
        patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
    ):
        from api.main import app

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post("/run/stream", json={"query": "test"})

    errors = [
        json.loads(line.strip()[6:])
        for line in response.text.strip().split("\n")
        if line.strip().startswith("data: ")
        and json.loads(line.strip()[6:]).get("type") == "error"
    ]
    assert len(errors) >= 1


@pytest.mark.asyncio
async def test_stream_empty_query_returns_400() -> None:
    """POST /run/stream with empty query returns 400, not 200 SSE."""
    from core.security import RateLimiter

    _reset_module_state()
    permissive = RateLimiter(max_requests=10_000, window_seconds=60.0)

    with (
        patch("api.main._rate_limiter", permissive),
        patch("api.main.get_shared_llm", return_value=MagicMock(spec=True)),
        patch("api.main.get_shared_checkpointer", return_value=MagicMock()),
    ):
        from api.main import app

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post("/run/stream", json={"query": "   "})

    assert response.status_code == 400
