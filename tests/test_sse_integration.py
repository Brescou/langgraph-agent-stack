"""tests/test_sse_integration.py — SSE streaming with real AsyncSqliteSaver.

Exercises the full API path without mocking the checkpointer: real sqlite file,
LLM_PROVIDER=mock, and consumption of Server-Sent Events end-to-end.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def sse_client(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient, None, None]:
    """TestClient with real checkpointer on an isolated sqlite path."""
    from core.config import get_settings

    db_path = tmp_path / "sse_checkpoint.db"
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SQLITE_PATH", str(db_path))
    monkeypatch.delenv("API_KEY", raising=False)
    get_settings.cache_clear()

    import api.state as api_state

    api_state.shared_llm = None
    api_state.shared_checkpointer = None

    from api.main import app

    with TestClient(app) as client:
        yield client

    api_state.shared_llm = None
    api_state.shared_checkpointer = None
    get_settings.cache_clear()


def _collect_sse_events(response: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in response.iter_lines():
        if not line or not line.startswith("data: "):
            continue
        events.append(json.loads(line.removeprefix("data: ")))
    return events


def test_run_stream_sse_with_real_sqlite_checkpointer(sse_client: TestClient) -> None:
    """POST /run/stream must emit phase events and done — not error — with real saver."""
    with sse_client.stream(
        "POST", "/run/stream", json={"query": "What is quantum computing?"}
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        events = _collect_sse_events(response)

    types = [event.get("type") for event in events]
    assert "error" not in types, f"Unexpected error events: {events}"
    assert "phase_started" in types or "phase_completed" in types
    assert types[-1] == "done"


def test_pack_stream_sse_with_real_sqlite_checkpointer(sse_client: TestClient) -> None:
    """POST /packs/research_analysis/run/stream must stream without checkpointer error."""
    with sse_client.stream(
        "POST",
        "/packs/research_analysis/run/stream",
        json={"query": "Summarise LangGraph streaming."},
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        events = _collect_sse_events(response)

    types = [event.get("type") for event in events]
    assert "error" not in types, f"Unexpected error events: {events}"
    assert len(events) >= 1


@pytest.mark.parametrize(
    ("pack_id", "body"),
    [
        ("summariser", {"text": "LangGraph builds stateful multi-agent apps."}),
        (
            "meeting_prep",
            {"company": "Acme", "person": "Jane", "meeting_goal": "discovery"},
        ),
    ],
)
def test_sync_graph_pack_stream_sse_with_async_checkpointer(
    sse_client: TestClient, pack_id: str, body: dict[str, Any]
) -> None:
    """Packs whose stream wraps a sync graph.invoke must not fail on AsyncSqliteSaver.

    Regression test (v0.6.2): summariser and StructuredLLMPack streams called
    ``run_from_input`` on the event loop thread, which AsyncSqliteSaver rejects
    ("Synchronous calls ... only allowed from a different thread"), turning
    every typed pack stream into a generic error event on the default config.
    """
    with sse_client.stream(
        "POST", f"/packs/{pack_id}/run/stream", json=body
    ) as response:
        assert response.status_code == 200
        events = _collect_sse_events(response)

    types = [event.get("type") for event in events]
    assert "error" not in types, f"Unexpected error events: {events}"
    assert types[-1] == "pipeline_completed"
