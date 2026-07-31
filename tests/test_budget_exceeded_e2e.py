"""tests/test_budget_exceeded_e2e.py — Issue #86: real graph budget enforcement.

Unlike the unit tests in test_graph.py (which mock ``graph.astream_events``
directly) and the API tests in test_api.py (which mock ``stream_events``),
these tests drive the *real* compiled LangGraph with the mock LLM provider
and a nonzero mock cost table (``LLM_COST_TABLE_PATH``), so the full chain —
node -> CostTracker -> AgentBudgetExceededError -> astream_events/graph.invoke
-> pack.run()/stream_events() -> API handler — is actually exercised. If the
node-level or run()-level re-raise fix regressed, these would fail even
though the mocked unit tests above would still pass.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.mock_llm import MOCK_MODEL_ID


@pytest.fixture()
def tiny_budget_mock_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Generator[TestClient, None, None]:
    """Real app, mock LLM provider, nonzero mock pricing, near-zero budget."""
    from core.config import get_settings
    from core.cost import _reset_effective_table

    cost_file = tmp_path / "mock_cost.json"
    # High enough that a single mock call exceeds the budget below.
    cost_file.write_text(json.dumps({MOCK_MODEL_ID: [10.0, 10.0]}), encoding="utf-8")

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("REGULATED_PACKS_ENABLED", "false")
    monkeypatch.setenv("PACK_DEFAULT_BUDGET_USD", "0.001")
    monkeypatch.setenv("LLM_COST_TABLE_PATH", str(cost_file))
    get_settings.cache_clear()
    _reset_effective_table()

    import api.state as api_state

    api_state.shared_llm = None
    api_state.shared_checkpointer = None

    from api.main import app

    with TestClient(app) as client:
        yield client

    api_state.shared_llm = None
    api_state.shared_checkpointer = None
    get_settings.cache_clear()
    _reset_effective_table()


class TestSyncBudgetExceededRealGraph:
    """POST /packs/{id}/run must return 402, not 500, when the real graph
    exceeds budget (regression test for the run() re-wrap bug)."""

    def test_research_only_run_returns_402(
        self, tiny_budget_mock_client: TestClient
    ) -> None:
        response = tiny_budget_mock_client.post(
            "/packs/research_only/run", json={"query": "budget e2e"}
        )
        assert response.status_code == 402, response.text

    def test_analysis_only_run_returns_402(
        self, tiny_budget_mock_client: TestClient
    ) -> None:
        response = tiny_budget_mock_client.post(
            "/packs/analysis_only/run",
            json={
                "query": "budget e2e",
                "findings": ["f1"],
                "summary": "s",
            },
        )
        assert response.status_code == 402, response.text

    def test_legacy_run_returns_402(self, tiny_budget_mock_client: TestClient) -> None:
        response = tiny_budget_mock_client.post("/run", json={"query": "budget e2e"})
        assert response.status_code == 402, response.text


class TestStreamBudgetExceededRealGraph:
    """SSE routes must emit an explicit budget_exceeded error event, driven
    by the real graph rather than a mocked stream_events()."""

    def test_research_only_stream_emits_budget_exceeded_event(
        self, tiny_budget_mock_client: TestClient
    ) -> None:
        with tiny_budget_mock_client.stream(
            "POST",
            "/packs/research_only/run/stream",
            json={"query": "budget e2e"},
        ) as response:
            assert response.status_code == 200
            events: list[dict] = []
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                events.append(json.loads(line.removeprefix("data: ")))

        error_events = [e for e in events if e.get("type") == "error"]
        assert error_events, events
        assert error_events[0].get("code") == "budget_exceeded"

    def test_legacy_run_stream_emits_budget_exceeded_event(
        self, tiny_budget_mock_client: TestClient
    ) -> None:
        with tiny_budget_mock_client.stream(
            "POST",
            "/run/stream",
            json={"query": "budget e2e"},
        ) as response:
            assert response.status_code == 200
            events: list[dict] = []
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                events.append(json.loads(line.removeprefix("data: ")))

        error_events = [e for e in events if e.get("type") == "error"]
        assert error_events, events
        assert error_events[0].get("code") == "budget_exceeded"
