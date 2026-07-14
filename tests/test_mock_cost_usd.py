"""tests/test_mock_cost_usd.py — Issue #88: consistent cost_usd in mock mode."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from core.cost import (
    CostTracker,
    UnknownModelPricingError,
    compute_call_cost,
    provider_from_model_id,
    resolve_model_pricing,
)
from core.mock_llm import MOCK_MODEL_ID, MockProviderChatModel


@pytest.fixture()
def mock_provider_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    """Real app with LLM_PROVIDER=mock."""
    from core.config import get_settings

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("REGULATED_PACKS_ENABLED", "false")
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


class TestMockUsageMetadata:
    def test_mock_model_emits_deterministic_usage(self) -> None:
        model = MockProviderChatModel()
        result_a = model.invoke([HumanMessage(content="hello world")])
        result_b = model.invoke([HumanMessage(content="hello world")])

        assert result_a.usage_metadata is not None
        assert result_b.usage_metadata is not None
        assert (
            result_a.usage_metadata["input_tokens"]
            == result_b.usage_metadata["input_tokens"]
        )
        assert result_a.usage_metadata["output_tokens"] >= 1
        assert result_a.usage_metadata["input_tokens"] >= 1
        assert (result_a.response_metadata or {}).get("model_name") == MOCK_MODEL_ID


class TestMockPricing:
    def test_mock_model_has_zero_pricing(self) -> None:
        pricing = resolve_model_pricing(MOCK_MODEL_ID)
        assert pricing is not None
        assert pricing.input_per_1k == 0.0
        assert pricing.output_per_1k == 0.0
        assert compute_call_cost(MOCK_MODEL_ID, 1000, 1000) == 0.0
        assert provider_from_model_id(MOCK_MODEL_ID) == "mock"

    def test_mock_model_never_raises_unknown_pricing(self) -> None:
        # Even in production-strict mode the mock entry must resolve.
        cost = compute_call_cost(MOCK_MODEL_ID, 10, 10, strict=True)
        assert cost == 0.0
        # Sanity: unknown models still raise when strict.
        with pytest.raises(UnknownModelPricingError):
            compute_call_cost("totally-unknown-model-xyz", 10, 10, strict=True)


class TestMockCostTrackerAccounting:
    def test_cost_tracker_records_mock_usage_at_zero_cost(self) -> None:
        tracker = CostTracker()
        model = MockProviderChatModel().with_config({"callbacks": [tracker]})
        model.invoke([HumanMessage(content="track me please")])

        assert tracker.input_tokens >= 1
        assert tracker.output_tokens >= 1
        assert tracker.total_cost_usd == 0.0


class TestMockCostUsdApiConsistency:
    def test_legacy_run_and_typed_pack_return_zero(
        self, mock_provider_client: TestClient
    ) -> None:
        legacy = mock_provider_client.post("/run", json={"query": "cost consistency"})
        assert legacy.status_code == 200, legacy.text
        assert legacy.json()["cost_usd"] == 0.0

        pack = mock_provider_client.post(
            "/packs/meeting_prep/run",
            json={
                "company": "Acme",
                "person": "Jane",
                "meeting_goal": "discovery",
            },
        )
        assert pack.status_code == 200, pack.text
        assert pack.json()["cost_usd"] == 0.0

    def test_meeting_prep_stream_final_payload_has_zero_cost(
        self, mock_provider_client: TestClient
    ) -> None:
        with mock_provider_client.stream(
            "POST",
            "/packs/meeting_prep/run/stream",
            json={
                "company": "Acme",
                "person": "Jane",
                "meeting_goal": "discovery",
            },
        ) as response:
            assert response.status_code == 200, response.text
            events: list[dict] = []
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                events.append(json.loads(line.removeprefix("data: ")))

        completed = [e for e in events if e.get("type") == "pipeline_completed"]
        assert completed, events
        result = completed[-1].get("result") or {}
        assert result.get("cost_usd") == 0.0


class TestMockBudgetEnforcement:
    def test_budget_exceeded_returns_402_via_cost_table_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Nonzero mock pricing + tiny budget must yield HTTP 402 without an API key."""
        from core.config import get_settings
        from core.cost import _reset_effective_table

        cost_file = tmp_path / "mock_cost.json"
        # High enough that a single mock call exceeds $0.001.
        cost_file.write_text(
            json.dumps({MOCK_MODEL_ID: [10.0, 10.0]}),
            encoding="utf-8",
        )

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

        try:
            with TestClient(app) as client:
                response = client.post(
                    "/packs/meeting_prep/run",
                    json={
                        "company": "Acme",
                        "person": "Jane",
                        "meeting_goal": "discovery",
                    },
                )
            assert response.status_code == 402, response.text
        finally:
            api_state.shared_llm = None
            api_state.shared_checkpointer = None
            get_settings.cache_clear()
            _reset_effective_table()
