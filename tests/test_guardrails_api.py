"""tests/test_guardrails_api.py — Guardrails at the API boundaries (mock LLM)."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from control_plane import GuardrailPolicy, PackPolicy, PolicyRegistry

_INPUT_SET = "test_input"
_OUTPUT_SETS = {
    "flag": "test_output_flag",
    "escalate": "test_output_escalate",
    "block": "test_output_block",
}
_BLOCKED_DETAIL = {
    "code": "guardrail_blocked",
    "message": "Request blocked by a guardrail policy.",
}

#: Each input token triggers exactly one input rule.
_INPUT_TOKENS = {"flag": "FLAGME", "escalate": "ESCALATEME", "block": "BLOCKME"}

_TEST_RULES: dict[str, Any] = {
    "rule_sets": {
        _INPUT_SET: {
            "rules": [
                {"id": f"input_{action}", "pattern": token, "action": action}
                for action, token in _INPUT_TOKENS.items()
            ]
        },
        **{
            name: {
                "rules": [{"id": f"any_{action}", "pattern": r"\S", "action": action}]
            }
            for action, name in _OUTPUT_SETS.items()
        },
    }
}

PolicySetter = Callable[..., None]


@pytest.fixture()
def rules_file(tmp_path: Path) -> Path:
    """Rule file with one input set and one output set per action."""
    path = tmp_path / "guardrails.json"
    path.write_text(json.dumps(_TEST_RULES), encoding="utf-8")
    return path


@pytest.fixture()
def set_policy() -> Generator[PolicySetter, None, None]:
    """Bind test rule sets to a pack policy; restore every original afterwards."""
    originals: dict[str, PackPolicy | None] = {}

    def _set(pack_id: str, *, output: str | None = None) -> None:
        original = originals.setdefault(pack_id, PolicyRegistry.get(pack_id))
        base = original or PackPolicy(pack_id=pack_id)
        PolicyRegistry.register(
            dataclasses.replace(
                base,
                guardrails=GuardrailPolicy(
                    input_rule_sets=(_INPUT_SET,),
                    output_rule_sets=(_OUTPUT_SETS[output],) if output else (),
                ),
            )
        )

    yield _set
    for pack_id, original in originals.items():
        PolicyRegistry.register(original or PackPolicy(pack_id=pack_id))


@contextmanager
def _client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool,
    rules_path: Path | None = None,
) -> Iterator[TestClient]:
    """Start the app with the mock LLM and a fresh in-memory review queue."""
    import api.state as api_state
    from api.main import app
    from core.config import get_settings

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("REVIEW_STORE_BACKEND", "memory")
    monkeypatch.setenv("GUARDRAILS_ENABLED", "true" if enabled else "false")
    if rules_path is not None:
        monkeypatch.setenv("GUARDRAILS_RULES_PATH", str(rules_path))
    get_settings.cache_clear()
    api_state.review_store = None
    try:
        with TestClient(app) as client:
            yield client
    finally:
        api_state.review_store = None
        get_settings.cache_clear()


@pytest.fixture()
def client(
    monkeypatch: pytest.MonkeyPatch, rules_file: Path
) -> Generator[TestClient, None, None]:
    with _client(monkeypatch, enabled=True, rules_path=rules_file) as test_client:
        yield test_client


def _reviews() -> list[Any]:
    import api.state as api_state

    assert api_state.review_store is not None
    return api_state.review_store.list_reviews()


def _summariser_body(token: str = "") -> dict[str, Any]:
    return {"text": f"Quarterly revenue grew while costs held flat. {token}".strip()}


def _sse_events(text: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def _sample(name: str, labels: dict[str, str]) -> float:
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value(name, labels) or 0.0


# ---------------------------------------------------------------------------
# Flag off
# ---------------------------------------------------------------------------


def test_flag_off_never_evaluates_on_any_boundary(
    monkeypatch: pytest.MonkeyPatch, rules_file: Path, set_policy: PolicySetter
) -> None:
    for pack_id in ("summariser", "research_analysis", "research_only"):
        set_policy(pack_id, output="block")

    with (
        _client(monkeypatch, enabled=False, rules_path=rules_file) as client,
        patch("api.guardrails.evaluate") as evaluate,
    ):
        body = _summariser_body("BLOCKME")
        assert client.post("/packs/summariser/run", json=body).status_code == 200
        stream = client.post("/packs/summariser/run/stream", json=body)
        assert _sse_events(stream.text)[-1]["type"] == "pipeline_completed"
        assert client.post("/run", json={"query": "BLOCKME"}).status_code == 200
        legacy_stream = client.post("/run/stream", json={"query": "BLOCKME"})
        assert _sse_events(legacy_stream.text)[-1]["type"] == "done"
        assert client.post("/research", json={"query": "BLOCKME"}).status_code == 200

    evaluate.assert_not_called()


# ---------------------------------------------------------------------------
# Typed /packs/{id}/run (also the MCP path)
# ---------------------------------------------------------------------------


def test_typed_run_input_flag_continues_without_review(
    client: TestClient, set_policy: PolicySetter
) -> None:
    set_policy("summariser")
    labels = {
        "pack_id": "summariser",
        "phase": "input",
        "rule_set": _INPUT_SET,
        "rule_id": "input_flag",
        "action": "flag",
    }
    before = _sample("guardrail_findings_total", labels)

    response = client.post("/packs/summariser/run", json=_summariser_body("FLAGME"))

    assert response.status_code == 200
    assert _reviews() == []
    assert _sample("guardrail_findings_total", labels) == before + 1


def test_typed_run_input_escalate_queues_review_with_reason(
    client: TestClient, set_policy: PolicySetter
) -> None:
    set_policy("summariser")

    response = client.post("/packs/summariser/run", json=_summariser_body("ESCALATEME"))

    assert response.status_code == 200
    reviews = _reviews()
    assert len(reviews) == 1
    assert reviews[0].pack_id == "summariser"
    assert reviews[0].reason == f"guardrail: {_INPUT_SET}/input_escalate (input)"

    listed = client.get("/reviews").json()
    assert listed["reviews"][0]["reason"] == reviews[0].reason


def test_typed_run_input_block_skips_llm_history_and_idempotency(
    client: TestClient, set_policy: PolicySetter
) -> None:
    import api.state as api_state
    from api import pack_execution

    set_policy("summariser")
    outcome_labels = {
        "pack_id": "summariser",
        "version": "1.0",
        "outcome": "guardrail_blocked",
    }
    before = _sample("pack_runs_total", outcome_labels)

    with (
        patch.object(
            pack_execution, "invoke_pack_run", wraps=pack_execution.invoke_pack_run
        ) as invoke,
        patch.object(
            pack_execution,
            "save_run_best_effort",
            wraps=pack_execution.save_run_best_effort,
        ) as save,
    ):
        response = client.post(
            "/packs/summariser/run",
            json=_summariser_body("BLOCKME"),
            headers={"Idempotency-Key": "guardrail-input-block"},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == {**_BLOCKED_DETAIL, "phase": "input"}
    assert "BLOCKME" not in response.text
    invoke.assert_not_called()
    save.assert_not_called()
    assert api_state.get_shared_idempotency_store().get("guardrail-input-block") is None
    assert _sample("pack_runs_total", outcome_labels) == before + 1


def test_typed_run_output_flag_returns_response(
    client: TestClient, set_policy: PolicySetter
) -> None:
    set_policy("summariser", output="flag")

    response = client.post("/packs/summariser/run", json=_summariser_body())

    assert response.status_code == 200
    assert response.json()["bullets"]
    assert _reviews() == []


def test_typed_run_output_escalate_returns_response_and_queues_review(
    client: TestClient, set_policy: PolicySetter
) -> None:
    set_policy("summariser", output="escalate")

    response = client.post("/packs/summariser/run", json=_summariser_body())

    assert response.status_code == 200
    reviews = _reviews()
    assert len(reviews) == 1
    assert reviews[0].reason == "guardrail: test_output_escalate/any_escalate (output)"


def test_typed_run_both_phases_escalate_queue_a_single_review(
    client: TestClient, set_policy: PolicySetter
) -> None:
    set_policy("summariser", output="escalate")

    response = client.post("/packs/summariser/run", json=_summariser_body("ESCALATEME"))

    assert response.status_code == 200
    reviews = _reviews()
    assert len(reviews) == 1
    assert reviews[0].reason == (
        f"guardrail: {_INPUT_SET}/input_escalate (input); "
        "test_output_escalate/any_escalate (output)"
    )


def test_typed_run_output_block_withholds_response_and_retry_reruns(
    client: TestClient, set_policy: PolicySetter
) -> None:
    from api import pack_execution

    set_policy("summariser", output="block")

    with (
        patch.object(
            pack_execution, "invoke_pack_run", wraps=pack_execution.invoke_pack_run
        ) as invoke,
        patch.object(
            pack_execution,
            "save_run_best_effort",
            wraps=pack_execution.save_run_best_effort,
        ) as save,
    ):
        responses = [
            client.post(
                "/packs/summariser/run",
                json=_summariser_body(),
                headers={"Idempotency-Key": "guardrail-output-block"},
            )
            for _ in range(2)
        ]

    for response in responses:
        assert response.status_code == 502
        assert response.json()["detail"] == {**_BLOCKED_DETAIL, "phase": "output"}
    assert invoke.call_count == 2
    save.assert_not_called()
    assert _reviews() == []


# ---------------------------------------------------------------------------
# Typed /packs/{id}/run/stream
# ---------------------------------------------------------------------------


def test_typed_stream_input_block_is_http_422(
    client: TestClient, set_policy: PolicySetter
) -> None:
    set_policy("summariser")

    response = client.post(
        "/packs/summariser/run/stream", json=_summariser_body("BLOCKME")
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {**_BLOCKED_DETAIL, "phase": "input"}


def test_typed_stream_output_block_replaces_final_event(
    client: TestClient, set_policy: PolicySetter
) -> None:
    set_policy("summariser", output="block")

    response = client.post("/packs/summariser/run/stream", json=_summariser_body())

    assert response.status_code == 200
    events = _sse_events(response.text)
    assert all(event["type"] != "pipeline_completed" for event in events)
    assert events[-1] == {"type": "error", "phase": "output", **_BLOCKED_DETAIL}
    assert _reviews() == []


def test_typed_stream_output_escalate_queues_review(
    client: TestClient, set_policy: PolicySetter
) -> None:
    set_policy("summariser", output="escalate")

    response = client.post("/packs/summariser/run/stream", json=_summariser_body())

    assert _sse_events(response.text)[-1]["type"] == "pipeline_completed"
    reviews = _reviews()
    assert len(reviews) == 1
    assert reviews[0].reason == "guardrail: test_output_escalate/any_escalate (output)"


# ---------------------------------------------------------------------------
# Legacy routes
# ---------------------------------------------------------------------------


def test_legacy_run_input_block(client: TestClient, set_policy: PolicySetter) -> None:
    set_policy("research_analysis")

    response = client.post("/run", json={"query": "Explain BLOCKME please"})

    assert response.status_code == 422
    assert response.json()["detail"] == {**_BLOCKED_DETAIL, "phase": "input"}


def test_legacy_run_output_block(client: TestClient, set_policy: PolicySetter) -> None:
    set_policy("research_analysis", output="block")

    response = client.post("/run", json={"query": "Explain vector databases"})

    assert response.status_code == 502
    assert response.json()["detail"] == {**_BLOCKED_DETAIL, "phase": "output"}


def test_legacy_run_escalate_queues_review(
    client: TestClient, set_policy: PolicySetter
) -> None:
    set_policy("research_analysis")

    response = client.post("/run", json={"query": "Explain ESCALATEME please"})

    assert response.status_code == 200
    reviews = _reviews()
    assert len(reviews) == 1
    assert reviews[0].pack_id == "research_analysis"


def test_legacy_stream_output_block_replaces_done(
    client: TestClient, set_policy: PolicySetter
) -> None:
    set_policy("research_analysis", output="block")

    response = client.post("/run/stream", json={"query": "Explain vector databases"})

    events = _sse_events(response.text)
    assert all(event["type"] != "done" for event in events)
    assert events[-1] == {"type": "error", "phase": "output", **_BLOCKED_DETAIL}


def test_legacy_stream_input_block_is_http_422(
    client: TestClient, set_policy: PolicySetter
) -> None:
    set_policy("research_analysis")

    response = client.post("/run/stream", json={"query": "Explain BLOCKME please"})

    assert response.status_code == 422


def test_legacy_research_uses_research_only_policy(
    client: TestClient, set_policy: PolicySetter
) -> None:
    set_policy("research_only")

    response = client.post("/research", json={"query": "Explain BLOCKME please"})

    assert response.status_code == 422
    assert response.json()["detail"] == {**_BLOCKED_DETAIL, "phase": "input"}


# ---------------------------------------------------------------------------
# MCP (shares execute_typed_pack_run)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_input_block_maps_to_invalid_params(
    monkeypatch: pytest.MonkeyPatch, rules_file: Path, set_policy: PolicySetter
) -> None:
    pytest.importorskip("mcp")
    from mcp import Client
    from mcp.shared.exceptions import MCPError
    from mcp.types import INVALID_PARAMS

    from api.mcp_server import build_mcp_server

    set_policy("summariser")
    # Build the server directly: MCP_SERVER_ENABLED would mount /mcp on the
    # shared app for the rest of the session.
    with _client(monkeypatch, enabled=True, rules_path=rules_file):
        async with Client(build_mcp_server()) as session:
            with pytest.raises(MCPError) as excinfo:
                await session.call_tool("summariser", _summariser_body("BLOCKME"))

    assert excinfo.value.code == INVALID_PARAMS
    assert "guardrail_blocked" in excinfo.value.message
    assert "BLOCKME" not in excinfo.value.message
