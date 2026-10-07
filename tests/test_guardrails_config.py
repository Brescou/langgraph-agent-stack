"""tests/test_guardrails_config.py — Guardrail settings, policy resolution, startup."""

from __future__ import annotations

import dataclasses
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from control_plane import GuardrailPolicy, PackPolicy, PolicyRegistry
from control_plane.enforce import guardrail_rule_sets, validate_guardrail_policies
from core.guardrails import GuardrailConfigError, load_builtin_rule_sets

_PACK_ID = "summariser"


@pytest.fixture()
def summariser_guardrails() -> Generator[None, None, None]:
    """Subscribe the summariser policy to built-in sets; restore it afterwards."""
    original = PolicyRegistry.get(_PACK_ID)
    assert original is not None
    PolicyRegistry.register(
        dataclasses.replace(
            original,
            guardrails=GuardrailPolicy(
                input_rule_sets=("prompt_injection", "pii_basic"),
                output_rule_sets=("pii_basic",),
            ),
        )
    )
    yield
    PolicyRegistry.register(original)


@pytest.fixture()
def unknown_set_policy() -> Generator[None, None, None]:
    original = PolicyRegistry.get(_PACK_ID)
    assert original is not None
    PolicyRegistry.register(
        dataclasses.replace(
            original, guardrails=GuardrailPolicy(input_rule_sets=("no_such_set",))
        )
    )
    yield
    PolicyRegistry.register(original)


@pytest.fixture()
def guardrails_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Mock provider and a clean settings cache around each startup test."""
    from core.config import get_settings

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("API_KEY", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_policy_guardrails_default_to_empty() -> None:
    policy = PackPolicy(pack_id="anything")

    assert policy.guardrails == GuardrailPolicy()
    assert policy.guardrails.input_rule_sets == ()
    assert policy.guardrails.output_rule_sets == ()


def test_no_builtin_policy_subscribes_to_guardrails() -> None:
    for pack_id in PolicyRegistry.list_policies():
        policy = PolicyRegistry.get(pack_id)
        assert policy is not None
        assert policy.guardrails == GuardrailPolicy(), pack_id


def test_guardrail_rule_sets_resolves_names_in_order(
    summariser_guardrails: None,
) -> None:
    loaded = load_builtin_rule_sets()

    assert [
        rule_set.name for rule_set in guardrail_rule_sets(_PACK_ID, "input", loaded)
    ] == ["prompt_injection", "pii_basic"]
    assert [
        rule_set.name for rule_set in guardrail_rule_sets(_PACK_ID, "output", loaded)
    ] == ["pii_basic"]


def test_guardrail_rule_sets_empty_without_policy() -> None:
    assert guardrail_rule_sets("unregistered_pack", "input", {}) == []


def test_validate_guardrail_policies_names_pack_and_set(
    unknown_set_policy: None,
) -> None:
    with pytest.raises(GuardrailConfigError) as exc_info:
        validate_guardrail_policies(load_builtin_rule_sets())

    assert "'summariser'" in str(exc_info.value)
    assert "'no_such_set'" in str(exc_info.value)


def test_startup_disabled_ignores_invalid_rule_file(
    guardrails_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad = tmp_path / "rules.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("GUARDRAILS_RULES_PATH", str(bad))

    import api.state as api_state
    from api.main import app

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert api_state.guardrail_rule_sets is None


def test_startup_enabled_loads_rule_sets(
    guardrails_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GUARDRAILS_ENABLED", "true")

    import api.state as api_state
    from api.main import app

    with TestClient(app):
        assert api_state.guardrail_rule_sets is not None
        assert "pii_basic" in api_state.guardrail_rule_sets

    monkeypatch.delenv("GUARDRAILS_ENABLED")
    from core.config import get_settings

    get_settings.cache_clear()
    with TestClient(app):
        assert api_state.guardrail_rule_sets is None


def test_startup_enabled_fails_on_invalid_rule_file(
    guardrails_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad = tmp_path / "rules.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("GUARDRAILS_ENABLED", "true")
    monkeypatch.setenv("GUARDRAILS_RULES_PATH", str(bad))

    from api.main import app

    with pytest.raises(GuardrailConfigError, match="rules.json"):
        with TestClient(app):
            pass


def test_startup_enabled_fails_on_unknown_policy_set(
    guardrails_env: None,
    monkeypatch: pytest.MonkeyPatch,
    unknown_set_policy: None,
) -> None:
    monkeypatch.setenv("GUARDRAILS_ENABLED", "true")

    from api.main import app

    with pytest.raises(GuardrailConfigError, match="no_such_set"):
        with TestClient(app):
            pass
