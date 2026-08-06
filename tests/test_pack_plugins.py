"""
tests/test_pack_plugins.py — Distributable pack discovery via entry points.

Covers contract validation (strict schemas required for third-party code),
allowlist gating (deny by default), broken-plugin resilience, registration
incl. built-in collision rejection, and the settings allowlist parsing.
Entry points are faked — nothing is installed or imported from outside the
repo.
"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ConfigDict

from examples.custom_pack.pack import EchoPack
from pack_kernel.builtin_packs import register_builtin_packs
from pack_kernel.plugins import (
    PluginPackError,
    discover_plugin_packs,
    register_plugin_packs,
    validate_plugin_pack_class,
)
from pack_kernel.registry import PackRegistry

register_builtin_packs()


# ---------------------------------------------------------------------------
# Helpers — fake entry points and candidate pack classes
# ---------------------------------------------------------------------------


class _FakeEntryPoint:
    def __init__(self, name: str, target: Any, *, raises: bool = False) -> None:
        self.name = name
        self.value = f"fake.module:{name}"
        self._target = target
        self._raises = raises

    def load(self) -> Any:
        if self._raises:
            raise ImportError("broken plugin module")
        return self._target


class _StrictIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class _StrictOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    echoed: str


class _GoodPlugin(EchoPack):
    pack_id: ClassVar[str] = "plugin_good"
    name: ClassVar[str] = "Good plugin"
    description: ClassVar[str] = "A valid plugin pack."
    version: ClassVar[str] = "1.0"
    input_schema: ClassVar[type[BaseModel]] = _StrictIn
    output_schema: ClassVar[type[BaseModel]] = _StrictOut


class _LaxOut(BaseModel):  # no extra="forbid"
    echoed: str


class _LaxSchemaPlugin(_GoodPlugin):
    pack_id: ClassVar[str] = "plugin_lax"
    output_schema: ClassVar[type[BaseModel]] = _LaxOut


class _CollidingPlugin(_GoodPlugin):
    pack_id: ClassVar[str] = "summariser"  # built-in id


def _patched_eps(*eps: _FakeEntryPoint):
    return patch("pack_kernel.plugins._iter_entry_points", return_value=tuple(eps))


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    PackRegistry._reset()
    register_builtin_packs()


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------


class TestValidatePluginPackClass:
    def test_valid_pack_accepted(self) -> None:
        assert validate_plugin_pack_class(_GoodPlugin) is _GoodPlugin

    def test_non_pack_rejected(self) -> None:
        with pytest.raises(PluginPackError, match="BaseDomainPack"):
            validate_plugin_pack_class(object)

    @pytest.mark.parametrize("attr", ["pack_id", "name", "description", "version"])
    def test_missing_metadata_rejected(self, attr: str) -> None:
        bad = type("Bad", (_GoodPlugin,), {attr: ""})
        with pytest.raises(PluginPackError, match=attr):
            validate_plugin_pack_class(bad)

    def test_lax_schema_rejected(self) -> None:
        with pytest.raises(PluginPackError, match="extra"):
            validate_plugin_pack_class(_LaxSchemaPlugin)

    def test_non_model_schema_rejected(self) -> None:
        bad = type("Bad", (_GoodPlugin,), {"input_schema": dict})
        with pytest.raises(PluginPackError, match="BaseModel"):
            validate_plugin_pack_class(bad)


# ---------------------------------------------------------------------------
# Discovery — allowlist gating and resilience
# ---------------------------------------------------------------------------


class TestDiscoverPluginPacks:
    def test_empty_allowlist_loads_nothing(self) -> None:
        with _patched_eps(_FakeEntryPoint("good", _GoodPlugin)):
            assert discover_plugin_packs(()) == []

    def test_allowlisted_plugin_loads(self) -> None:
        with _patched_eps(_FakeEntryPoint("good", _GoodPlugin)):
            assert discover_plugin_packs(("good",)) == [_GoodPlugin]

    def test_non_allowlisted_plugin_skipped(self) -> None:
        with _patched_eps(
            _FakeEntryPoint("good", _GoodPlugin),
            _FakeEntryPoint("other", _GoodPlugin),
        ):
            assert discover_plugin_packs(("good",)) == [_GoodPlugin]

    def test_broken_import_skipped_not_raised(self, caplog) -> None:
        with _patched_eps(
            _FakeEntryPoint("broken", None, raises=True),
            _FakeEntryPoint("good", _GoodPlugin),
        ):
            loaded = discover_plugin_packs(("broken", "good"))
        assert loaded == [_GoodPlugin]
        assert any("failed to import" in r.message for r in caplog.records)

    def test_contract_violation_skipped(self, caplog) -> None:
        with _patched_eps(_FakeEntryPoint("lax", _LaxSchemaPlugin)):
            assert discover_plugin_packs(("lax",)) == []
        assert any("contract violation" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegisterPluginPacks:
    def test_disabled_by_default(self) -> None:
        with _patched_eps(_FakeEntryPoint("good", _GoodPlugin)):
            assert register_plugin_packs(enabled=False, allowlist=("good",)) == []
        assert "plugin_good" not in PackRegistry.list_packs()

    def test_enabled_without_allowlist_loads_nothing(self, caplog) -> None:
        with _patched_eps(_FakeEntryPoint("good", _GoodPlugin)):
            assert register_plugin_packs(enabled=True, allowlist=()) == []
        assert any("deny by default" in r.message for r in caplog.records)

    def test_registers_and_serves_versions(self) -> None:
        with _patched_eps(_FakeEntryPoint("good", _GoodPlugin)):
            registered = register_plugin_packs(enabled=True, allowlist=("good",))
        assert registered == ["plugin_good"]
        assert PackRegistry.get("plugin_good") is _GoodPlugin
        in_schema, out_schema = PackRegistry.get_schemas("plugin_good")
        assert in_schema is _StrictIn
        assert out_schema is _StrictOut

    def test_builtin_collision_rejected(self, caplog) -> None:
        with _patched_eps(_FakeEntryPoint("collide", _CollidingPlugin)):
            registered = register_plugin_packs(enabled=True, allowlist=("collide",))
        assert registered == []
        # The built-in summariser must be untouched.
        assert PackRegistry.get("summariser") is not _CollidingPlugin
        assert any("collides" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Settings — allowlist parsing
# ---------------------------------------------------------------------------


def test_resolved_pack_plugins_allowlist_parsing() -> None:
    from core.config import Settings

    settings = Settings(
        llm_provider="anthropic",
        anthropic_api_key="sk-ant-test123456789012345",
        pack_plugins_allowlist=" good , other ,, good ",
    )
    assert settings.resolved_pack_plugins_allowlist == ("good", "other")
    assert settings.pack_plugins_enabled is False  # opt-in default


# ---------------------------------------------------------------------------
# Serialization — strict third-party schemas must survive the response model
# ---------------------------------------------------------------------------


class _StrictOutputNoCost(BaseModel):
    """A compliant plugin output schema that does not track cost."""

    model_config = ConfigDict(extra="forbid")

    answer: str


class _StrictOutputWithCost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    cost_usd: float = 0.0


def test_serialize_does_not_inject_cost_into_schema_without_the_field() -> None:
    """Plugin packs must not get an undeclared ``cost_usd`` bolted on.

    Regression guard: ``plugins.py`` *requires* ``extra="forbid"`` on
    third-party schemas, so unconditionally injecting ``cost_usd`` made every
    successful run of a compliant plugin pack fail response validation (500).
    """
    from api.pack_execution import serialize_pack_result

    data = serialize_pack_result(
        _StrictOutputNoCost(answer="ok"), _StrictOutputNoCost, 0.0
    )

    assert data == {"answer": "ok"}
    # The declared response model must accept it unchanged.
    assert _StrictOutputNoCost.model_validate(data).answer == "ok"


def test_serialize_still_injects_cost_when_schema_declares_it() -> None:
    """Packs that declare ``cost_usd`` keep receiving the run cost."""
    from api.pack_execution import serialize_pack_result

    data = serialize_pack_result(
        _StrictOutputWithCost(answer="ok"), _StrictOutputWithCost, 0.25
    )

    assert data["cost_usd"] == 0.25
