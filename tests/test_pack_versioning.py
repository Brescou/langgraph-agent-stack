"""
tests/test_pack_versioning.py — Tests for PackVersion dataclass and
PackRegistry multi-version support.

Covers:
- PackVersion dataclass fields and defaults
- Registering two versions of the same pack_id
- Replacing a pack when same pack_id + version is re-registered
- get() returns the correct class when only one version exists
- get() probabilistically selects from multiple versions using weights
- _get_versions() returns the full list, raises KeyError for unknown packs
"""

from __future__ import annotations

import random
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from pack_kernel.base_pack import BaseDomainPack
from pack_kernel.registry import PackRegistry, PackVersion

# ---------------------------------------------------------------------------
# Minimal concrete packs for testing
# ---------------------------------------------------------------------------


class _PackV1(BaseDomainPack):
    pack_id = "test_versioned_pack"
    name = "Test Pack"
    description = "A pack used in versioning tests."
    version = "1.0"

    def run(self, query: str) -> Any:
        return {"version": "1.0"}

    async def arun(self, query: str) -> Any:
        return {"version": "1.0"}

    async def _iter_stream_events(self, query: str) -> AsyncGenerator[dict[str, Any], None]:
        yield {"type": "test", "version": "1.0"}


class _PackV2(BaseDomainPack):
    pack_id = "test_versioned_pack"
    name = "Test Pack"
    description = "A pack used in versioning tests."
    version = "2.0"

    def run(self, query: str) -> Any:
        return {"version": "2.0"}

    async def arun(self, query: str) -> Any:
        return {"version": "2.0"}

    async def _iter_stream_events(self, query: str) -> AsyncGenerator[dict[str, Any], None]:
        yield {"type": "test", "version": "2.0"}


class _PackV1Replacement(BaseDomainPack):
    """Same pack_id + version as _PackV1 — used to test replacement."""

    pack_id = "test_versioned_pack"
    name = "Test Pack"
    description = "A pack used in versioning tests."
    version = "1.0"

    def run(self, query: str) -> Any:
        return {"version": "1.0-replaced"}

    async def arun(self, query: str) -> Any:
        return {"version": "1.0-replaced"}

    async def _iter_stream_events(self, query: str) -> AsyncGenerator[dict[str, Any], None]:
        yield {"type": "test", "version": "1.0-replaced"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def clean_registry():
    """Reset the registry before and after each versioning test.

    This fixture is NOT autouse — each test that requires an isolated
    registry must request it explicitly as a parameter.  Making it autouse
    would clear PackRegistry for *all* tests in the session, which breaks
    test_api.py and test_pack_contracts.py because the Python import cache
    prevents pack_kernel/__init__.py from re-registering ResearchAnalysisPack.

    Teardown re-registers ResearchAnalysisPack so that tests running after
    this fixture (e.g. test_api.py) still find the registry in its normal
    state.
    """
    from pack_kernel.builtin_packs import register_builtin_packs

    PackRegistry._reset()
    yield
    PackRegistry._reset()
    register_builtin_packs()


# ---------------------------------------------------------------------------
# PackVersion dataclass
# ---------------------------------------------------------------------------


def test_pack_version_defaults(clean_registry) -> None:
    """PackVersion.weight defaults to 1.0."""
    pv = PackVersion(version="1.0", pack_cls=_PackV1)
    assert pv.version == "1.0"
    assert pv.pack_cls is _PackV1
    assert pv.weight == 1.0


def test_pack_version_custom_weight(clean_registry) -> None:
    """PackVersion.weight can be overridden."""
    pv = PackVersion(version="1.0", pack_cls=_PackV1, weight=0.3)
    assert pv.weight == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Register two different versions for the same pack_id
# ---------------------------------------------------------------------------


def test_register_same_pack_id_different_versions(clean_registry) -> None:
    """Registering two classes with the same pack_id but different versions
    stores both in _get_versions."""
    PackRegistry.register(_PackV1)
    PackRegistry.register(_PackV2)

    versions = PackRegistry._get_versions("test_versioned_pack")
    assert len(versions) == 2
    version_strings = {pv.version for pv in versions}
    assert version_strings == {"1.0", "2.0"}


def test_register_same_pack_id_different_versions_classes_match(clean_registry) -> None:
    """Each PackVersion entry maps to the correct pack class."""
    PackRegistry.register(_PackV1)
    PackRegistry.register(_PackV2)

    versions = PackRegistry._get_versions("test_versioned_pack")
    by_version = {pv.version: pv.pack_cls for pv in versions}
    assert by_version["1.0"] is _PackV1
    assert by_version["2.0"] is _PackV2


# ---------------------------------------------------------------------------
# Replacing a pack when same pack_id + version is re-registered
# ---------------------------------------------------------------------------


def test_register_same_pack_id_same_version_replaces(clean_registry) -> None:
    """Re-registering the same pack_id+version replaces the existing entry;
    the list still has exactly one entry."""
    PackRegistry.register(_PackV1)
    PackRegistry.register(_PackV1Replacement)  # same pack_id + version "1.0"

    versions = PackRegistry._get_versions("test_versioned_pack")
    assert len(versions) == 1
    assert versions[0].pack_cls is _PackV1Replacement


def test_register_same_pack_id_same_version_emits_warning(
    clean_registry, caplog
) -> None:
    """Re-registering the same pack_id+version logs a warning."""
    import logging

    PackRegistry.register(_PackV1)
    with caplog.at_level(logging.WARNING, logger="pack_kernel.registry"):
        PackRegistry.register(_PackV1Replacement)

    assert any("replacing" in record.message.lower() for record in caplog.records), (
        f"Expected a replacement warning, got: {[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# get() — single version path (no random.choices)
# ---------------------------------------------------------------------------


def test_get_uses_only_version_when_one_registered(clean_registry) -> None:
    """get() returns the correct class when only one version is registered
    (no random selection should occur)."""
    PackRegistry.register(_PackV1)
    assert PackRegistry.get("test_versioned_pack") is _PackV1


def test_get_raises_for_unknown_pack(clean_registry) -> None:
    """get() raises KeyError for an unregistered pack_id."""
    with pytest.raises(KeyError, match="not registered"):
        PackRegistry.get("no_such_pack")


# ---------------------------------------------------------------------------
# get() — multiple versions path (random.choices with weights)
# ---------------------------------------------------------------------------


def test_get_selects_from_multiple_versions(clean_registry) -> None:
    """With two versions registered, repeated get() calls return both classes
    (assuming equal weights, the probability of always picking the same one
    over 100 calls is negligibly small)."""
    PackRegistry.register(_PackV1)
    PackRegistry.register(_PackV2)

    results = {PackRegistry.get("test_versioned_pack") for _ in range(100)}
    assert _PackV1 in results
    assert _PackV2 in results


def test_get_respects_zero_weight(clean_registry) -> None:
    """A PackVersion with weight=0 must never be selected."""
    pv1 = PackVersion(version="1.0", pack_cls=_PackV1, weight=1.0)
    pv2 = PackVersion(version="2.0", pack_cls=_PackV2, weight=0.0)
    PackRegistry._registry["zero_weight_pack"] = [pv1, pv2]

    # All 50 calls must return _PackV1 (weight=1.0), never _PackV2 (weight=0.0)
    for _ in range(50):
        result = PackRegistry.get("zero_weight_pack")
        assert result is _PackV1, (
            "Only _PackV1 should be returned when _PackV2 has weight=0"
        )


def test_get_with_seeded_random_is_deterministic(clean_registry) -> None:
    """With a fixed random seed and two equal-weight versions, get() is
    deterministic."""
    PackRegistry.register(_PackV1)
    PackRegistry.register(_PackV2)

    random.seed(42)
    first_run = [PackRegistry.get("test_versioned_pack") for _ in range(10)]
    random.seed(42)
    second_run = [PackRegistry.get("test_versioned_pack") for _ in range(10)]

    assert first_run == second_run


# ---------------------------------------------------------------------------
# _get_versions
# ---------------------------------------------------------------------------


def test_get_versions_raises_for_unknown_pack(clean_registry) -> None:
    """_get_versions raises KeyError for an unregistered pack_id."""
    with pytest.raises(KeyError, match="not registered"):
        PackRegistry._get_versions("no_such_pack")


def test_get_versions_returns_list_of_pack_versions(clean_registry) -> None:
    """_get_versions returns a list of PackVersion instances."""
    PackRegistry.register(_PackV1)
    versions = PackRegistry._get_versions("test_versioned_pack")
    assert isinstance(versions, list)
    assert all(isinstance(pv, PackVersion) for pv in versions)


# ---------------------------------------------------------------------------
# set_weights()
# ---------------------------------------------------------------------------


def test_set_weights_updates_traffic_split(clean_registry) -> None:
    """set_weights() to heavily favor one version; over 100 calls that version
    is returned the vast majority of the time."""
    PackRegistry.register(_PackV1)
    PackRegistry.register(_PackV2)

    # Give v2 a much higher weight so it dominates
    PackRegistry.set_weights("test_versioned_pack", {"1.0": 0.01, "2.0": 100.0})

    results = [PackRegistry.get("test_versioned_pack") for _ in range(100)]
    v2_count = results.count(_PackV2)
    # With weights 0.01 vs 100.0 the probability of v2 on each draw is ~99.99%.
    # Getting fewer than 90 v2 results in 100 draws is astronomically unlikely.
    assert v2_count >= 90, f"Expected _PackV2 to dominate, got only {v2_count}/100"


def test_set_weights_unknown_pack_raises_key_error(clean_registry) -> None:
    """set_weights() raises KeyError when the pack_id is not registered."""
    with pytest.raises(KeyError, match="not registered"):
        PackRegistry.set_weights("nonexistent_pack", {"1.0": 0.5})


def test_set_weights_unknown_version_raises_key_error(clean_registry) -> None:
    """set_weights() raises KeyError when a version string is not registered."""
    PackRegistry.register(_PackV1)
    with pytest.raises(KeyError):
        PackRegistry.set_weights("test_versioned_pack", {"99.0": 1.0})


def test_set_weights_negative_weight_raises_value_error(clean_registry) -> None:
    """set_weights() raises ValueError when a weight is negative."""
    PackRegistry.register(_PackV1)
    with pytest.raises(ValueError, match=">="):
        PackRegistry.set_weights("test_versioned_pack", {"1.0": -0.5})


def test_set_weights_partial_update_leaves_others_unchanged(clean_registry) -> None:
    """set_weights() for only one version leaves the other version's weight at 1.0."""
    PackRegistry.register(_PackV1)
    PackRegistry.register(_PackV2)

    PackRegistry.set_weights("test_versioned_pack", {"2.0": 5.0})

    versions = PackRegistry._registry["test_versioned_pack"]
    by_version = {pv.version: pv.weight for pv in versions}
    assert by_version["1.0"] == pytest.approx(1.0), "v1 weight should be unchanged"
    assert by_version["2.0"] == pytest.approx(5.0), "v2 weight should be updated"


# ---------------------------------------------------------------------------
# get() with explicit version
# ---------------------------------------------------------------------------


def test_get_with_version_returns_exact_version(clean_registry) -> None:
    """get(pack_id, version='2.0') always returns the v2 class regardless of weights."""
    PackRegistry.register(_PackV1)
    PackRegistry.register(_PackV2)

    # Verify over multiple calls that the exact version is returned
    for _ in range(20):
        result = PackRegistry.get("test_versioned_pack", version="2.0")
        assert result is _PackV2

    for _ in range(20):
        result = PackRegistry.get("test_versioned_pack", version="1.0")
        assert result is _PackV1


def test_get_with_unknown_version_raises_key_error(clean_registry) -> None:
    """get(pack_id, version='99.0') raises KeyError when version is not registered."""
    PackRegistry.register(_PackV1)
    with pytest.raises(KeyError):
        PackRegistry.get("test_versioned_pack", version="99.0")
