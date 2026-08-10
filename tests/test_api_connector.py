"""tests/test_api_connector.py — API wiring for CONNECTOR_ENABLED."""

from __future__ import annotations

import api.state as api_state
from api.dependencies import pack_runtime_kwargs
from connectors.examples.example_connector import ExampleMemoryConnector
from domain_packs.research.research_analysis.pack import ResearchAnalysisPack
from domain_packs.research.research_only.pack import ResearchOnlyPack


def test_pack_runtime_kwargs_injects_connector_for_research_analysis() -> None:
    api_state.shared_connector = ExampleMemoryConnector()
    try:
        kwargs = pack_runtime_kwargs(ResearchAnalysisPack)
        assert kwargs == {"connector": api_state.shared_connector}
    finally:
        api_state.shared_connector = None


def test_pack_runtime_kwargs_injects_connector_for_rfp_assistant() -> None:
    from domain_packs.productivity.rfp_assistant.pack import RfpAssistantPack

    api_state.shared_connector = ExampleMemoryConnector()
    try:
        kwargs = pack_runtime_kwargs(RfpAssistantPack)
        assert kwargs == {"connector": api_state.shared_connector}
    finally:
        api_state.shared_connector = None


def test_pack_runtime_kwargs_skips_research_only() -> None:
    api_state.shared_connector = ExampleMemoryConnector()
    try:
        assert pack_runtime_kwargs(ResearchOnlyPack) == {}
    finally:
        api_state.shared_connector = None


def test_pack_runtime_kwargs_empty_when_connector_disabled() -> None:
    api_state.shared_connector = None
    assert pack_runtime_kwargs(ResearchAnalysisPack) == {}


def test_pack_runtime_kwargs_forwards_pack_identity() -> None:
    kwargs = pack_runtime_kwargs(
        ResearchAnalysisPack,
        pack_id="research_analysis",
        pack_version="2.0",
    )
    assert kwargs == {
        "pack_id": "research_analysis",
        "pack_version": "2.0",
    }
