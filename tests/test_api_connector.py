"""tests/test_api_connector.py — API wiring for CONNECTOR_ENABLED."""

from __future__ import annotations

import api.main as api_main
from connectors.examples.example_connector import ExampleMemoryConnector
from domain_packs.research_analysis.pack import ResearchAnalysisPack
from domain_packs.research_only.pack import ResearchOnlyPack


def test_pack_runtime_kwargs_injects_connector_for_research_analysis() -> None:
    api_main._shared_connector = ExampleMemoryConnector()
    try:
        kwargs = api_main._pack_runtime_kwargs(ResearchAnalysisPack)
        assert kwargs == {"connector": api_main._shared_connector}
    finally:
        api_main._shared_connector = None


def test_pack_runtime_kwargs_injects_connector_for_rfp_assistant() -> None:
    from domain_packs.rfp_assistant.pack import RfpAssistantPack

    api_main._shared_connector = ExampleMemoryConnector()
    try:
        kwargs = api_main._pack_runtime_kwargs(RfpAssistantPack)
        assert kwargs == {"connector": api_main._shared_connector}
    finally:
        api_main._shared_connector = None


def test_pack_runtime_kwargs_skips_research_only() -> None:
    api_main._shared_connector = ExampleMemoryConnector()
    try:
        assert api_main._pack_runtime_kwargs(ResearchOnlyPack) == {}
    finally:
        api_main._shared_connector = None


def test_pack_runtime_kwargs_empty_when_connector_disabled() -> None:
    api_main._shared_connector = None
    assert api_main._pack_runtime_kwargs(ResearchAnalysisPack) == {}
