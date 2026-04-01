"""
tests/test_integration.py — Integration tests verifying constructor signature
compatibility between agents and the API layer.

These tests instantiate the real agent classes (not MagicMock replacements)
with a mocked LLM, ensuring that keyword arguments accepted by
``api/main.py`` are actually forwarded through the class hierarchy.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver

from agents.analyst import AnalystAgent
from agents.researcher import ResearchAgent
from core.graph import MultiAgentGraph


@pytest.fixture()
def mock_llm() -> MagicMock:
    """Return a MagicMock that satisfies the ``BaseChatModel`` interface."""
    llm = MagicMock(spec=BaseChatModel)
    llm.bind_tools.return_value = llm
    llm.invoke.return_value = MagicMock(content='{"insights": [], "confidence": 0.5}')
    return llm


@pytest.fixture()
def mock_checkpointer() -> MemorySaver:
    """Return a real MemorySaver — LangGraph rejects non-BaseCheckpointSaver objects."""
    return MemorySaver()


class TestResearchAgentSignature:
    """Verify that ResearchAgent accepts injected llm and checkpointer."""

    def test_accepts_llm_and_checkpointer(
        self, mock_llm: MagicMock, mock_checkpointer: MemorySaver
    ) -> None:
        agent = ResearchAgent(
            thread_id="test-thread",
            llm=mock_llm,
            checkpointer=mock_checkpointer,
        )
        assert agent.llm is mock_llm
        assert agent.checkpointer is mock_checkpointer


class TestAnalystAgentSignature:
    """Verify that AnalystAgent accepts injected llm and checkpointer."""

    def test_accepts_llm_and_checkpointer(
        self, mock_llm: MagicMock, mock_checkpointer: MemorySaver
    ) -> None:
        agent = AnalystAgent(
            thread_id="test-thread",
            llm=mock_llm,
            checkpointer=mock_checkpointer,
        )
        assert agent.llm is mock_llm
        assert agent.checkpointer is mock_checkpointer


class TestMultiAgentGraphSignature:
    """Verify that MultiAgentGraph stores and propagates llm/checkpointer."""

    def test_stores_llm_and_checkpointer(
        self, mock_llm: MagicMock, mock_checkpointer: MemorySaver
    ) -> None:
        graph = MultiAgentGraph(
            run_id="test-run",
            llm=mock_llm,
            checkpointer=mock_checkpointer,
        )
        assert graph._llm is mock_llm
        assert graph._checkpointer is mock_checkpointer

    def test_propagates_to_research_agent(
        self, mock_llm: MagicMock, mock_checkpointer: MemorySaver
    ) -> None:
        """ResearchAgent inside the graph must receive the injected llm."""
        mock_research = MagicMock()
        mock_analyst = MagicMock()

        with (
            patch("core.graph.ResearchAgent", return_value=mock_research) as ra_cls,
            patch("core.graph.AnalystAgent", return_value=mock_analyst),
        ):
            graph = MultiAgentGraph(
                run_id="prop-test",
                llm=mock_llm,
                checkpointer=mock_checkpointer,
            )

            mock_research.run_structured.return_value = MagicMock(
                query="q",
                findings=[],
                summary="s",
                sources=[],
                confidence=0.5,
                metadata={},
                to_dict=lambda: {
                    "query": "q",
                    "findings": [],
                    "summary": "s",
                    "sources": [],
                    "confidence": 0.5,
                    "metadata": {},
                },
            )
            mock_analyst.run_structured.return_value = MagicMock(
                query="q",
                executive_summary="es",
                key_insights=[],
                patterns=[],
                implications=[],
                confidence=0.5,
                research_summary="s",
                metadata={},
                to_dict=lambda: {
                    "query": "q",
                    "executive_summary": "es",
                    "key_insights": [],
                    "patterns": [],
                    "implications": [],
                    "confidence": 0.5,
                    "research_summary": "s",
                    "metadata": {},
                },
            )

            graph.run("test query")

        ra_cls.assert_called_once()
        call_kwargs = ra_cls.call_args[1]
        assert call_kwargs["llm"] is mock_llm
        assert call_kwargs["checkpointer"] is mock_checkpointer
