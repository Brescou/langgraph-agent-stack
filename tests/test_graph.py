"""
tests/test_graph.py — Unit tests for MultiAgentGraph orchestrator.

ResearchAgent and AnalystAgent are fully mocked so no LLM calls are made.
Tests verify the orchestrator's routing, state propagation, and error handling.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from agents.analyst import AnalysisReport
from agents.researcher import ResearchResult
from core.graph import MultiAgentGraph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_research_result() -> ResearchResult:
    return ResearchResult(
        query="What is AI?",
        findings=["Finding 1", "Finding 2"],
        summary="AI is a field of computer science.",
        sources=["https://example.com"],
        confidence=0.85,
        metadata={"agent": "ResearchAgent"},
    )


@pytest.fixture
def mock_analysis_report() -> AnalysisReport:
    return AnalysisReport(
        query="What is AI?",
        executive_summary="AI transforms industries by automating complex tasks.",
        key_insights=["Insight 1", "Insight 2"],
        patterns=["Pattern 1"],
        implications=["Implication 1"],
        confidence=0.82,
        research_summary="AI is a field of computer science.",
        metadata={"agent": "AnalystAgent"},
    )


# ---------------------------------------------------------------------------
# Pipeline success path
# ---------------------------------------------------------------------------


class TestMultiAgentGraphRun:
    def test_run_returns_analysis_report(
        self,
        mock_research_result: ResearchResult,
        mock_analysis_report: AnalysisReport,
    ) -> None:
        """Full pipeline should return an AnalysisReport on success."""
        mock_research_agent = MagicMock()
        mock_research_agent.run_structured.return_value = mock_research_result

        mock_analyst_agent = MagicMock()
        mock_analyst_agent.run_structured.return_value = mock_analysis_report

        with (
            patch("core.graph.ResearchAgent", return_value=mock_research_agent),
            patch("core.graph.AnalystAgent", return_value=mock_analyst_agent),
        ):
            graph = MultiAgentGraph(run_id="test-run-001")
            report = graph.run("What is AI?")

        assert isinstance(report, AnalysisReport)
        assert report.query == "What is AI?"
        assert report.confidence == 0.82

    def test_run_passes_research_result_to_analyst(
        self,
        mock_research_result: ResearchResult,
        mock_analysis_report: AnalysisReport,
    ) -> None:
        """AnalystAgent.run_structured should receive the ResearchResult."""
        mock_research_agent = MagicMock()
        mock_research_agent.run_structured.return_value = mock_research_result

        mock_analyst_agent = MagicMock()
        mock_analyst_agent.run_structured.return_value = mock_analysis_report

        with (
            patch("core.graph.ResearchAgent", return_value=mock_research_agent),
            patch("core.graph.AnalystAgent", return_value=mock_analyst_agent),
        ):
            graph = MultiAgentGraph()
            graph.run("What is AI?")

        # Analyst must have been called with the research result
        call_args = mock_analyst_agent.run_structured.call_args
        assert call_args is not None
        passed_result = call_args[0][0]
        assert isinstance(passed_result, ResearchResult)
        assert passed_result.query == "What is AI?"

    def test_run_raises_on_empty_query(self) -> None:
        """Empty query should raise AgentValidationError immediately."""
        from agents.base_agent import AgentValidationError

        graph = MultiAgentGraph()
        with pytest.raises(AgentValidationError):
            graph.run("   ")

    def test_run_raises_on_empty_string(self) -> None:
        """Empty string query should raise AgentValidationError."""
        from agents.base_agent import AgentValidationError

        graph = MultiAgentGraph()
        with pytest.raises(AgentValidationError):
            graph.run("")

    def test_run_id_defaults_to_uuid(self) -> None:
        """When no run_id is supplied, a UUID string is generated."""
        graph = MultiAgentGraph()
        assert graph.run_id
        assert len(graph.run_id) == 36  # UUID4 format: 8-4-4-4-12

    def test_run_id_is_preserved(self) -> None:
        """A supplied run_id should be stored on the instance."""
        graph = MultiAgentGraph(run_id="my-custom-run-id")
        assert graph.run_id == "my-custom-run-id"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestMultiAgentGraphErrors:
    def test_research_failure_raises_execution_error(self) -> None:
        """When ResearchAgent fails the pipeline should raise AgentExecutionError."""
        from agents.base_agent import AgentExecutionError

        mock_research_agent = MagicMock()
        mock_research_agent.run_structured.side_effect = AgentExecutionError(
            "Research failed"
        )

        with patch("core.graph.ResearchAgent", return_value=mock_research_agent):
            graph = MultiAgentGraph()
            with pytest.raises(AgentExecutionError):
                graph.run("What is AI?")

    def test_analysis_not_called_when_research_fails(self) -> None:
        """When research fails, AnalystAgent should never be invoked."""
        from agents.base_agent import AgentExecutionError

        mock_research_agent = MagicMock()
        mock_research_agent.run_structured.side_effect = AgentExecutionError("fail")

        mock_analyst_agent = MagicMock()

        with (
            patch("core.graph.ResearchAgent", return_value=mock_research_agent),
            patch("core.graph.AnalystAgent", return_value=mock_analyst_agent),
        ):
            graph = MultiAgentGraph()
            with pytest.raises(AgentExecutionError):
                graph.run("What is AI?")

        mock_analyst_agent.run_structured.assert_not_called()

    def test_analysis_failure_raises_execution_error(
        self, mock_research_result: ResearchResult
    ) -> None:
        """When AnalystAgent fails the pipeline should raise AgentExecutionError."""
        from agents.base_agent import AgentExecutionError

        mock_research_agent = MagicMock()
        mock_research_agent.run_structured.return_value = mock_research_result

        mock_analyst_agent = MagicMock()
        mock_analyst_agent.run_structured.side_effect = AgentExecutionError(
            "Analysis failed"
        )

        with (
            patch("core.graph.ResearchAgent", return_value=mock_research_agent),
            patch("core.graph.AnalystAgent", return_value=mock_analyst_agent),
        ):
            graph = MultiAgentGraph()
            with pytest.raises(AgentExecutionError):
                graph.run("What is AI?")


# ---------------------------------------------------------------------------
# get_research_result
# ---------------------------------------------------------------------------


class TestMultiAgentGraphResearchOnly:
    def test_get_research_result_returns_research_result(
        self, mock_research_result: ResearchResult
    ) -> None:
        """get_research_result() should return a ResearchResult without analysis."""
        mock_research_agent = MagicMock()
        mock_research_agent.run_structured.return_value = mock_research_result

        with patch("core.graph.ResearchAgent", return_value=mock_research_agent):
            graph = MultiAgentGraph()
            result = graph.get_research_result("What is AI?")

        assert isinstance(result, ResearchResult)
        assert result.query == "What is AI?"

    def test_get_research_result_raises_on_empty_query(self) -> None:
        """Empty query should raise AgentValidationError."""
        from agents.base_agent import AgentValidationError

        graph = MultiAgentGraph()
        with pytest.raises(AgentValidationError):
            graph.get_research_result("")


# ---------------------------------------------------------------------------
# Async pipeline
# ---------------------------------------------------------------------------


class TestMultiAgentGraphAsync:
    def test_arun_returns_analysis_report(
        self,
        mock_research_result: ResearchResult,
        mock_analysis_report: AnalysisReport,
    ) -> None:
        """arun() should return an AnalysisReport via async execution."""
        mock_research_agent = MagicMock()
        mock_research_agent.run_structured.return_value = mock_research_result

        mock_analyst_agent = MagicMock()
        mock_analyst_agent.run_structured.return_value = mock_analysis_report

        with (
            patch("core.graph.ResearchAgent", return_value=mock_research_agent),
            patch("core.graph.AnalystAgent", return_value=mock_analyst_agent),
        ):
            graph = MultiAgentGraph(run_id="async-test")
            report = asyncio.run(graph.arun("What is AI?"))

        assert isinstance(report, AnalysisReport)
        assert report.query == "What is AI?"
