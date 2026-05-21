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
from agents.base_agent import AgentExecutionError, AgentValidationError
from agents.researcher import ResearchResult
from core.graph import MultiAgentGraph
from domain_packs.research_analysis.pack import ResearchAnalysisPack

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
        graph = MultiAgentGraph()
        with pytest.raises(AgentValidationError):
            graph.run("   ")

    def test_run_raises_on_empty_string(self) -> None:
        """Empty string query should raise AgentValidationError."""
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


# ---------------------------------------------------------------------------
# stream_events
# ---------------------------------------------------------------------------


async def _mock_events(events):
    """Helper: turn a list of dicts into an async generator."""
    for event in events:
        yield event


def _analysis_report_dict(**overrides):
    """Build a valid analysis_report dict with sensible defaults."""
    base = {
        "query": "test query",
        "executive_summary": "summary",
        "key_insights": ["insight"],
        "patterns": ["pattern"],
        "implications": ["impl"],
        "confidence": 0.85,
        "research_summary": "research",
    }
    base.update(overrides)
    return base


class TestMultiAgentGraphStreamEvents:
    """Tests for ``MultiAgentGraph.stream_events()``."""

    @pytest.fixture()
    def graph(self) -> MultiAgentGraph:
        """Return a ``MultiAgentGraph`` with a real compiled graph."""
        return MultiAgentGraph(run_id="test-stream")

    @pytest.mark.asyncio
    async def test_stream_events_raises_on_empty_query(self, graph):
        """Empty / whitespace query must raise AgentValidationError."""
        with pytest.raises(AgentValidationError):
            async for _ in graph.stream_events(""):
                pass

        with pytest.raises(AgentValidationError):
            async for _ in graph.stream_events("   "):
                pass

    @pytest.mark.asyncio
    async def test_stream_events_yields_phase_events(self, graph):
        """Phase start/end events for both nodes must be emitted."""
        report_dict = _analysis_report_dict()
        events = [
            {"event": "on_chain_start", "name": "research_node", "data": {}},
            {
                "event": "on_chain_end",
                "name": "research_node",
                "data": {"output": {"research_result": {}}},
            },
            {"event": "on_chain_start", "name": "analysis_node", "data": {}},
            {
                "event": "on_chain_end",
                "name": "analysis_node",
                "data": {"output": {"analysis_report": report_dict}},
            },
        ]
        graph._graph.astream_events = lambda *a, **kw: _mock_events(events)

        collected = []
        async for evt in graph.stream_events("test query"):
            collected.append(evt)

        event_types = [e["type"] for e in collected]
        assert event_types.count("phase_started") == 2
        assert event_types.count("phase_completed") == 2

        phases = [e["phase"] for e in collected if e["type"] == "phase_started"]
        assert phases == ["research", "analysis"]

    @pytest.mark.asyncio
    async def test_stream_events_yields_token_events(self, graph):
        """on_chat_model_stream events must surface as token events."""
        mock_chunk = MagicMock()
        mock_chunk.content = "hello"

        report_dict = _analysis_report_dict()
        events = [
            {"event": "on_chain_start", "name": "research_node", "data": {}},
            {
                "event": "on_chat_model_stream",
                "name": "llm",
                "data": {"chunk": mock_chunk},
                "metadata": {"langgraph_node": "research_node"},
            },
            {
                "event": "on_chain_end",
                "name": "research_node",
                "data": {"output": {}},
            },
            {"event": "on_chain_start", "name": "analysis_node", "data": {}},
            {
                "event": "on_chain_end",
                "name": "analysis_node",
                "data": {"output": {"analysis_report": report_dict}},
            },
        ]
        graph._graph.astream_events = lambda *a, **kw: _mock_events(events)

        collected = []
        async for evt in graph.stream_events("test query"):
            collected.append(evt)

        token_events = [e for e in collected if e["type"] == "token"]
        assert len(token_events) == 1
        assert token_events[0]["content"] == "hello"
        assert token_events[0]["node"] == "research_node"

    @pytest.mark.asyncio
    async def test_stream_events_yields_pipeline_completed(self, graph):
        """Final event must be pipeline_completed with the AnalysisReport."""
        report_dict = _analysis_report_dict(confidence=0.92)
        events = [
            {"event": "on_chain_start", "name": "research_node", "data": {}},
            {
                "event": "on_chain_end",
                "name": "research_node",
                "data": {"output": {}},
            },
            {"event": "on_chain_start", "name": "analysis_node", "data": {}},
            {
                "event": "on_chain_end",
                "name": "analysis_node",
                "data": {"output": {"analysis_report": report_dict}},
            },
        ]
        graph._graph.astream_events = lambda *a, **kw: _mock_events(events)

        collected = []
        async for evt in graph.stream_events("test query"):
            collected.append(evt)

        last = collected[-1]
        assert last["type"] == "pipeline_completed"
        assert isinstance(last["report"], dict)
        assert last["report"]["confidence"] == 0.92

    @pytest.mark.asyncio
    async def test_stream_events_raises_when_no_report(self, graph):
        """Missing analysis_report in output must raise AgentExecutionError."""
        events = [
            {"event": "on_chain_start", "name": "research_node", "data": {}},
            {
                "event": "on_chain_end",
                "name": "research_node",
                "data": {"output": {}},
            },
            {"event": "on_chain_start", "name": "analysis_node", "data": {}},
            {
                "event": "on_chain_end",
                "name": "analysis_node",
                "data": {"output": {}},
            },
        ]
        graph._graph.astream_events = lambda *a, **kw: _mock_events(events)

        with pytest.raises(AgentExecutionError, match="without an AnalysisReport"):
            async for _ in graph.stream_events("test query"):
                pass


# ---------------------------------------------------------------------------
# ResearchAnalysisPack — budget propagation
# ---------------------------------------------------------------------------


class TestResearchAnalysisPackBudget:
    def test_pack_propagates_budget_to_agents(self) -> None:
        """budget_usd passed to ResearchAnalysisPack must be stored on the instance."""
        pack = ResearchAnalysisPack(budget_usd=1.0)
        assert pack._budget_usd == 1.0
        pack.close()

    def test_pack_no_budget_by_default(self) -> None:
        """When budget_usd is omitted the pack must store None."""
        pack = ResearchAnalysisPack()
        assert pack._budget_usd is None
        pack.close()

    def test_pack_cost_usd_property_sums_agents(self) -> None:
        """cost_usd must return the sum of both agents' cost_usd values."""
        pack = ResearchAnalysisPack(budget_usd=5.0)

        mock_research = MagicMock()
        mock_research.cost_usd = 0.30
        mock_analyst = MagicMock()
        mock_analyst.cost_usd = 0.15

        pack._research_agent = mock_research
        pack._analyst_agent = mock_analyst

        assert pack.cost_usd == pytest.approx(0.45)
        pack.close()

    def test_pack_cost_usd_zero_when_no_agents_run(self) -> None:
        """cost_usd must be 0.0 before any agents are instantiated."""
        pack = ResearchAnalysisPack()
        assert pack.cost_usd == 0.0
        pack.close()

    def test_negative_budget_raises(self) -> None:
        """A negative budget_usd must raise ValueError at construction time."""
        with pytest.raises(ValueError, match="non-negative"):
            ResearchAnalysisPack(budget_usd=-1.0)
