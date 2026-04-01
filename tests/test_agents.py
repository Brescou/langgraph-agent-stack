"""
tests/test_agents.py — Unit tests for ResearchAgent and AnalystAgent.

All LLM calls are intercepted by patching ``core.llm.get_llm`` so no real
API requests are made.  The agents run their full LangGraph pipelines with a
mock LLM that returns deterministic JSON responses.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.analyst import AnalysisReport, AnalystAgent
from agents.researcher import ResearchAgent, ResearchResult

# ---------------------------------------------------------------------------
# Mock LLM factory helpers
# ---------------------------------------------------------------------------


def _make_ai_message(content: str) -> AIMessage:
    """Wrap a string in an AIMessage."""
    return AIMessage(content=content)


def _build_researcher_llm() -> MagicMock:
    """
    Return a MagicMock LLM whose ``invoke`` returns appropriate JSON for each
    ResearchAgent node:

    * query expansion  → JSON array of sub-queries
    * validation       → ``{"sufficient": true, "reason": "…"}``
    * summarize        → ``{"summary": "…", "confidence": 0.8}``
    """
    llm = MagicMock()
    llm.bind_tools.return_value = llm  # llm_with_tools == same mock

    call_count = [0]

    def smart_invoke(messages: list[Any]) -> AIMessage:
        call_count[0] += 1
        system_content = str(messages[0].content) if messages else ""

        if "query expander" in system_content.lower():
            return _make_ai_message(
                json.dumps(["sub-query 1", "sub-query 2", "sub-query 3"])
            )
        elif "quality assessor" in system_content.lower():
            return _make_ai_message(
                json.dumps({"sufficient": True, "reason": "Findings look good."})
            )
        elif "summariser" in system_content.lower():
            return _make_ai_message(
                json.dumps(
                    {"summary": "AI is a field of computer science.", "confidence": 0.8}
                )
            )
        # Default fallback
        return _make_ai_message(
            json.dumps({"summary": "Default summary.", "confidence": 0.5})
        )

    llm.invoke.side_effect = smart_invoke
    return llm


def _build_analyst_llm() -> MagicMock:
    """
    Return a MagicMock LLM whose ``invoke`` returns appropriate JSON for each
    AnalystAgent node:

    * analyze    → ``{"insights": […], "confidence": 0.85}``
    * synthesize → ``{"patterns": […], "implications": […]}``
    * report     → plain executive summary text
    """
    llm = MagicMock()
    llm.bind_tools.return_value = llm

    def smart_invoke(messages: list[Any]) -> AIMessage:
        system_content = str(messages[0].content) if messages else ""

        if "analytical thinker" in system_content.lower():
            return _make_ai_message(
                json.dumps(
                    {
                        "insights": [
                            "AI is transforming industries.",
                            "Deep learning drives progress.",
                        ],
                        "confidence": 0.85,
                    }
                )
            )
        elif "pattern synthesiser" in system_content.lower():
            return _make_ai_message(
                json.dumps(
                    {
                        "patterns": ["Rapid capability growth across modalities."],
                        "implications": ["Increased automation of knowledge work."],
                    }
                )
            )
        elif "report writer" in system_content.lower():
            return _make_ai_message(
                "AI represents a paradigm shift in how software is built and deployed."
            )
        return _make_ai_message("Default analyst response.")

    llm.invoke.side_effect = smart_invoke
    return llm


# ---------------------------------------------------------------------------
# ResearchAgent tests
# ---------------------------------------------------------------------------


class TestResearchAgent:
    def test_run_structured_returns_research_result(self) -> None:
        """run_structured() should return a populated ResearchResult."""
        mock_llm = _build_researcher_llm()
        with patch("agents.base_agent.get_llm", return_value=mock_llm):
            agent = ResearchAgent()
            result = agent.run_structured("What is artificial intelligence?")

        assert isinstance(result, ResearchResult)
        assert result.query == "What is artificial intelligence?"
        assert result.summary != ""
        assert 0.0 <= result.confidence <= 1.0

    def test_run_returns_string(self) -> None:
        """run() should return the research summary as a plain string."""
        mock_llm = _build_researcher_llm()
        with patch("agents.base_agent.get_llm", return_value=mock_llm):
            agent = ResearchAgent()
            output = agent.run("Explain quantum computing")

        assert isinstance(output, str)
        assert len(output) > 0

    def test_run_structured_raises_on_empty_input(self) -> None:
        """run_structured() with empty input raises AgentValidationError."""
        from agents.base_agent import AgentValidationError

        mock_llm = _build_researcher_llm()
        with patch("agents.base_agent.get_llm", return_value=mock_llm):
            agent = ResearchAgent()

        with pytest.raises(AgentValidationError):
            agent.run_structured("")

    def test_research_result_has_findings(self) -> None:
        """Findings list should be non-empty after a successful run."""
        mock_llm = _build_researcher_llm()
        with patch("agents.base_agent.get_llm", return_value=mock_llm):
            agent = ResearchAgent()
            result = agent.run_structured("Machine learning basics")

        assert isinstance(result.findings, list)
        # The mock search tool always returns at least one result per sub-query
        assert len(result.findings) > 0

    def test_research_result_serialises_to_dict(self) -> None:
        """to_dict() should return all expected keys."""
        mock_llm = _build_researcher_llm()
        with patch("agents.base_agent.get_llm", return_value=mock_llm):
            agent = ResearchAgent()
            result = agent.run_structured("What is a neural network?")

        d = result.to_dict()
        assert set(d.keys()) == {
            "query",
            "findings",
            "summary",
            "sources",
            "confidence",
            "metadata",
        }

    def test_thread_id_is_set(self) -> None:
        """A ResearchAgent should have a non-empty thread_id."""
        mock_llm = _build_researcher_llm()
        with patch("agents.base_agent.get_llm", return_value=mock_llm):
            agent = ResearchAgent(thread_id="test-thread-001")

        assert agent.thread_id == "test-thread-001"


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def test_invoke_llm_with_retry_succeeds_after_transient_errors(self) -> None:
        """Retry logic should handle transient errors and succeed."""
        mock_llm = MagicMock()
        call_count = [0]

        def flaky_invoke(messages):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ConnectionError("Transient failure")
            return _make_ai_message("Success")

        mock_llm.invoke.side_effect = flaky_invoke
        mock_llm.bind_tools.return_value = mock_llm

        with patch("agents.base_agent.get_llm", return_value=mock_llm):
            agent = ResearchAgent()
            with patch("time.sleep"):
                result = agent._invoke_llm_with_retry(
                    [HumanMessage(content="test")], max_retries=3
                )

        assert result.content == "Success"
        assert call_count[0] == 3

    def test_invoke_llm_with_retry_raises_after_max_attempts(self) -> None:
        """After exhausting retries, should raise AgentExecutionError."""
        from agents.base_agent import AgentExecutionError

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("Always fails")
        mock_llm.bind_tools.return_value = mock_llm

        with patch("agents.base_agent.get_llm", return_value=mock_llm):
            agent = ResearchAgent()
            with patch("time.sleep"):
                with pytest.raises(AgentExecutionError, match="failed after"):
                    agent._invoke_llm_with_retry(
                        [HumanMessage(content="test")], max_retries=2
                    )


# ---------------------------------------------------------------------------
# Research validation loop tests
# ---------------------------------------------------------------------------


class TestResearchValidationLoop:
    def test_validation_loop_retries_when_insufficient(self) -> None:
        """When validation says findings insufficient, research should re-run."""
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        validation_calls = [0]

        def smart_invoke(messages):
            system = str(messages[0].content).lower() if messages else ""
            if "query expander" in system:
                return _make_ai_message(json.dumps(["q1", "q2"]))
            elif "quality assessor" in system:
                validation_calls[0] += 1
                if validation_calls[0] <= 1:
                    return _make_ai_message(
                        json.dumps({"sufficient": False, "reason": "Need more data"})
                    )
                return _make_ai_message(
                    json.dumps({"sufficient": True, "reason": "Good enough"})
                )
            elif "summariser" in system:
                return _make_ai_message(
                    json.dumps({"summary": "Final summary", "confidence": 0.9})
                )
            return _make_ai_message(
                json.dumps({"summary": "Default", "confidence": 0.5})
            )

        llm.invoke.side_effect = smart_invoke

        with patch("agents.base_agent.get_llm", return_value=llm):
            agent = ResearchAgent()
            result = agent.run_structured("Test query for validation loop")

        assert isinstance(result, ResearchResult)
        assert validation_calls[0] == 2


# ---------------------------------------------------------------------------
# AnalystAgent tests
# ---------------------------------------------------------------------------


class TestAnalystAgent:
    @pytest.fixture
    def research_result(self) -> ResearchResult:
        return ResearchResult(
            query="What is AI?",
            findings=["AI finding 1", "AI finding 2"],
            summary="AI is a broad field of computer science.",
            sources=["https://example.com/ai"],
            confidence=0.8,
            metadata={"agent": "ResearchAgent"},
        )

    def test_run_structured_returns_analysis_report(
        self, research_result: ResearchResult
    ) -> None:
        """run_structured() should return a populated AnalysisReport."""
        mock_llm = _build_analyst_llm()
        with patch("agents.base_agent.get_llm", return_value=mock_llm):
            agent = AnalystAgent()
            report = agent.run_structured(research_result)

        assert isinstance(report, AnalysisReport)
        assert report.query == "What is AI?"
        assert report.executive_summary != ""
        assert 0.0 <= report.confidence <= 1.0

    def test_run_structured_has_insights(self, research_result: ResearchResult) -> None:
        """key_insights should be a non-empty list."""
        mock_llm = _build_analyst_llm()
        with patch("agents.base_agent.get_llm", return_value=mock_llm):
            agent = AnalystAgent()
            report = agent.run_structured(research_result)

        assert isinstance(report.key_insights, list)
        assert len(report.key_insights) > 0

    def test_run_structured_has_patterns_and_implications(
        self, research_result: ResearchResult
    ) -> None:
        """patterns and implications should be populated lists."""
        mock_llm = _build_analyst_llm()
        with patch("agents.base_agent.get_llm", return_value=mock_llm):
            agent = AnalystAgent()
            report = agent.run_structured(research_result)

        assert isinstance(report.patterns, list)
        assert isinstance(report.implications, list)

    def test_analysis_report_to_dict(self, research_result: ResearchResult) -> None:
        """to_dict() should return all expected keys."""
        mock_llm = _build_analyst_llm()
        with patch("agents.base_agent.get_llm", return_value=mock_llm):
            agent = AnalystAgent()
            report = agent.run_structured(research_result)

        d = report.to_dict()
        expected_keys = {
            "query",
            "executive_summary",
            "key_insights",
            "patterns",
            "implications",
            "confidence",
            "research_summary",
            "metadata",
        }
        assert set(d.keys()) == expected_keys

    def test_analysis_report_to_markdown(self, research_result: ResearchResult) -> None:
        """to_markdown() should produce a non-empty Markdown string."""
        mock_llm = _build_analyst_llm()
        with patch("agents.base_agent.get_llm", return_value=mock_llm):
            agent = AnalystAgent()
            report = agent.run_structured(research_result)

        md = report.to_markdown()
        assert "# Analysis Report" in md
        assert "## Executive Summary" in md
        assert "## Key Insights" in md

    def test_run_raises_on_empty_input(self) -> None:
        """run() with empty input raises AgentValidationError."""
        from agents.base_agent import AgentValidationError

        mock_llm = _build_analyst_llm()
        with patch("agents.base_agent.get_llm", return_value=mock_llm):
            agent = AnalystAgent()

        with pytest.raises(AgentValidationError):
            agent.run("")

    def test_analyst_run_string_input(self) -> None:
        """AnalystAgent.run() with a plain string should work."""
        mock_llm = _build_analyst_llm()
        with patch("agents.base_agent.get_llm", return_value=mock_llm):
            agent = AnalystAgent()
            output = agent.run(
                '{"query": "What is AI?", "summary": "AI overview",'
                ' "findings": ["f1"], "sources": ["s1"],'
                ' "confidence": 0.8, "metadata": {}}'
            )

        assert isinstance(output, str)
        assert len(output) > 0
