"""tests/test_pack_error_paths.py — Error-path coverage for the older packs
(summariser, research_only, analysis_only) that previously only had happy-path
tests. All agents/LLMs are mocked; no network access."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from agents.base_agent import (
    AgentExecutionError,
    AgentTimeoutError,
    AgentValidationError,
)
from domain_packs.productivity.summariser.pack import SummariserPack
from domain_packs.productivity.summariser.schemas import SummaryInput
from domain_packs.research.analysis_only.pack import AnalysisOnlyPack
from domain_packs.research.analysis_only.schemas import AnalysisOnlyInput
from domain_packs.research.research_only.pack import ResearchOnlyPack
from domain_packs.research.research_only.schemas import ResearchOnlyInput

# ---------------------------------------------------------------------------
# SummariserPack
# ---------------------------------------------------------------------------


def test_summariser_without_llm_raises_execution_error() -> None:
    pack = SummariserPack(run_id="sum-err-1", llm=None)
    with pytest.raises(AgentExecutionError, match="requires an LLM"):
        pack.run_from_input(SummaryInput(text="Some text"))


def test_summariser_llm_timeout_surfaces_as_execution_error() -> None:
    llm = MagicMock()
    llm.invoke.side_effect = AgentTimeoutError("LLM call timed out")
    pack = SummariserPack(run_id="sum-err-2", llm=llm)
    with pytest.raises(AgentExecutionError, match="timed out"):
        pack.run_from_input(SummaryInput(text="Some text"))


def test_summariser_unexpected_llm_exception_wrapped() -> None:
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("socket closed")
    pack = SummariserPack(run_id="sum-err-3", llm=llm)
    with pytest.raises(AgentExecutionError, match="Pipeline execution failed"):
        pack.run_from_input(SummaryInput(text="Some text"))


def test_summariser_run_empty_query_raises_validation_error() -> None:
    pack = SummariserPack(run_id="sum-err-4", llm=MagicMock())
    with pytest.raises(AgentValidationError, match="non-empty"):
        pack.run("   ")


def test_summariser_non_bullet_response_falls_back_to_lines() -> None:
    """If the LLM ignores the '- ' format, raw lines are used as bullets."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="First point\nSecond point")
    pack = SummariserPack(run_id="sum-err-5", llm=llm)
    result = pack.run_from_input(SummaryInput(text="Some text", bullet_count=3))
    assert result.bullets == ["First point", "Second point"]


def test_summary_input_extra_field_rejected() -> None:
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        SummaryInput(text="hello", surprise="nope")


def test_summary_input_max_length_rejected() -> None:
    with pytest.raises(ValidationError, match="at most"):
        SummaryInput(text="x" * 4001)


# ---------------------------------------------------------------------------
# ResearchOnlyPack
# ---------------------------------------------------------------------------


def test_research_only_agent_timeout_surfaces_as_execution_error() -> None:
    mock_agent = MagicMock()
    mock_agent.run_structured.side_effect = AgentTimeoutError("research timed out")
    with patch("core.graph.ResearchAgent", return_value=mock_agent):
        pack = ResearchOnlyPack(run_id="ro-err-1")
        with pytest.raises(AgentExecutionError, match="research timed out"):
            pack.run("What is AI?")


def test_research_only_agent_execution_error_surfaces() -> None:
    mock_agent = MagicMock()
    mock_agent.run_structured.side_effect = AgentExecutionError("provider down")
    with patch("core.graph.ResearchAgent", return_value=mock_agent):
        pack = ResearchOnlyPack(run_id="ro-err-2")
        with pytest.raises(AgentExecutionError, match="provider down"):
            pack.run("What is AI?")


def test_research_only_run_empty_query_raises_validation_error() -> None:
    pack = ResearchOnlyPack(run_id="ro-err-3")
    with pytest.raises(AgentValidationError, match="non-empty"):
        pack.run("")


def test_research_only_input_extra_field_rejected() -> None:
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        ResearchOnlyInput(query="q", bonus="nope")


def test_research_only_input_max_length_rejected() -> None:
    with pytest.raises(ValidationError, match="at most"):
        ResearchOnlyInput(query="x" * 2001)


# ---------------------------------------------------------------------------
# AnalysisOnlyPack
# ---------------------------------------------------------------------------


def test_analysis_only_agent_error_surfaces_as_execution_error() -> None:
    mock_agent = MagicMock()
    mock_agent.run_structured.side_effect = AgentExecutionError("analysis blew up")
    inp = AnalysisOnlyInput(query="What is AI?", summary="prior research")
    with patch("core.graph.AnalystAgent", return_value=mock_agent):
        pack = AnalysisOnlyPack(run_id="ao-err-1")
        with pytest.raises(AgentExecutionError, match="analysis blew up"):
            pack.run_from_input(inp)


def test_analysis_only_agent_timeout_surfaces_as_execution_error() -> None:
    mock_agent = MagicMock()
    mock_agent.run_structured.side_effect = AgentTimeoutError("analysis timed out")
    inp = AnalysisOnlyInput(query="What is AI?", summary="prior research")
    with patch("core.graph.AnalystAgent", return_value=mock_agent):
        pack = AnalysisOnlyPack(run_id="ao-err-2")
        with pytest.raises(AgentExecutionError, match="analysis timed out"):
            pack.run_from_input(inp)


def test_analysis_only_run_empty_query_raises_validation_error() -> None:
    pack = AnalysisOnlyPack(run_id="ao-err-3")
    with pytest.raises(AgentValidationError, match="non-empty"):
        pack.run("  ")


def test_analysis_only_input_extra_field_rejected() -> None:
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        AnalysisOnlyInput(query="q", summary="s", bonus="nope")


def test_analysis_only_input_max_length_rejected() -> None:
    with pytest.raises(ValidationError, match="at most"):
        AnalysisOnlyInput(query="x" * 2001, summary="s")
