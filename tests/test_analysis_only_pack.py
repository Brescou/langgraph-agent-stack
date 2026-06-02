"""tests/test_analysis_only_pack.py — Unit tests for AnalysisOnlyPack."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents.models import AnalysisReport
from domain_packs.research.analysis_only.pack import AnalysisOnlyPack
from domain_packs.research.analysis_only.schemas import (
    AnalysisOnlyInput,
    AnalysisOnlyOutput,
)


def test_analysis_only_run_from_input_returns_analysis_report() -> None:
    mock_report = AnalysisReport(
        query="What is AI?",
        executive_summary="AI summary",
        key_insights=["insight"],
        patterns=["pattern"],
        implications=["implication"],
        confidence=0.8,
        research_summary="Prior research",
    )
    mock_agent = MagicMock()
    mock_agent.run_structured.return_value = mock_report

    inp = AnalysisOnlyInput(
        query="What is AI?",
        summary="Prior research",
        findings=["finding one"],
    )

    with patch("core.graph.AnalystAgent", return_value=mock_agent):
        pack = AnalysisOnlyPack(run_id="ao-test-1")
        result = pack.run_from_input(inp)

    assert isinstance(result, AnalysisReport)
    assert result.executive_summary == "AI summary"
    mock_agent.run_structured.assert_called_once()


def test_analysis_only_output_from_analysis_report(
    mock_analysis_report: AnalysisReport,
) -> None:
    output = AnalysisOnlyOutput.from_analysis_report(
        mock_analysis_report, cost_usd=0.01
    )
    assert output.executive_summary == mock_analysis_report.executive_summary
    assert output.cost_usd == 0.01
