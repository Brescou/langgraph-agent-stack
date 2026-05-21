"""domain_packs/analysis_only/schemas.py — Typed I/O for AnalysisOnlyPack."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalysisOnlyInput(BaseModel):
    """Input: pre-collected research material for AnalystAgent."""

    query: str = Field(..., min_length=1, max_length=2000)
    summary: str = Field(default="", description="Research summary to analyse")
    findings: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class AnalysisOnlyOutput(BaseModel):
    """Output schema aligned with AnalysisReport fields."""

    query: str
    executive_summary: str
    key_insights: list[str]
    patterns: list[str]
    implications: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    research_summary: str
    cost_usd: float | None = None

    @classmethod
    def from_analysis_report(
        cls, report: Any, cost_usd: float | None = None
    ) -> AnalysisOnlyOutput:
        from agents.models import AnalysisReport

        if not isinstance(report, AnalysisReport):
            raise TypeError(f"Expected AnalysisReport, got {type(report).__name__!r}")
        return cls(
            query=report.query,
            executive_summary=report.executive_summary,
            key_insights=report.key_insights,
            patterns=report.patterns,
            implications=report.implications,
            confidence=report.confidence,
            research_summary=report.research_summary,
            cost_usd=cost_usd,
        )
