"""agents/models.py — Shared data models for agent outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResearchResult:
    """
    Structured output produced by the ResearchAgent.

    Attributes:
        query: Original research query.
        findings: List of raw text snippets collected during retrieval.
        summary: LLM-generated summary of the consolidated findings.
        sources: List of source identifiers (URLs, doc IDs, …).
        confidence: Self-reported confidence score between 0.0 and 1.0.
        metadata: Arbitrary run-level metadata forwarded from ``AgentState``.
    """

    query: str
    findings: list[str] = field(default_factory=list)
    summary: str = ""
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a plain dictionary."""
        return {
            "query": self.query,
            "findings": self.findings,
            "summary": self.summary,
            "sources": self.sources,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class AnalysisReport:
    """
    Structured output produced by the AnalystAgent.

    Attributes:
        query: The original research question this report addresses.
        executive_summary: One-paragraph high-level conclusion.
        key_insights: Bulleted list of the most important findings.
        patterns: Identified recurring themes or structural patterns.
        implications: Practical consequences and recommendations.
        confidence: Self-reported confidence score between 0.0 and 1.0.
        research_summary: The input ``ResearchResult.summary`` for traceability.
        metadata: Forwarded run-level metadata.
    """

    query: str
    executive_summary: str = ""
    key_insights: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    implications: list[str] = field(default_factory=list)
    confidence: float = 0.0
    research_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a plain dictionary."""
        return {
            "query": self.query,
            "executive_summary": self.executive_summary,
            "key_insights": self.key_insights,
            "patterns": self.patterns,
            "implications": self.implications,
            "confidence": self.confidence,
            "research_summary": self.research_summary,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        """Render the report as a Markdown string."""
        lines: list[str] = [
            f"# Analysis Report: {self.query}",
            "",
            "## Executive Summary",
            self.executive_summary,
            "",
            "## Key Insights",
        ]
        for insight in self.key_insights:
            lines.append(f"- {insight}")
        lines += ["", "## Identified Patterns"]
        for pattern in self.patterns:
            lines.append(f"- {pattern}")
        lines += ["", "## Implications & Recommendations"]
        for impl in self.implications:
            lines.append(f"- {impl}")
        lines += [
            "",
            f"*Confidence: {self.confidence:.0%}*",
        ]
        return "\n".join(lines)
