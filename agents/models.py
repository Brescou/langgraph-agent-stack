"""agents/models.py — Shared data models for agent outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


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

    # Expected type per field — used by ``from_dict`` for shape validation.
    _FIELD_TYPES: ClassVar[dict[str, type | tuple[type, ...]]] = {
        "query": str,
        "findings": list,
        "summary": str,
        "sources": list,
        "confidence": (int, float),
        "metadata": dict,
    }

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> ResearchResult:
        """Rebuild a ``ResearchResult`` from a serialised dict, validating shape.

        Args:
            data: Dict produced by :meth:`to_dict` (possibly round-tripped
                through JSON by an orchestrator).

        Returns:
            A populated ``ResearchResult``.

        Raises:
            ValueError: When ``data`` is not a dict, a required field is
                missing, a field has the wrong type, or unknown fields are
                present.
        """
        if not isinstance(data, dict):
            raise ValueError(
                f"ResearchResult payload must be a dict, got {type(data).__name__}."
            )

        errors: list[str] = []
        if "query" not in data:
            errors.append("missing required field 'query'")

        unknown = sorted(set(data) - set(cls._FIELD_TYPES))
        if unknown:
            errors.append(f"unknown fields: {', '.join(unknown)}")

        for name, expected in cls._FIELD_TYPES.items():
            if name in data and not isinstance(data[name], expected):
                expected_names = (
                    "/".join(t.__name__ for t in expected)
                    if isinstance(expected, tuple)
                    else expected.__name__
                )
                errors.append(
                    f"field '{name}' must be {expected_names}, "
                    f"got {type(data[name]).__name__}"
                )

        for name in ("findings", "sources"):
            value = data.get(name)
            if isinstance(value, list) and not all(
                isinstance(item, str) for item in value
            ):
                errors.append(f"field '{name}' must contain only strings")

        if errors:
            raise ValueError(
                "Invalid ResearchResult payload: " + "; ".join(errors) + "."
            )

        return cls(**data)


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
        return asdict(self)

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
