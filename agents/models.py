"""agents/models.py — Shared data models for agent outputs.

``ResearchResult`` and ``AnalysisReport`` form the inter-agent contract
between ``ResearchAgent`` and ``AnalystAgent``.  Both are strict Pydantic
models (``extra="forbid"``) so that malformed payloads from upstream
orchestrators fail fast with clear, aggregated error messages.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _ContractModel(BaseModel):
    """Base class for the strict inter-agent contract models."""

    model_config = ConfigDict(extra="forbid")

    def to_dict(self) -> dict[str, Any]:
        """Serialise the model to a plain dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Any) -> Self:
        """Rebuild a model instance from a serialised dict, validating shape.

        Args:
            data: Dict produced by :meth:`to_dict` (possibly round-tripped
                through JSON by an orchestrator).

        Returns:
            A populated model instance.

        Raises:
            ValueError: When ``data`` is not a dict, a required field is
                missing, a field has the wrong type, or unknown fields are
                present.
        """
        if not isinstance(data, dict):
            raise ValueError(
                f"{cls.__name__} payload must be a dict, got {type(data).__name__}."
            )

        try:
            return cls.model_validate(data, strict=True)
        except ValidationError as exc:
            errors: list[str] = []
            unknown: list[str] = []
            for err in exc.errors():
                loc = ".".join(str(part) for part in err["loc"])
                if err["type"] == "missing":
                    errors.append(f"missing required field '{loc}'")
                elif err["type"] == "extra_forbidden":
                    unknown.append(loc)
                else:
                    errors.append(f"field '{loc}': {err['msg']}")
            if unknown:
                errors.append(f"unknown fields: {', '.join(sorted(unknown))}")
            raise ValueError(
                f"Invalid {cls.__name__} payload: " + "; ".join(errors) + "."
            ) from exc


class ResearchResult(_ContractModel):
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
    findings: list[str] = Field(default_factory=list)
    summary: str = ""
    sources: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisReport(_ContractModel):
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
    key_insights: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    implications: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    research_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

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
