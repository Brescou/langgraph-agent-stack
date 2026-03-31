"""
api/models.py — Pydantic v2 request/response models for the FastAPI layer.

These models are intentionally decoupled from the agent dataclasses
(``ResearchResult``, ``AnalysisReport``).  Keeping the API contract separate
from the internal domain objects means either side can evolve independently
without forcing breaking changes on the other.

Serialisation helpers (``from_research_result``, ``from_analysis_report``) live
on the response models so the conversion logic stays co-located with the schema.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / generic
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response schema for ``GET /health``."""

    status: str = Field(
        description="Service health status. Always 'ok' when the endpoint responds.",
        examples=["ok"],
    )
    version: str = Field(
        description="Application version string.",
        examples=["0.1.0"],
    )
    uptime_seconds: float = Field(
        description="Seconds elapsed since the server process started.",
        ge=0.0,
    )
    environment: str = Field(
        description="Deployment environment tag (development / staging / production).",
        examples=["development"],
    )


# ---------------------------------------------------------------------------
# /run  —  full Research + Analysis pipeline
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """Request body for ``POST /run``."""

    query: str = Field(
        min_length=1,
        max_length=2000,
        description="Research question or topic to investigate.",
        examples=["What are the latest advancements in quantum computing?"],
    )


class RunResponse(BaseModel):
    """Response schema for ``POST /run``."""

    query: str = Field(description="The original query echoed back for correlation.")
    executive_summary: str = Field(
        description="One-paragraph high-level conclusion produced by the AnalystAgent."
    )
    key_insights: list[str] = Field(
        default_factory=list,
        description="Ordered list of the most important findings.",
    )
    patterns: list[str] = Field(
        default_factory=list,
        description="Identified recurring themes or structural patterns.",
    )
    implications: list[str] = Field(
        default_factory=list,
        description="Practical consequences and recommendations.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-reported confidence score between 0.0 and 1.0.",
    )
    research_summary: str = Field(
        description="The research summary that fed the analysis (for traceability)."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Run-level metadata forwarded from the agent pipeline.",
    )

    @classmethod
    def from_analysis_report(cls, report: Any) -> "RunResponse":
        """
        Build a ``RunResponse`` from an ``AnalysisReport`` dataclass instance.

        Args:
            report: An ``agents.analyst.AnalysisReport`` instance.

        Returns:
            A populated ``RunResponse`` ready for serialisation.
        """
        return cls(
            query=report.query,
            executive_summary=report.executive_summary,
            key_insights=report.key_insights,
            patterns=report.patterns,
            implications=report.implications,
            confidence=report.confidence,
            research_summary=report.research_summary,
            metadata=report.metadata,
        )


# ---------------------------------------------------------------------------
# /research  —  research-only pipeline
# ---------------------------------------------------------------------------


class ResearchRequest(BaseModel):
    """Request body for ``POST /research``."""

    query: str = Field(
        min_length=1,
        max_length=2000,
        description="Research question or topic to investigate.",
        examples=["Explain the CAP theorem in distributed systems."],
    )


class ResearchResponse(BaseModel):
    """Response schema for ``POST /research``."""

    query: str = Field(description="The original query echoed back for correlation.")
    summary: str = Field(
        description="LLM-generated summary of the consolidated research findings."
    )
    findings: list[str] = Field(
        default_factory=list,
        description="Raw text snippets collected during retrieval.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Source identifiers (URLs, document IDs, …) used during research.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-reported confidence score between 0.0 and 1.0.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Run-level metadata forwarded from the ResearchAgent.",
    )

    @classmethod
    def from_research_result(cls, result: Any) -> "ResearchResponse":
        """
        Build a ``ResearchResponse`` from a ``ResearchResult`` dataclass instance.

        Args:
            result: An ``agents.researcher.ResearchResult`` instance.

        Returns:
            A populated ``ResearchResponse`` ready for serialisation.
        """
        return cls(
            query=result.query,
            summary=result.summary,
            findings=result.findings,
            sources=result.sources,
            confidence=result.confidence,
            metadata=result.metadata,
        )
