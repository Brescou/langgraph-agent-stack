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
# Session history models
# ---------------------------------------------------------------------------


class HistoryEntry(BaseModel):
    """A single run record returned inside :class:`HistoryResponse`."""

    run_id: str = Field(description="Unique identifier for the agent run.")
    query: str = Field(description="The original user query submitted for this run.")
    result_summary: str = Field(
        description="Truncated (≤ 200 chars) serialisation of the run result."
    )
    created_at: str = Field(
        description="ISO-8601 UTC timestamp when the run was persisted."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Run-level metadata stored alongside the result.",
    )


class HistoryResponse(BaseModel):
    """Response schema for ``GET /sessions/{session_id}/history``."""

    session_id: str = Field(
        description="The session ID whose run history is returned."
    )
    entries: list[HistoryEntry] = Field(
        description="Ordered list of run history entries (newest first)."
    )
    total: int = Field(
        ge=0,
        description="Total number of entries returned.",
    )


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
    session_id: str | None = Field(
        default=None,
        description=(
            "Optional session ID for conversation continuity. "
            "Auto-generated if absent."
        ),
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
    session_id: str = Field(
        description="Session ID used for this run (echoed from the request or auto-generated).",
    )

    @classmethod
    def from_analysis_report(cls, report: Any, session_id: str = "") -> RunResponse:
        """
        Build a ``RunResponse`` from an ``AnalysisReport`` dataclass instance.

        Args:
            report: An ``agents.analyst.AnalysisReport`` instance.
            session_id: The session ID associated with this run.

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
            session_id=session_id,
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
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for conversation continuity.",
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
    session_id: str = Field(
        description="Session ID used for this run (echoed from the request or auto-generated).",
    )

    @classmethod
    def from_research_result(cls, result: Any, session_id: str = "") -> ResearchResponse:
        """
        Build a ``ResearchResponse`` from a ``ResearchResult`` dataclass instance.

        Args:
            result: An ``agents.researcher.ResearchResult`` instance.
            session_id: The session ID associated with this run.

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
            session_id=session_id,
        )
