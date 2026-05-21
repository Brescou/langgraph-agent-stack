"""domain_packs/summariser/schemas.py — Typed I/O for SummariserPack."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SummaryInput(BaseModel):
    """Input schema for the text summariser pack."""

    text: str = Field(
        ..., min_length=1, max_length=4000, description="Text to summarise"
    )
    bullet_count: int = Field(
        default=3, ge=1, le=10, description="Number of bullet points"
    )


class SummaryOutput(BaseModel):
    """Output schema for the text summariser pack."""

    original_length: int
    bullets: list[str]
    cost_usd: float | None = None

    @classmethod
    def from_summary_result(
        cls, result: Any, cost_usd: float | None = None
    ) -> SummaryOutput:
        """Build output from a ``SummaryOutput`` dataclass-like result or model."""
        if isinstance(result, SummaryOutput):
            return cls(
                original_length=result.original_length,
                bullets=result.bullets,
                cost_usd=cost_usd if cost_usd is not None else result.cost_usd,
            )
        if hasattr(result, "original_length") and hasattr(result, "bullets"):
            return cls(
                original_length=result.original_length,
                bullets=list(result.bullets),
                cost_usd=cost_usd,
            )
        raise TypeError(f"Expected summary result, got {type(result).__name__!r}")
