"""domain_packs/executive_brief/schemas.py — Typed I/O for ExecutiveBriefPack."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExecutiveBriefInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)
    audience: str = Field(default="CEO", max_length=100)
    bullet_count: int = Field(default=5, ge=1, le=10)


class ExecutiveBriefOutput(BaseModel):
    audience: str
    bullets: list[str]
    so_what: str
    recommended_decisions: list[str]
    risks: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    cost_usd: float | None = None
