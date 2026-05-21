"""domain_packs/meeting_prep/schemas.py — Typed I/O for MeetingPrepPack."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MeetingPrepInput(BaseModel):
    company: str = Field(..., min_length=1, max_length=500)
    person: str = Field(default="", max_length=200)
    meeting_goal: str = Field(default="discovery call", max_length=500)
    context: str = Field(default="", max_length=2000)


class MeetingPrepOutput(BaseModel):
    company: str
    person: str
    company_overview: str
    recent_news: list[str]
    talking_points: list[str]
    questions_to_ask: list[str]
    landmines: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    cost_usd: float | None = None
