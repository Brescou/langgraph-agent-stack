"""domain_packs/support_triage/schemas.py — Typed I/O for SupportTriagePack."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SupportTriageInput(BaseModel):
    ticket_subject: str = Field(..., min_length=1, max_length=500)
    body: str = Field(..., min_length=1, max_length=8000)
    customer_tier: str = Field(default="standard", max_length=50)


class SupportTriageOutput(BaseModel):
    category: str
    priority: str
    sentiment: str
    draft_reply: str
    escalate: bool
    escalation_reason: str
    suggested_tags: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    cost_usd: float | None = None
