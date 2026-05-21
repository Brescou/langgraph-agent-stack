"""domain_packs/rfp_assistant/schemas.py — Typed I/O for RfpAssistantPack."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RfpAssistantInput(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Short label for this RFP (e.g. client name + project)",
    )
    rfp_text: str = Field(default="", max_length=20000)
    our_capabilities: str = Field(default="", max_length=4000)


class RfpAssistantOutput(BaseModel):
    query: str
    requirements: list[str]
    gaps: list[str]
    risks: list[str]
    response_plan: list[str]
    draft_sections: dict[str, str]
    confidence: float = Field(ge=0.0, le=1.0)
    cost_usd: float | None = None
