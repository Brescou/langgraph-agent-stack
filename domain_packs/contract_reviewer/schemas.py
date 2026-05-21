"""domain_packs/contract_reviewer/schemas.py — Typed I/O for ContractReviewerPack."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ContractReviewerInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Contract label")
    contract_text: str = Field(default="", max_length=50000)
    contract_type: str = Field(default="msa", max_length=100)
    jurisdiction: str = Field(default="", max_length=100)


class ContractReviewerOutput(BaseModel):
    query: str
    risk_score: float = Field(ge=0.0, le=1.0)
    flagged_clauses: list[str]
    deviations_from_playbook: list[str]
    recommended_actions: list[str]
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    cost_usd: float | None = None
