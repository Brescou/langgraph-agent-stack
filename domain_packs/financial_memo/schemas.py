"""domain_packs/financial_memo/schemas.py — Typed I/O for FinancialMemoPack."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FinancialMemoInput(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    hypothesis: str = Field(default="", max_length=2000)
    metrics: str = Field(default="", max_length=2000)
    time_horizon: str = Field(default="12 months", max_length=100)


class FinancialMemoOutput(BaseModel):
    topic: str
    situation: str
    complications: list[str]
    options: list[str]
    recommendation: str
    risks: list[str]
    next_steps: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    cost_usd: float | None = None
