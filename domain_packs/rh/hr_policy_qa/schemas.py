"""domain_packs/rh/hr_policy_qa/schemas.py — Typed I/O for HrPolicyQaPack."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HrPolicyQaInput(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    employee_context: str = Field(default="", max_length=2000)
    document_text: str = Field(default="", max_length=30000)


class HrPolicyQaOutput(BaseModel):
    question: str
    answer: str
    citations: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    escalate_to_hr: bool
    disclaimer: str
    cost_usd: float | None = None
