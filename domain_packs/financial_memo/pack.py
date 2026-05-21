"""domain_packs/financial_memo/pack.py — Consulting-style financial memo pack."""

from __future__ import annotations

import json

from pydantic import BaseModel

from domain_packs.common.structured_llm import StructuredLLMPack
from domain_packs.financial_memo.schemas import FinancialMemoInput, FinancialMemoOutput


class FinancialMemoPack(StructuredLLMPack):
    pack_id = "financial_memo"
    name = "Financial Memo"
    description = (
        "Produces a SCQA-style financial/strategy memo: situation, complications, "
        "options, recommendation, risks, and next steps."
    )
    input_schema = FinancialMemoInput
    output_schema = FinancialMemoOutput

    @classmethod
    def build_prompt(cls, inp: BaseModel, *, reference_text: str = "") -> str:
        assert isinstance(inp, FinancialMemoInput)
        ref = f"\n\nReference data:\n{reference_text}" if reference_text else ""
        schema = json.dumps(FinancialMemoOutput.model_json_schema(), indent=2)
        return (
            "You are a strategy consultant writing a financial memo (SCQA format).\n"
            f"Topic: {inp.topic}\n"
            f"Hypothesis: {inp.hypothesis or 'none stated'}\n"
            f"Key metrics/context: {inp.metrics or 'none provided'}\n"
            f"Time horizon: {inp.time_horizon}\n"
            f"{ref}\n\n"
            "Return ONLY valid JSON matching this schema:\n"
            f"{schema}\n"
            "Set topic from input. Be concise and decision-oriented."
        )
