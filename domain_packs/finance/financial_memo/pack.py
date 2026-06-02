"""domain_packs/financial_memo/pack.py — Consulting-style financial memo pack."""

from __future__ import annotations

import json

from pydantic import BaseModel

from domain_packs.common.prompt_safety import format_vertical_prompt
from domain_packs.common.structured_llm import StructuredLLMPack
from domain_packs.finance.financial_memo.schemas import (
    FinancialMemoInput,
    FinancialMemoOutput,
)


class FinancialMemoPack(StructuredLLMPack):
    pack_id = "financial_memo"
    name = "Financial Memo"
    description = (
        "Produces a SCQA-style financial/strategy memo: situation, complications, "
        "options, recommendation, risks, and next steps."
    )
    input_schema = FinancialMemoInput
    output_schema = FinancialMemoOutput
    primary_field = "topic"

    @classmethod
    def build_prompt(cls, inp: BaseModel, *, reference_text: str = "") -> str:
        data = cls._coerce_input(inp).model_dump()
        schema = json.dumps(FinancialMemoOutput.model_json_schema(), indent=2)
        return format_vertical_prompt(
            task_instructions=(
                "You are a strategy consultant writing a financial memo (SCQA format)."
            ),
            fields={
                "Topic": str(data["topic"]),
                "Hypothesis": str(data.get("hypothesis") or "none stated"),
                "Key metrics/context": str(data.get("metrics") or "none provided"),
                "Time horizon": str(data.get("time_horizon", "12 months")),
            },
            output_schema_json=schema,
            reference_text=reference_text,
            closing_instructions="Set topic from input. Be concise and decision-oriented.",
        )
