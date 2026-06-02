"""domain_packs/support_triage/pack.py — Customer support ticket triage pack."""

from __future__ import annotations

import json

from pydantic import BaseModel

from domain_packs.common.prompt_safety import format_vertical_prompt
from domain_packs.common.structured_llm import StructuredLLMPack
from domain_packs.productivity.support_triage.schemas import (
    SupportTriageInput,
    SupportTriageOutput,
)


class SupportTriagePack(StructuredLLMPack):
    pack_id = "support_triage"
    name = "Support Triage"
    description = (
        "Triages a support ticket: category, priority, sentiment, draft reply, "
        "and escalation recommendation."
    )
    input_schema = SupportTriageInput
    output_schema = SupportTriageOutput
    primary_field = "ticket_subject"

    @classmethod
    def build_prompt(cls, inp: BaseModel, *, reference_text: str = "") -> str:
        data = cls._coerce_input(inp).model_dump()
        schema = json.dumps(SupportTriageOutput.model_json_schema(), indent=2)
        return format_vertical_prompt(
            task_instructions="You are a senior customer support lead. Triage this ticket.",
            fields={
                "Customer tier": str(data.get("customer_tier", "standard")),
                "Subject": str(data["ticket_subject"]),
                "Ticket body": str(data["body"]),
            },
            output_schema_json=schema,
            reference_text=reference_text,
            closing_instructions=(
                "priority must be one of: low, medium, high, critical. "
                "sentiment: positive, neutral, negative, frustrated."
            ),
        )
