"""domain_packs/support_triage/pack.py — Customer support ticket triage pack."""

from __future__ import annotations

import json

from pydantic import BaseModel

from domain_packs.common.structured_llm import StructuredLLMPack
from domain_packs.support_triage.schemas import SupportTriageInput, SupportTriageOutput


class SupportTriagePack(StructuredLLMPack):
    pack_id = "support_triage"
    name = "Support Triage"
    description = (
        "Triages a support ticket: category, priority, sentiment, draft reply, "
        "and escalation recommendation."
    )
    input_schema = SupportTriageInput
    output_schema = SupportTriageOutput

    @classmethod
    def build_prompt(cls, inp: BaseModel, *, reference_text: str = "") -> str:
        assert isinstance(inp, SupportTriageInput)
        kb = f"\n\nKnowledge base snippets:\n{reference_text}" if reference_text else ""
        schema = json.dumps(SupportTriageOutput.model_json_schema(), indent=2)
        return (
            "You are a senior customer support lead. Triage this ticket.\n"
            f"Customer tier: {inp.customer_tier}\n"
            f"Subject: {inp.ticket_subject}\n"
            f"Body:\n{inp.body}\n"
            f"{kb}\n\n"
            "Return ONLY valid JSON matching this schema:\n"
            f"{schema}\n"
            "priority must be one of: low, medium, high, critical. "
            "sentiment: positive, neutral, negative, frustrated."
        )
