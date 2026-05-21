"""domain_packs/rfp_assistant/pack.py — RFP analysis and response planning pack."""

from __future__ import annotations

import json

from pydantic import BaseModel

from domain_packs.common.structured_llm import StructuredLLMPack
from domain_packs.rfp_assistant.schemas import RfpAssistantInput, RfpAssistantOutput


class RfpAssistantPack(StructuredLLMPack):
    pack_id = "rfp_assistant"
    name = "RFP Assistant"
    description = (
        "Analyses an RFP document: extracts requirements, gaps, risks, "
        "and drafts a response plan with section outlines."
    )
    input_schema = RfpAssistantInput
    output_schema = RfpAssistantOutput

    @classmethod
    def build_prompt(cls, inp: BaseModel, *, reference_text: str = "") -> str:
        assert isinstance(inp, RfpAssistantInput)
        rfp_body = inp.rfp_text or reference_text
        schema = json.dumps(RfpAssistantOutput.model_json_schema(), indent=2)
        return (
            "You are an expert proposal manager. Analyse the RFP below.\n"
            f"Project label: {inp.query}\n"
            f"Our capabilities: {inp.our_capabilities or 'not provided'}\n\n"
            f"RFP DOCUMENT:\n{rfp_body or '(no document provided)'}\n\n"
            "Return ONLY valid JSON matching this schema:\n"
            f"{schema}\n"
            "Set query to the project label. draft_sections keys: executive_summary, "
            "approach, team, pricing_notes."
        )
