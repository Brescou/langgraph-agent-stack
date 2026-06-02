"""domain_packs/rfp_assistant/pack.py — RFP analysis and response planning pack."""

from __future__ import annotations

import json

from pydantic import BaseModel

from domain_packs.common.prompt_safety import format_vertical_prompt
from domain_packs.common.structured_llm import StructuredLLMPack
from domain_packs.productivity.rfp_assistant.schemas import (
    RfpAssistantInput,
    RfpAssistantOutput,
)


class RfpAssistantPack(StructuredLLMPack):
    pack_id = "rfp_assistant"
    name = "RFP Assistant"
    description = (
        "Analyses an RFP document: extracts requirements, gaps, risks, "
        "and drafts a response plan with section outlines."
    )
    input_schema = RfpAssistantInput
    output_schema = RfpAssistantOutput
    primary_field = "query"

    @classmethod
    def build_prompt(cls, inp: BaseModel, *, reference_text: str = "") -> str:
        data = cls._coerce_input(inp).model_dump()
        rfp_body = str(data.get("rfp_text") or reference_text)
        schema = json.dumps(RfpAssistantOutput.model_json_schema(), indent=2)
        return format_vertical_prompt(
            task_instructions="You are an expert proposal manager. Analyse the RFP.",
            fields={
                "Project label": str(data["query"]),
                "Our capabilities": str(data.get("our_capabilities") or "not provided"),
                "RFP document": rfp_body or "(no document provided)",
            },
            output_schema_json=schema,
            reference_text="",
            closing_instructions=(
                "Set query to the project label. draft_sections keys: "
                "executive_summary, approach, team, pricing_notes."
            ),
        )
