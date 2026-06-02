"""domain_packs/meeting_prep/pack.py — Sales meeting preparation pack."""

from __future__ import annotations

import json

from pydantic import BaseModel

from domain_packs.common.prompt_safety import format_vertical_prompt
from domain_packs.common.structured_llm import StructuredLLMPack
from domain_packs.productivity.meeting_prep.schemas import (
    MeetingPrepInput,
    MeetingPrepOutput,
)


class MeetingPrepPack(StructuredLLMPack):
    pack_id = "meeting_prep"
    name = "Meeting Prep"
    description = (
        "Generates a structured sales meeting brief: company overview, news, "
        "talking points, questions, and landmines."
    )
    input_schema = MeetingPrepInput
    output_schema = MeetingPrepOutput
    primary_field = "company"

    @classmethod
    def build_prompt(cls, inp: BaseModel, *, reference_text: str = "") -> str:
        data = cls._coerce_input(inp).model_dump()
        schema = json.dumps(MeetingPrepOutput.model_json_schema(), indent=2)
        return format_vertical_prompt(
            task_instructions=(
                "You are a B2B sales intelligence assistant. Prepare a meeting brief."
            ),
            fields={
                "Company": str(data["company"]),
                "Contact": str(data.get("person") or "unknown"),
                "Meeting goal": str(data.get("meeting_goal", "")),
                "Additional context": str(data.get("context") or "none"),
            },
            output_schema_json=schema,
            reference_text=reference_text,
            closing_instructions=(
                "Set company and person fields from the input. Be specific and actionable."
            ),
        )
