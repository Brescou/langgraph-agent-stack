"""domain_packs/meeting_prep/pack.py — Sales meeting preparation pack."""

from __future__ import annotations

import json

from pydantic import BaseModel

from domain_packs.common.structured_llm import StructuredLLMPack
from domain_packs.meeting_prep.schemas import MeetingPrepInput, MeetingPrepOutput


class MeetingPrepPack(StructuredLLMPack):
    pack_id = "meeting_prep"
    name = "Meeting Prep"
    description = (
        "Generates a structured sales meeting brief: company overview, news, "
        "talking points, questions, and landmines."
    )
    input_schema = MeetingPrepInput
    output_schema = MeetingPrepOutput

    @classmethod
    def build_prompt(cls, inp: BaseModel, *, reference_text: str = "") -> str:
        assert isinstance(inp, MeetingPrepInput)
        ref = f"\n\nReference material:\n{reference_text}" if reference_text else ""
        schema = json.dumps(MeetingPrepOutput.model_json_schema(), indent=2)
        return (
            "You are a B2B sales intelligence assistant. Prepare a meeting brief.\n"
            f"Company: {inp.company}\n"
            f"Contact: {inp.person or 'unknown'}\n"
            f"Meeting goal: {inp.meeting_goal}\n"
            f"Additional context: {inp.context or 'none'}\n"
            f"{ref}\n\n"
            "Return ONLY valid JSON matching this schema (no markdown):\n"
            f"{schema}\n"
            "Set company and person fields from the input. Be specific and actionable."
        )
