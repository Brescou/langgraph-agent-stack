"""domain_packs/executive_brief/pack.py — Executive summary brief pack."""

from __future__ import annotations

import json

from pydantic import BaseModel

from domain_packs.common.structured_llm import StructuredLLMPack
from domain_packs.executive_brief.schemas import (
    ExecutiveBriefInput,
    ExecutiveBriefOutput,
)


class ExecutiveBriefPack(StructuredLLMPack):
    pack_id = "executive_brief"
    name = "Executive Brief"
    description = (
        "Distils long content into executive bullets, a 'so what', "
        "recommended decisions, and risks for a target audience."
    )
    input_schema = ExecutiveBriefInput
    output_schema = ExecutiveBriefOutput

    @classmethod
    def build_prompt(cls, inp: BaseModel, *, reference_text: str = "") -> str:
        assert isinstance(inp, ExecutiveBriefInput)
        content = inp.text or reference_text
        schema = json.dumps(ExecutiveBriefOutput.model_json_schema(), indent=2)
        return (
            "You are a chief of staff preparing an executive brief.\n"
            f"Target audience: {inp.audience}\n"
            f"Number of bullets: {inp.bullet_count}\n\n"
            f"SOURCE MATERIAL:\n{content}\n\n"
            "Return ONLY valid JSON matching this schema:\n"
            f"{schema}\n"
            f"Provide exactly {inp.bullet_count} bullets. so_what must be one sharp paragraph."
        )
