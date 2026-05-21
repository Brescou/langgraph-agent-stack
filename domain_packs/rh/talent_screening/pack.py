"""domain_packs/rh/talent_screening/pack.py — CV vs job description screening pack."""

from __future__ import annotations

import json

from pydantic import BaseModel

from domain_packs.common.structured_llm import StructuredLLMPack
from domain_packs.rh.talent_screening.schemas import (
    TalentScreeningInput,
    TalentScreeningOutput,
)


class TalentScreeningPack(StructuredLLMPack):
    pack_id = "talent_screening"
    name = "Talent Screening"
    description = (
        "Screens a candidate resume against a job description: fit score, "
        "skill gaps, interview questions, and red flags."
    )
    input_schema = TalentScreeningInput
    output_schema = TalentScreeningOutput

    @classmethod
    def primary_text(cls, inp: BaseModel) -> str:
        assert isinstance(inp, TalentScreeningInput)
        return inp.job_description[:500]

    @classmethod
    def build_prompt(cls, inp: BaseModel, *, reference_text: str = "") -> str:
        assert isinstance(inp, TalentScreeningInput)
        schema = json.dumps(TalentScreeningOutput.model_json_schema(), indent=2)
        return (
            "You are an expert technical recruiter. Screen this candidate.\n"
            f"Must-have skills: {', '.join(inp.must_have_skills) or 'not specified'}\n"
            f"Nice-to-have skills: {', '.join(inp.nice_to_have_skills) or 'not specified'}\n\n"
            f"JOB DESCRIPTION:\n{inp.job_description}\n\n"
            f"RESUME:\n{inp.resume_text}\n\n"
            "Return ONLY valid JSON matching this schema:\n"
            f"{schema}\n"
            "fit_score: 0=poor fit, 1=excellent fit. Be objective and cite evidence."
        )
