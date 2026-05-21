"""domain_packs/rh/job_description_writer/pack.py — Inclusive job description writer."""

from __future__ import annotations

import json

from pydantic import BaseModel

from domain_packs.common.structured_llm import StructuredLLMPack
from domain_packs.rh.job_description_writer.schemas import (
    JobDescriptionWriterInput,
    JobDescriptionWriterOutput,
)


class JobDescriptionWriterPack(StructuredLLMPack):
    pack_id = "job_description_writer"
    name = "Job Description Writer"
    description = (
        "Drafts an inclusive job description, competency matrix, screening rubric, "
        "and bias-check notes from hiring-manager inputs."
    )
    input_schema = JobDescriptionWriterInput
    output_schema = JobDescriptionWriterOutput

    @classmethod
    def primary_text(cls, inp: BaseModel) -> str:
        assert isinstance(inp, JobDescriptionWriterInput)
        return inp.role_title

    @classmethod
    def build_prompt(cls, inp: BaseModel, *, reference_text: str = "") -> str:
        assert isinstance(inp, JobDescriptionWriterInput)
        schema = json.dumps(JobDescriptionWriterOutput.model_json_schema(), indent=2)
        return (
            "You are an HR business partner writing an inclusive job description.\n"
            f"Role: {inp.role_title}\n"
            f"Seniority: {inp.seniority}\n"
            f"Team context: {inp.team_context or 'not provided'}\n"
            f"Must-haves: {', '.join(inp.must_haves) or 'not specified'}\n"
            f"Culture notes: {inp.culture_notes or 'none'}\n\n"
            "Return ONLY valid JSON matching this schema:\n"
            f"{schema}\n"
            "jd_markdown should use inclusive language. Avoid gendered or age-biased terms."
        )
