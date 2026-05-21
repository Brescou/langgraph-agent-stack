"""domain_packs/rh/hr_policy_qa/pack.py — HR handbook / policy Q&A pack."""

from __future__ import annotations

import json

from pydantic import BaseModel

from domain_packs.common.structured_llm import StructuredLLMPack
from domain_packs.rh.hr_policy_qa.schemas import HrPolicyQaInput, HrPolicyQaOutput


class HrPolicyQaPack(StructuredLLMPack):
    pack_id = "hr_policy_qa"
    name = "HR Policy Q&A"
    description = (
        "Answers employee HR policy questions with citations, confidence, "
        "and escalation guidance when sensitive."
    )
    input_schema = HrPolicyQaInput
    output_schema = HrPolicyQaOutput

    @classmethod
    def build_prompt(cls, inp: BaseModel, *, reference_text: str = "") -> str:
        assert isinstance(inp, HrPolicyQaInput)
        policy = inp.document_text or reference_text
        schema = json.dumps(HrPolicyQaOutput.model_json_schema(), indent=2)
        return (
            "You are an HR policy assistant. Answer ONLY from the policy material provided.\n"
            f"Employee context: {inp.employee_context or 'not provided'}\n"
            f"Question: {inp.question}\n\n"
            f"POLICY MATERIAL:\n{policy or '(no policy document — answer generally with low confidence)'}\n\n"
            "Return ONLY valid JSON matching this schema:\n"
            f"{schema}\n"
            "Set question from input. escalate_to_hr=true for legal/sensitive topics. "
            "disclaimer must state this is informational, not legal advice."
        )
