"""domain_packs/contract_reviewer/pack.py — Contract review and risk flagging pack."""

from __future__ import annotations

import json

from pydantic import BaseModel

from domain_packs.common.structured_llm import StructuredLLMPack
from domain_packs.contract_reviewer.schemas import (
    ContractReviewerInput,
    ContractReviewerOutput,
)


class ContractReviewerPack(StructuredLLMPack):
    pack_id = "contract_reviewer"
    name = "Contract Reviewer"
    description = (
        "Reviews a contract against standard playbook expectations: "
        "flags risky clauses, deviations, and recommended actions."
    )
    input_schema = ContractReviewerInput
    output_schema = ContractReviewerOutput

    @classmethod
    def build_prompt(cls, inp: BaseModel, *, reference_text: str = "") -> str:
        assert isinstance(inp, ContractReviewerInput)
        contract = inp.contract_text or reference_text
        schema = json.dumps(ContractReviewerOutput.model_json_schema(), indent=2)
        return (
            "You are a commercial legal reviewer. Analyse the contract below.\n"
            f"Label: {inp.query}\n"
            f"Type: {inp.contract_type}\n"
            f"Jurisdiction: {inp.jurisdiction or 'unspecified'}\n\n"
            f"CONTRACT:\n{contract or '(no text provided)'}\n\n"
            "Return ONLY valid JSON matching this schema:\n"
            f"{schema}\n"
            "Set query to the label. risk_score: 0=low risk, 1=critical risk."
        )
