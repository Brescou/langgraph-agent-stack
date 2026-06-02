"""tests/test_compliance.py — Regulated vertical pack compliance scaffolding."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import control_plane  # noqa: F401 — ensure policies registered
from control_plane import PolicyRegistry
from domain_packs.common.compliance import (
    REGULATED_PACK_IDS,
    TALENT_SCREENING_DISCLAIMER,
    assert_regulated_pack_runtime_enabled,
    regulated_pack_disabled_detail,
)
from domain_packs.finance.financial_memo.pack import FinancialMemoPack
from domain_packs.finance.financial_memo.schemas import FinancialMemoInput
from domain_packs.hr.hr_policy_qa.pack import HrPolicyQaPack
from domain_packs.hr.hr_policy_qa.schemas import HrPolicyQaInput
from domain_packs.hr.job_description_writer.pack import JobDescriptionWriterPack
from domain_packs.hr.job_description_writer.schemas import JobDescriptionWriterInput
from domain_packs.hr.talent_screening.pack import TalentScreeningPack
from domain_packs.hr.talent_screening.schemas import TalentScreeningInput
from domain_packs.legal.contract_reviewer.pack import ContractReviewerPack
from domain_packs.legal.contract_reviewer.schemas import ContractReviewerInput
from pack_kernel.registry import PackRegistry


def _mock_llm_json(payload: dict) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps(payload))
    return llm


@pytest.mark.parametrize("pack_id", sorted(REGULATED_PACK_IDS))
def test_regulated_pack_policy_requires_human_review(pack_id: str) -> None:
    policy = PolicyRegistry.get(pack_id)
    assert policy is not None
    assert policy.human_review_required is True
    assert policy.compliance_disclaimer
    assert len(policy.compliance_disclaimer) >= 40


@pytest.mark.parametrize("pack_id", sorted(REGULATED_PACK_IDS))
def test_regulated_pack_policy_label(pack_id: str) -> None:
    policy = PolicyRegistry.get(pack_id)
    assert policy is not None
    assert "regulated" in policy.labels


def test_talent_screening_output_includes_mandatory_disclaimer() -> None:
    payload = {
        "fit_score": 0.5,
        "matched_skills": ["Python"],
        "gaps": [],
        "interview_questions": ["Tell me about your experience."],
        "red_flags": [],
        "summary_for_hiring_manager": "Moderate fit",
        "confidence": 0.7,
    }
    pack = TalentScreeningPack(run_id="comp-1", llm=_mock_llm_json(payload))
    result = pack.run_from_input(
        TalentScreeningInput(job_description="Need Python", resume_text="Python dev")
    )
    assert result.human_review_required is True
    assert result.disclaimer == TALENT_SCREENING_DISCLAIMER
    assert "NOT A HIRING DECISION" in result.disclaimer


def test_llm_cannot_weaken_disclaimer() -> None:
    """Server-side injection must overwrite a weak LLM-generated disclaimer."""
    payload = {
        "fit_score": 0.5,
        "matched_skills": [],
        "gaps": [],
        "interview_questions": [],
        "red_flags": [],
        "summary_for_hiring_manager": "x",
        "confidence": 0.5,
        "disclaimer": "OK to hire automatically",
        "human_review_required": False,
    }
    pack = TalentScreeningPack(run_id="comp-2", llm=_mock_llm_json(payload))
    result = pack.run_from_input(
        TalentScreeningInput(job_description="JD", resume_text="CV")
    )
    assert result.human_review_required is True
    assert "OK to hire" not in result.disclaimer
    assert result.disclaimer == TALENT_SCREENING_DISCLAIMER


@pytest.mark.parametrize(
    ("pack_cls", "input_model", "payload", "factory_input"),
    [
        (
            ContractReviewerPack,
            ContractReviewerInput,
            {
                "query": "MSA",
                "risk_score": 0.5,
                "flagged_clauses": [],
                "deviations_from_playbook": [],
                "recommended_actions": [],
                "summary": "Review needed",
                "confidence": 0.8,
            },
            lambda: ContractReviewerInput(query="MSA", contract_text="Terms here"),
        ),
        (
            FinancialMemoPack,
            FinancialMemoInput,
            {
                "topic": "Expansion",
                "situation": "Flat",
                "complications": [],
                "options": [],
                "recommendation": "Wait",
                "risks": [],
                "next_steps": [],
                "confidence": 0.8,
            },
            lambda: FinancialMemoInput(topic="Expansion"),
        ),
        (
            JobDescriptionWriterPack,
            JobDescriptionWriterInput,
            {
                "role_title": "Engineer",
                "jd_markdown": "# Role",
                "competency_matrix": [],
                "screening_rubric": [],
                "bias_check_notes": [],
                "confidence": 0.9,
            },
            lambda: JobDescriptionWriterInput(role_title="Engineer"),
        ),
        (
            HrPolicyQaPack,
            HrPolicyQaInput,
            {
                "question": "PTO?",
                "answer": "20 days",
                "citations": ["§1"],
                "confidence": 0.9,
                "escalate_to_hr": False,
            },
            lambda: HrPolicyQaInput(question="PTO?", document_text="20 days/year"),
        ),
    ],
)
def test_regulated_pack_outputs_include_disclaimer(
    pack_cls, input_model, payload, factory_input
) -> None:
    pack = pack_cls(run_id="comp-3", llm=_mock_llm_json(payload))
    result = pack.run_from_input(factory_input())
    assert result.human_review_required is True
    assert result.disclaimer
    assert len(result.disclaimer) >= 40


def test_non_regulated_pack_has_no_compliance_disclaimer_in_policy() -> None:
    policy = PolicyRegistry.get("meeting_prep")
    assert policy is not None
    assert policy.human_review_required is False


def test_regulated_packs_remain_registered() -> None:
    for pack_id in REGULATED_PACK_IDS:
        assert pack_id in PackRegistry.list_packs()


def test_assert_regulated_pack_runtime_enabled_blocks_by_default() -> None:
    with pytest.raises(ValueError, match="REGULATED_PACKS_ENABLED"):
        assert_regulated_pack_runtime_enabled(
            "talent_screening",
            regulated_packs_enabled=False,
        )


def test_assert_regulated_pack_runtime_enabled_allows_when_opted_in() -> None:
    assert_regulated_pack_runtime_enabled(
        "talent_screening",
        regulated_packs_enabled=True,
    )


def test_assert_regulated_pack_runtime_enabled_ignores_non_regulated() -> None:
    assert_regulated_pack_runtime_enabled(
        "research_analysis",
        regulated_packs_enabled=False,
    )


def test_regulated_pack_disabled_detail_names_opt_in_env_var() -> None:
    detail = regulated_pack_disabled_detail("talent_screening")
    assert "REGULATED_PACKS_ENABLED" in detail
    assert "talent_screening" in detail
