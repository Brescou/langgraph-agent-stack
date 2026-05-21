"""tests/test_vertical_packs.py — Vertical / sellable domain pack tests."""

from __future__ import annotations

import json
from platform.registry import PackRegistry
from unittest.mock import MagicMock

import pytest

from domain_packs.contract_reviewer.pack import ContractReviewerPack
from domain_packs.contract_reviewer.schemas import ContractReviewerInput
from domain_packs.executive_brief.pack import ExecutiveBriefPack
from domain_packs.executive_brief.schemas import ExecutiveBriefInput
from domain_packs.financial_memo.pack import FinancialMemoPack
from domain_packs.financial_memo.schemas import FinancialMemoInput
from domain_packs.meeting_prep.pack import MeetingPrepPack
from domain_packs.meeting_prep.schemas import MeetingPrepInput, MeetingPrepOutput
from domain_packs.rfp_assistant.pack import RfpAssistantPack
from domain_packs.rfp_assistant.schemas import RfpAssistantInput
from domain_packs.rh.hr_policy_qa.pack import HrPolicyQaPack
from domain_packs.rh.hr_policy_qa.schemas import HrPolicyQaInput
from domain_packs.rh.job_description_writer.pack import JobDescriptionWriterPack
from domain_packs.rh.job_description_writer.schemas import JobDescriptionWriterInput
from domain_packs.rh.talent_screening.pack import TalentScreeningPack
from domain_packs.rh.talent_screening.schemas import TalentScreeningInput
from domain_packs.support_triage.pack import SupportTriagePack
from domain_packs.support_triage.schemas import SupportTriageInput

VERTICAL_PACK_IDS = (
    "meeting_prep",
    "rfp_assistant",
    "support_triage",
    "executive_brief",
    "contract_reviewer",
    "financial_memo",
    "talent_screening",
    "job_description_writer",
    "hr_policy_qa",
)


@pytest.mark.parametrize("pack_id", VERTICAL_PACK_IDS)
def test_vertical_pack_is_registered(pack_id: str) -> None:
    assert pack_id in PackRegistry.list_packs()
    pack_cls = PackRegistry.get(pack_id)
    assert pack_cls.pack_id == pack_id
    assert pack_cls.input_schema is not None
    assert pack_cls.output_schema is not None


def _mock_llm_json(payload: dict) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps(payload))
    return llm


def test_meeting_prep_run_from_input() -> None:
    payload = {
        "company": "Acme Corp",
        "person": "Jane Doe",
        "company_overview": "B2B SaaS",
        "recent_news": ["Raised Series B"],
        "talking_points": ["Platform expansion"],
        "questions_to_ask": ["What is your timeline?"],
        "landmines": ["Competitor X"],
        "confidence": 0.85,
    }
    pack = MeetingPrepPack(run_id="mp-1", llm=_mock_llm_json(payload))
    result = pack.run_from_input(
        MeetingPrepInput(company="Acme Corp", person="Jane Doe")
    )
    assert isinstance(result, MeetingPrepOutput)
    assert result.company == "Acme Corp"
    assert result.talking_points


def test_support_triage_run_from_input() -> None:
    payload = {
        "category": "billing",
        "priority": "high",
        "sentiment": "frustrated",
        "draft_reply": "We are looking into your invoice.",
        "escalate": True,
        "escalation_reason": "Payment dispute over $10k",
        "suggested_tags": ["billing", "enterprise"],
        "confidence": 0.9,
    }
    pack = SupportTriagePack(run_id="st-1", llm=_mock_llm_json(payload))
    result = pack.run_from_input(
        SupportTriageInput(ticket_subject="Wrong invoice", body="Charged twice.")
    )
    assert result.escalate is True
    assert result.category == "billing"


@pytest.mark.parametrize(
    ("pack_cls", "input_model", "payload", "field", "expected"),
    [
        (
            RfpAssistantPack,
            RfpAssistantInput,
            {
                "query": "City IT RFP",
                "requirements": ["SOC2"],
                "gaps": ["No pricing template"],
                "risks": ["Tight deadline"],
                "response_plan": ["Assign SME"],
                "draft_sections": {"executive_summary": "We propose..."},
                "confidence": 0.8,
            },
            "requirements",
            ["SOC2"],
        ),
        (
            ExecutiveBriefPack,
            ExecutiveBriefInput,
            {
                "audience": "CEO",
                "bullets": ["Revenue up"],
                "so_what": "Invest now",
                "recommended_decisions": ["Approve budget"],
                "risks": ["Supply chain"],
                "confidence": 0.75,
            },
            "so_what",
            "Invest now",
        ),
        (
            ContractReviewerPack,
            ContractReviewerInput,
            {
                "query": "Vendor MSA",
                "risk_score": 0.7,
                "flagged_clauses": ["Unlimited liability"],
                "deviations_from_playbook": ["Net 90 payment"],
                "recommended_actions": ["Negotiate liability cap"],
                "summary": "High risk MSA",
                "confidence": 0.8,
            },
            "risk_score",
            0.7,
        ),
        (
            FinancialMemoPack,
            FinancialMemoInput,
            {
                "topic": "Market entry",
                "situation": "Flat growth",
                "complications": ["Regulation"],
                "options": ["Partner", "Acquire"],
                "recommendation": "Partner first",
                "risks": ["FX exposure"],
                "next_steps": ["Run diligence"],
                "confidence": 0.82,
            },
            "recommendation",
            "Partner first",
        ),
        (
            TalentScreeningPack,
            TalentScreeningInput,
            {
                "fit_score": 0.78,
                "matched_skills": ["Python"],
                "gaps": ["Kubernetes"],
                "interview_questions": ["Describe scaling experience"],
                "red_flags": [],
                "summary_for_hiring_manager": "Strong backend fit",
                "confidence": 0.85,
            },
            "fit_score",
            0.78,
        ),
        (
            JobDescriptionWriterPack,
            JobDescriptionWriterInput,
            {
                "role_title": "Backend Engineer",
                "jd_markdown": "# Backend Engineer",
                "competency_matrix": ["Python", "API design"],
                "screening_rubric": ["System design"],
                "bias_check_notes": ["Avoid 'rockstar'"],
                "confidence": 0.9,
            },
            "role_title",
            "Backend Engineer",
        ),
        (
            HrPolicyQaPack,
            HrPolicyQaInput,
            {
                "question": "How many PTO days?",
                "answer": "25 days per year",
                "citations": ["Handbook §4.2"],
                "confidence": 0.95,
                "escalate_to_hr": False,
                "disclaimer": "Informational only",
            },
            "answer",
            "25 days per year",
        ),
    ],
)
def test_vertical_pack_run_from_input_parametrized(
    pack_cls, input_model, payload, field, expected
) -> None:
    llm = _mock_llm_json(payload)
    pack = pack_cls(run_id="v-1", llm=llm)
    if pack_cls is RfpAssistantPack:
        inp = input_model(query="City IT RFP", rfp_text="Must have SOC2.")
    elif pack_cls is ExecutiveBriefPack:
        inp = input_model(text="Long report content here.")
    elif pack_cls is ContractReviewerPack:
        inp = input_model(query="Vendor MSA", contract_text="Liability unlimited.")
    elif pack_cls is FinancialMemoPack:
        inp = input_model(topic="Market entry")
    elif pack_cls is TalentScreeningPack:
        inp = input_model(job_description="Need Python", resume_text="5y Python")
    elif pack_cls is JobDescriptionWriterPack:
        inp = input_model(role_title="Backend Engineer")
    else:
        inp = input_model(question="How many PTO days?", document_text="25 days/year")
    result = pack.run_from_input(inp)
    assert getattr(result, field) == expected
