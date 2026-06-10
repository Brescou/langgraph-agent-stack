"""tests/test_vertical_packs_validation.py — Validation/error coverage for the
unregulated productivity vertical packs (meeting_prep, rfp_assistant,
support_triage, executive_brief).

All LLM calls are mocked; no network access. Covers:
- happy path with precise field assertions,
- invalid LLM output (malformed JSON, missing required field, extra field),
- invalid pack input (extra field, max_length overflow),
- LLM-level exceptions surfaced as AgentExecutionError.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from agents.base_agent import (
    AgentExecutionError,
    AgentTimeoutError,
    AgentValidationError,
)
from domain_packs.productivity.executive_brief.pack import ExecutiveBriefPack
from domain_packs.productivity.executive_brief.schemas import (
    ExecutiveBriefInput,
    ExecutiveBriefOutput,
)
from domain_packs.productivity.meeting_prep.pack import MeetingPrepPack
from domain_packs.productivity.meeting_prep.schemas import (
    MeetingPrepInput,
    MeetingPrepOutput,
)
from domain_packs.productivity.rfp_assistant.pack import RfpAssistantPack
from domain_packs.productivity.rfp_assistant.schemas import (
    RfpAssistantInput,
    RfpAssistantOutput,
)
from domain_packs.productivity.support_triage.pack import SupportTriagePack
from domain_packs.productivity.support_triage.schemas import (
    SupportTriageInput,
    SupportTriageOutput,
)


def _mock_llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=content)
    return llm


def _mock_llm_json(payload: dict) -> MagicMock:
    return _mock_llm(json.dumps(payload))


MEETING_PREP_PAYLOAD = {
    "company": "Acme Corp",
    "person": "Jane Doe",
    "company_overview": "B2B SaaS for logistics",
    "recent_news": ["Raised Series B"],
    "talking_points": ["Platform expansion"],
    "questions_to_ask": ["What is your timeline?"],
    "landmines": ["Competitor X incumbent"],
    "confidence": 0.85,
}

RFP_PAYLOAD = {
    "query": "City IT RFP",
    "requirements": ["SOC2", "99.9% SLA"],
    "gaps": ["No pricing template"],
    "risks": ["Tight deadline"],
    "response_plan": ["Assign SME"],
    "draft_sections": {"executive_summary": "We propose..."},
    "confidence": 0.8,
}

SUPPORT_PAYLOAD = {
    "category": "billing",
    "priority": "high",
    "sentiment": "frustrated",
    "draft_reply": "We are looking into your invoice.",
    "escalate": True,
    "escalation_reason": "Payment dispute",
    "suggested_tags": ["billing"],
    "confidence": 0.9,
}

EXEC_BRIEF_PAYLOAD = {
    "audience": "CEO",
    "bullets": ["Revenue up 12%"],
    "so_what": "Invest now",
    "recommended_decisions": ["Approve budget"],
    "risks": ["Supply chain"],
    "confidence": 0.75,
}


def _meeting_prep_input() -> MeetingPrepInput:
    return MeetingPrepInput(company="Acme Corp", person="Jane Doe")


def _rfp_input() -> RfpAssistantInput:
    return RfpAssistantInput(query="City IT RFP", rfp_text="Must have SOC2.")


def _support_input() -> SupportTriageInput:
    return SupportTriageInput(ticket_subject="Wrong invoice", body="Charged twice.")


def _exec_brief_input() -> ExecutiveBriefInput:
    return ExecutiveBriefInput(text="Quarterly report: revenue up 12%.")


# (pack_cls, valid LLM payload, input factory, one required output key)
PACK_CASES = [
    (MeetingPrepPack, MEETING_PREP_PAYLOAD, _meeting_prep_input, "company_overview"),
    (RfpAssistantPack, RFP_PAYLOAD, _rfp_input, "requirements"),
    (SupportTriagePack, SUPPORT_PAYLOAD, _support_input, "draft_reply"),
    (ExecutiveBriefPack, EXEC_BRIEF_PAYLOAD, _exec_brief_input, "so_what"),
]

PACK_IDS = [case[0].pack_id for case in PACK_CASES]


# ---------------------------------------------------------------------------
# Happy paths — precise value assertions
# ---------------------------------------------------------------------------


def test_meeting_prep_happy_path_full_output() -> None:
    pack = MeetingPrepPack(run_id="mpv-1", llm=_mock_llm_json(MEETING_PREP_PAYLOAD))
    result = pack.run_from_input(_meeting_prep_input())
    assert isinstance(result, MeetingPrepOutput)
    assert result.company == "Acme Corp"
    assert result.person == "Jane Doe"
    assert result.company_overview == "B2B SaaS for logistics"
    assert result.recent_news == ["Raised Series B"]
    assert result.talking_points == ["Platform expansion"]
    assert result.questions_to_ask == ["What is your timeline?"]
    assert result.landmines == ["Competitor X incumbent"]
    assert result.confidence == 0.85
    assert result.cost_usd is None


def test_rfp_assistant_happy_path_full_output() -> None:
    pack = RfpAssistantPack(run_id="rfpv-1", llm=_mock_llm_json(RFP_PAYLOAD))
    result = pack.run_from_input(_rfp_input())
    assert isinstance(result, RfpAssistantOutput)
    assert result.query == "City IT RFP"
    assert result.requirements == ["SOC2", "99.9% SLA"]
    assert result.gaps == ["No pricing template"]
    assert result.risks == ["Tight deadline"]
    assert result.response_plan == ["Assign SME"]
    assert result.draft_sections == {"executive_summary": "We propose..."}
    assert result.confidence == 0.8


def test_support_triage_happy_path_full_output() -> None:
    pack = SupportTriagePack(run_id="stv-1", llm=_mock_llm_json(SUPPORT_PAYLOAD))
    result = pack.run_from_input(_support_input())
    assert isinstance(result, SupportTriageOutput)
    assert result.category == "billing"
    assert result.priority == "high"
    assert result.sentiment == "frustrated"
    assert result.draft_reply == "We are looking into your invoice."
    assert result.escalate is True
    assert result.escalation_reason == "Payment dispute"
    assert result.suggested_tags == ["billing"]
    assert result.confidence == 0.9


def test_executive_brief_happy_path_full_output() -> None:
    pack = ExecutiveBriefPack(run_id="ebv-1", llm=_mock_llm_json(EXEC_BRIEF_PAYLOAD))
    result = pack.run_from_input(_exec_brief_input())
    assert isinstance(result, ExecutiveBriefOutput)
    assert result.audience == "CEO"
    assert result.bullets == ["Revenue up 12%"]
    assert result.so_what == "Invest now"
    assert result.recommended_decisions == ["Approve budget"]
    assert result.risks == ["Supply chain"]
    assert result.confidence == 0.75


def test_support_triage_run_maps_query_to_subject_and_body() -> None:
    """``run()`` on support_triage fills both ticket_subject and body."""
    llm = _mock_llm_json(SUPPORT_PAYLOAD)
    pack = SupportTriagePack(run_id="stv-run", llm=llm)
    result = pack.run("Invoice was charged twice")
    assert result.category == "billing"
    prompt = llm.invoke.call_args[0][0]
    assert "Invoice was charged twice" in prompt


# ---------------------------------------------------------------------------
# Invalid LLM output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pack_cls", "payload", "input_factory", "required_key"),
    PACK_CASES,
    ids=PACK_IDS,
)
def test_malformed_llm_json_raises_execution_error(
    pack_cls, payload, input_factory, required_key
) -> None:
    pack = pack_cls(run_id="bad-json", llm=_mock_llm("this is not JSON at all"))
    with pytest.raises(AgentExecutionError, match="parse JSON"):
        pack.run_from_input(input_factory())


@pytest.mark.parametrize(
    ("pack_cls", "payload", "input_factory", "required_key"),
    PACK_CASES,
    ids=PACK_IDS,
)
def test_missing_required_output_field_raises_execution_error(
    pack_cls, payload, input_factory, required_key
) -> None:
    incomplete = {k: v for k, v in payload.items() if k != required_key}
    pack = pack_cls(run_id="missing-field", llm=_mock_llm_json(incomplete))
    with pytest.raises(AgentExecutionError, match="validation error"):
        pack.run_from_input(input_factory())


@pytest.mark.parametrize(
    ("pack_cls", "payload", "input_factory", "required_key"),
    PACK_CASES,
    ids=PACK_IDS,
)
def test_extra_llm_output_field_rejected(
    pack_cls, payload, input_factory, required_key
) -> None:
    """Output schemas use extra='forbid' — unexpected LLM keys must fail the run."""
    polluted = {**payload, "totally_unexpected_field": "injected"}
    pack = pack_cls(run_id="extra-field", llm=_mock_llm_json(polluted))
    with pytest.raises(AgentExecutionError, match="[Ee]xtra"):
        pack.run_from_input(input_factory())


def test_meeting_prep_non_strict_coerces_string_confidence() -> None:
    """Non-regulated packs validate without strict mode: '0.85' is coerced to float."""
    payload = {**MEETING_PREP_PAYLOAD, "confidence": "0.85"}
    pack = MeetingPrepPack(run_id="mpv-coerce", llm=_mock_llm_json(payload))
    result = pack.run_from_input(_meeting_prep_input())
    assert result.confidence == 0.85


# ---------------------------------------------------------------------------
# Invalid pack input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("input_cls", "kwargs"),
    [
        (MeetingPrepInput, {"company": "Acme"}),
        (RfpAssistantInput, {"query": "RFP"}),
        (SupportTriageInput, {"ticket_subject": "s", "body": "b"}),
        (ExecutiveBriefInput, {"text": "report"}),
    ],
    ids=PACK_IDS,
)
def test_input_extra_field_rejected(input_cls, kwargs) -> None:
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        input_cls(**kwargs, unexpected_extra="nope")


@pytest.mark.parametrize(
    ("input_cls", "kwargs"),
    [
        (MeetingPrepInput, {"company": "x" * 501}),
        (RfpAssistantInput, {"query": "x" * 501}),
        (SupportTriageInput, {"ticket_subject": "x" * 501, "body": "b"}),
        (ExecutiveBriefInput, {"text": "x" * 20001}),
    ],
    ids=PACK_IDS,
)
def test_input_max_length_rejected(input_cls, kwargs) -> None:
    with pytest.raises(ValidationError, match="at most"):
        input_cls(**kwargs)


def test_run_with_empty_query_raises_validation_error() -> None:
    pack = MeetingPrepPack(run_id="mpv-empty", llm=_mock_llm("{}"))
    with pytest.raises(AgentValidationError, match="needs text"):
        pack.run("   ")


# ---------------------------------------------------------------------------
# LLM-level failures
# ---------------------------------------------------------------------------


def test_llm_timeout_surfaces_as_execution_error() -> None:
    llm = MagicMock()
    llm.invoke.side_effect = AgentTimeoutError("LLM call timed out")
    pack = ExecutiveBriefPack(run_id="ebv-timeout", llm=llm)
    with pytest.raises(AgentExecutionError, match="timed out"):
        pack.run_from_input(_exec_brief_input())


def test_llm_unexpected_exception_wrapped_as_execution_error() -> None:
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("connection reset")
    pack = ExecutiveBriefPack(run_id="ebv-boom", llm=llm)
    with pytest.raises(AgentExecutionError, match="Pipeline failed"):
        pack.run_from_input(_exec_brief_input())


def test_pack_without_llm_raises_execution_error() -> None:
    pack = MeetingPrepPack(run_id="mpv-nollm", llm=None)
    with pytest.raises(AgentExecutionError, match="requires an LLM"):
        pack.run_from_input(_meeting_prep_input())
