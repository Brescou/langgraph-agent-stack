"""
control_plane — Pack-level policies and enforcement helpers.

Policies are registered explicitly (Approach B). The API applies constraints
via ``control_plane.enforce`` at request boundaries.
"""

from control_plane.enforce import (
    effective_budget_usd,
    effective_stream_timeout_seconds,
    validate_pack_body,
    validate_query_for_pack,
)
from control_plane.policies import ExecutionConstraints, PackPolicy
from control_plane.registry import PolicyRegistry
from domain_packs.common.compliance import (
    CONTRACT_REVIEWER_DISCLAIMER,
    FINANCIAL_MEMO_DISCLAIMER,
    HR_POLICY_QA_DISCLAIMER,
    JOB_DESCRIPTION_WRITER_DISCLAIMER,
    TALENT_SCREENING_DISCLAIMER,
)

_BUILTIN_POLICIES: tuple[PackPolicy, ...] = (
    PackPolicy(
        pack_id="research_analysis",
        constraints=ExecutionConstraints(max_query_chars=2000),
        labels=frozenset({"default", "full-pipeline"}),
    ),
    PackPolicy(
        pack_id="research_only",
        constraints=ExecutionConstraints(max_query_chars=2000),
        labels=frozenset({"research-only"}),
    ),
    PackPolicy(
        pack_id="summariser",
        constraints=ExecutionConstraints(max_query_chars=4000),
        labels=frozenset({"single-agent", "summarise"}),
    ),
    PackPolicy(
        pack_id="analysis_only",
        constraints=ExecutionConstraints(max_query_chars=2000),
        labels=frozenset({"analysis-only"}),
    ),
    PackPolicy(
        pack_id="meeting_prep",
        constraints=ExecutionConstraints(max_query_chars=2000),
        labels=frozenset({"sales", "meeting-prep"}),
    ),
    PackPolicy(
        pack_id="rfp_assistant",
        constraints=ExecutionConstraints(max_query_chars=500),
        labels=frozenset({"sales", "rfp", "rag"}),
    ),
    PackPolicy(
        pack_id="support_triage",
        constraints=ExecutionConstraints(max_query_chars=8000),
        labels=frozenset({"support", "single-agent"}),
    ),
    PackPolicy(
        pack_id="executive_brief",
        constraints=ExecutionConstraints(max_query_chars=20000),
        labels=frozenset({"executive", "summarise"}),
    ),
    PackPolicy(
        pack_id="contract_reviewer",
        constraints=ExecutionConstraints(max_query_chars=500),
        labels=frozenset({"legal", "rag", "regulated"}),
        human_review_required=True,
        compliance_disclaimer=CONTRACT_REVIEWER_DISCLAIMER,
        extensions={"output_integrity_fail_closed": True},
    ),
    PackPolicy(
        pack_id="financial_memo",
        constraints=ExecutionConstraints(max_query_chars=2000),
        labels=frozenset({"finance", "strategy", "regulated"}),
        human_review_required=True,
        compliance_disclaimer=FINANCIAL_MEMO_DISCLAIMER,
        extensions={"output_integrity_fail_closed": True},
    ),
    PackPolicy(
        pack_id="talent_screening",
        constraints=ExecutionConstraints(max_query_chars=10000),
        labels=frozenset({"hr", "recruiting", "regulated"}),
        human_review_required=True,
        compliance_disclaimer=TALENT_SCREENING_DISCLAIMER,
        extensions={"output_integrity_fail_closed": True},
    ),
    PackPolicy(
        pack_id="job_description_writer",
        constraints=ExecutionConstraints(max_query_chars=2000),
        labels=frozenset({"hr", "recruiting", "regulated"}),
        human_review_required=True,
        compliance_disclaimer=JOB_DESCRIPTION_WRITER_DISCLAIMER,
        extensions={"output_integrity_fail_closed": True},
    ),
    PackPolicy(
        pack_id="hr_policy_qa",
        constraints=ExecutionConstraints(max_query_chars=2000),
        labels=frozenset({"hr", "rag", "regulated"}),
        human_review_required=True,
        compliance_disclaimer=HR_POLICY_QA_DISCLAIMER,
        extensions={"output_integrity_fail_closed": True},
    ),
)

for _policy in _BUILTIN_POLICIES:
    PolicyRegistry.register(_policy)

__all__ = [
    "ExecutionConstraints",
    "PackPolicy",
    "PolicyRegistry",
    "effective_budget_usd",
    "effective_stream_timeout_seconds",
    "validate_pack_body",
    "validate_query_for_pack",
]
