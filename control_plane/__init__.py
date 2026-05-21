"""
control_plane — Pack-level policies and enforcement helpers.

Policies are registered explicitly (Approach B). The API applies constraints
via ``control_plane.enforce`` at request boundaries.
"""

from control_plane.enforce import (
    effective_budget_usd,
    effective_stream_timeout_seconds,
    validate_query_for_pack,
)
from control_plane.policies import ExecutionConstraints, PackPolicy
from control_plane.registry import PolicyRegistry

_BUILTIN_POLICIES: tuple[tuple[str, ExecutionConstraints, frozenset[str]], ...] = (
    (
        "research_analysis",
        ExecutionConstraints(max_query_chars=2000),
        frozenset({"default", "full-pipeline"}),
    ),
    (
        "research_only",
        ExecutionConstraints(max_query_chars=2000),
        frozenset({"research-only"}),
    ),
    (
        "summariser",
        ExecutionConstraints(max_query_chars=4000),
        frozenset({"single-agent", "summarise"}),
    ),
    (
        "analysis_only",
        ExecutionConstraints(max_query_chars=2000),
        frozenset({"analysis-only"}),
    ),
    (
        "meeting_prep",
        ExecutionConstraints(max_query_chars=2000),
        frozenset({"sales", "meeting-prep"}),
    ),
    (
        "rfp_assistant",
        ExecutionConstraints(max_query_chars=500),
        frozenset({"sales", "rfp", "rag"}),
    ),
    (
        "support_triage",
        ExecutionConstraints(max_query_chars=8000),
        frozenset({"support", "single-agent"}),
    ),
    (
        "executive_brief",
        ExecutionConstraints(max_query_chars=20000),
        frozenset({"executive", "summarise"}),
    ),
    (
        "contract_reviewer",
        ExecutionConstraints(max_query_chars=500),
        frozenset({"legal", "rag"}),
    ),
    (
        "financial_memo",
        ExecutionConstraints(max_query_chars=2000),
        frozenset({"finance", "strategy"}),
    ),
    (
        "talent_screening",
        ExecutionConstraints(max_query_chars=10000),
        frozenset({"hr", "recruiting"}),
    ),
    (
        "job_description_writer",
        ExecutionConstraints(max_query_chars=2000),
        frozenset({"hr", "recruiting"}),
    ),
    (
        "hr_policy_qa",
        ExecutionConstraints(max_query_chars=2000),
        frozenset({"hr", "rag"}),
    ),
)

for _pack_id, _constraints, _labels in _BUILTIN_POLICIES:
    PolicyRegistry.register(
        PackPolicy(pack_id=_pack_id, constraints=_constraints, labels=_labels)
    )

__all__ = [
    "ExecutionConstraints",
    "PackPolicy",
    "PolicyRegistry",
    "effective_budget_usd",
    "effective_stream_timeout_seconds",
    "validate_query_for_pack",
]
