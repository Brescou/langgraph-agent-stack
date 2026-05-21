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

PolicyRegistry.register(
    PackPolicy(
        pack_id="research_analysis",
        constraints=ExecutionConstraints(
            max_query_chars=2000,
            stream_timeout_seconds=None,
            budget_usd_ceiling=None,
        ),
        labels=frozenset({"default", "full-pipeline"}),
    )
)
PolicyRegistry.register(
    PackPolicy(
        pack_id="research_only",
        constraints=ExecutionConstraints(
            max_query_chars=2000,
            stream_timeout_seconds=None,
            budget_usd_ceiling=None,
        ),
        labels=frozenset({"research-only"}),
    )
)
PolicyRegistry.register(
    PackPolicy(
        pack_id="summariser",
        constraints=ExecutionConstraints(
            max_query_chars=4000,
            stream_timeout_seconds=None,
            budget_usd_ceiling=None,
        ),
        labels=frozenset({"single-agent", "summarise"}),
    )
)
PolicyRegistry.register(
    PackPolicy(
        pack_id="analysis_only",
        constraints=ExecutionConstraints(
            max_query_chars=2000,
            stream_timeout_seconds=None,
            budget_usd_ceiling=None,
        ),
        labels=frozenset({"analysis-only"}),
    )
)

__all__ = [
    "ExecutionConstraints",
    "PackPolicy",
    "PolicyRegistry",
    "effective_budget_usd",
    "effective_stream_timeout_seconds",
    "validate_query_for_pack",
]
