"""
control_plane/policies.py — Typed placeholders for pack governance (foundation only).

Nothing here is evaluated automatically. Call sites (future API or packs) may read
these records to align behaviour with ``PackRegistry`` entries keyed by ``pack_id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ExecutionConstraints:
    """Soft caps and hints for orchestration boundaries — **not enforced** in this package.

    Values are optional; ``None`` means « inherit from Settings / pack default ».
    Future work may wire these into middleware or pack wrappers.
    """

    max_query_chars: int | None = None
    stream_timeout_seconds: float | None = None
    budget_usd_ceiling: float | None = None


@dataclass(frozen=True, slots=True)
class GuardrailPolicy:
    """Rule sets (by name) screened on a pack's input and output.

    Applied only when ``GUARDRAILS_ENABLED`` is true; names must exist in the
    loaded rule sets or startup fails.
    """

    input_rule_sets: tuple[str, ...] = ()
    output_rule_sets: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PackPolicy:
    """Associates a registered ``pack_id`` with constraints and governance metadata.

    Compatible with the static ``PackRegistry``: ``pack_id`` must match a registered
    pack when policies are applied for real (validated elsewhere in a future sprint).

    When ``human_review_required`` is true, regulated packs must surface a mandatory
    ``disclaimer`` on every output (enforced in ``domain_packs.common.compliance``).
    """

    pack_id: str
    constraints: ExecutionConstraints = field(default_factory=ExecutionConstraints)
    labels: frozenset[str] = field(default_factory=frozenset)
    human_review_required: bool = False
    compliance_disclaimer: str | None = None
    guardrails: GuardrailPolicy = field(default_factory=GuardrailPolicy)
    extensions: dict[str, Any] = field(default_factory=dict)
