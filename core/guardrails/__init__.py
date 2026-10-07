"""core/guardrails — Deterministic, pattern-based input/output screening.

Pure engine: no FastAPI, no packs. The API adapter lives in ``api/guardrails.py``
and per-pack opt-in in ``control_plane.policies.GuardrailPolicy``.
"""

from core.guardrails.engine import evaluate, iter_string_fields
from core.guardrails.loader import (
    BUILTIN_RULES_PATH,
    load_builtin_rule_sets,
    load_rule_sets,
)
from core.guardrails.rules import (
    ACTION_SEVERITY,
    Finding,
    GuardrailAction,
    GuardrailConfigError,
    GuardrailPhase,
    GuardrailRule,
    RuleSet,
    Verdict,
)

__all__ = [
    "ACTION_SEVERITY",
    "BUILTIN_RULES_PATH",
    "Finding",
    "GuardrailAction",
    "GuardrailConfigError",
    "GuardrailPhase",
    "GuardrailRule",
    "RuleSet",
    "Verdict",
    "evaluate",
    "iter_string_fields",
    "load_builtin_rule_sets",
    "load_rule_sets",
]
