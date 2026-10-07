"""core/guardrails/rules.py — Value types for the deterministic guardrails engine."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

GuardrailAction = Literal["flag", "escalate", "block"]
GuardrailPhase = Literal["input", "output"]

ACTION_SEVERITY: dict[GuardrailAction, int] = {"flag": 1, "escalate": 2, "block": 3}


class GuardrailConfigError(ValueError):
    """Raised when a rule file or a pack policy references invalid guardrails."""


@dataclass(frozen=True, slots=True)
class GuardrailRule:
    """One compiled pattern and the action to take when it matches."""

    id: str
    pattern: re.Pattern[str]
    action: GuardrailAction
    description: str = ""


@dataclass(frozen=True, slots=True)
class RuleSet:
    """A named, ordered group of rules that a pack policy can subscribe to."""

    name: str
    rules: tuple[GuardrailRule, ...]


@dataclass(frozen=True, slots=True)
class Finding:
    """A rule match located in a field.

    Deliberately carries no matched text: a PII rule must not copy the value it
    detects into audit logs.
    """

    rule_set: str
    rule_id: str
    action: GuardrailAction
    field_path: str


@dataclass(frozen=True, slots=True)
class Verdict:
    """All findings for one screened payload."""

    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def action(self) -> GuardrailAction | None:
        """Most severe action among the findings, or ``None`` when clean."""
        if not self.findings:
            return None
        return max(
            (finding.action for finding in self.findings),
            key=ACTION_SEVERITY.__getitem__,
        )

    @property
    def rule_refs(self) -> list[str]:
        """``"rule_set/rule_id"`` for every matching rule, deduplicated in order."""
        return list(
            dict.fromkeys(
                f"{finding.rule_set}/{finding.rule_id}" for finding in self.findings
            )
        )
