"""core/guardrails/loader.py — Load built-in and override rule sets from JSON.

File format::

    {"rule_sets": {"<name>": {"rules": [
        {"id": "...", "pattern": "...", "action": "flag|escalate|block",
         "ignore_case": false, "multiline": false, "description": "..."}
    ]}}}

An override file replaces built-in sets by name and may add new ones; there is
no per-rule merge.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from core.guardrails.rules import (
    GuardrailAction,
    GuardrailConfigError,
    GuardrailRule,
    RuleSet,
)

BUILTIN_RULES_PATH = Path(__file__).with_name("builtin_rules.json")


class _RuleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    pattern: str
    action: GuardrailAction
    ignore_case: bool = False
    multiline: bool = False
    description: str = ""


class _RuleSetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules: list[_RuleSpec]


class _RuleFileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_sets: dict[str, _RuleSetSpec]


def _compile_rule(source: Path, set_name: str, spec: _RuleSpec) -> GuardrailRule:
    flags = (re.IGNORECASE if spec.ignore_case else 0) | (
        re.MULTILINE if spec.multiline else 0
    )
    try:
        pattern = re.compile(spec.pattern, flags)
    except re.error as exc:
        raise GuardrailConfigError(
            f"{source}: rule {spec.id!r} in set {set_name!r} has an invalid "
            f"pattern: {exc}"
        ) from exc
    return GuardrailRule(
        id=spec.id, pattern=pattern, action=spec.action, description=spec.description
    )


def _read_rule_file(path: Path) -> dict[str, RuleSet]:
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GuardrailConfigError(f"{path}: cannot read rule file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GuardrailConfigError(f"{path}: invalid JSON: {exc}") from exc

    try:
        spec = _RuleFileSpec.model_validate(raw)
    except ValidationError as exc:
        raise GuardrailConfigError(f"{path}: invalid rule file: {exc}") from exc

    rule_sets: dict[str, RuleSet] = {}
    for set_name, set_spec in spec.rule_sets.items():
        seen: set[str] = set()
        rules: list[GuardrailRule] = []
        for rule_spec in set_spec.rules:
            if rule_spec.id in seen:
                raise GuardrailConfigError(
                    f"{path}: duplicate rule id {rule_spec.id!r} in set {set_name!r}"
                )
            seen.add(rule_spec.id)
            rules.append(_compile_rule(path, set_name, rule_spec))
        rule_sets[set_name] = RuleSet(name=set_name, rules=tuple(rules))
    return rule_sets


@lru_cache(maxsize=1)
def _builtin_rule_sets() -> dict[str, RuleSet]:
    return _read_rule_file(BUILTIN_RULES_PATH)


def load_builtin_rule_sets() -> dict[str, RuleSet]:
    """Return the rule sets shipped with the package (parsed once)."""
    return dict(_builtin_rule_sets())


def load_rule_sets(override_path: Path | None) -> dict[str, RuleSet]:
    """Return built-in sets, with ``override_path`` sets replacing or adding by name.

    Raises:
        GuardrailConfigError: When the override file is missing, unreadable, or
            invalid (unknown keys, unknown action, duplicate id, bad regex).
    """
    rule_sets = load_builtin_rule_sets()
    if override_path is not None:
        rule_sets.update(_read_rule_file(override_path))
    return rule_sets
