"""core/guardrails/engine.py — Evaluate rule sets against strings in a payload."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from core.guardrails.rules import Finding, RuleSet, Verdict


def iter_string_fields(data: Any, prefix: str = "$") -> Iterator[tuple[str, str]]:
    """Yield ``(field_path, text)`` for every string nested in dicts and lists.

    Paths are JSONPath-like: ``$.summary``, ``$.attendees[2]``.
    """
    if isinstance(data, str):
        yield prefix, data
    elif isinstance(data, dict):
        for key, value in data.items():
            yield from iter_string_fields(value, f"{prefix}.{key}")
    elif isinstance(data, list | tuple):
        for index, item in enumerate(data):
            yield from iter_string_fields(item, f"{prefix}[{index}]")


def evaluate(data: Any, rule_sets: Sequence[RuleSet], *, prefix: str = "$") -> Verdict:
    """Return every rule match in ``data``; at most one finding per rule per field."""
    findings: list[Finding] = []
    for field_path, text in iter_string_fields(data, prefix):
        if not text:
            continue
        for rule_set in rule_sets:
            for rule in rule_set.rules:
                if rule.pattern.search(text) is not None:
                    findings.append(
                        Finding(
                            rule_set=rule_set.name,
                            rule_id=rule.id,
                            action=rule.action,
                            field_path=field_path,
                        )
                    )
    return Verdict(findings=tuple(findings))
