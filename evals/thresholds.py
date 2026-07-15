"""evals/thresholds.py — Per-pack pass_rate floors for the CI mock eval gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from evals.models import EvalReport

#: Default path shipped with the repo (used by CI and ``make eval``).
DEFAULT_THRESHOLDS_PATH = Path(__file__).parent / "thresholds.yaml"


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    """Minimum pass rates for the mock-mode CI gate."""

    default_pass_rate: float
    packs: dict[str, float]

    def min_pass_rate(self, pack_id: str) -> float:
        """Return the floor for *pack_id* (pack override or default)."""
        return self.packs.get(pack_id, self.default_pass_rate)


@dataclass(frozen=True, slots=True)
class ThresholdFailure:
    """One pack that fell below its configured pass_rate floor."""

    pack_id: str
    pass_rate: float
    threshold: float
    failed_case_ids: list[str]


def load_thresholds(path: str | Path) -> ThresholdConfig:
    """Load ``evals/thresholds.yaml`` (or an equivalent file).

    Expected layout::

        default_pass_rate: 1.0
        packs:
          summariser: 1.0   # optional per-pack override
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Thresholds file {path} must be a YAML mapping.")

    default = raw.get("default_pass_rate", 1.0)
    if not isinstance(default, (int, float)) or not 0.0 <= float(default) <= 1.0:
        raise ValueError(
            f"default_pass_rate must be a number in [0, 1], got {default!r}"
        )

    packs_raw = raw.get("packs") or {}
    if not isinstance(packs_raw, dict):
        raise ValueError(f"'packs' must be a mapping, got {type(packs_raw).__name__}")

    packs: dict[str, float] = {}
    for pack_id, rate in packs_raw.items():
        if not isinstance(pack_id, str) or not pack_id:
            raise ValueError(f"Invalid pack id in thresholds: {pack_id!r}")
        if not isinstance(rate, (int, float)) or not 0.0 <= float(rate) <= 1.0:
            raise ValueError(
                f"pass_rate for pack {pack_id!r} must be in [0, 1], got {rate!r}"
            )
        packs[pack_id] = float(rate)

    return ThresholdConfig(default_pass_rate=float(default), packs=packs)


def evaluate_thresholds(
    reports: list[EvalReport],
    config: ThresholdConfig,
) -> list[ThresholdFailure]:
    """Return packs whose ``pass_rate`` is strictly below their floor."""
    failures: list[ThresholdFailure] = []
    for report in reports:
        floor = config.min_pass_rate(report.pack_id)
        if report.pass_rate < floor:
            failures.append(
                ThresholdFailure(
                    pack_id=report.pack_id,
                    pass_rate=report.pass_rate,
                    threshold=floor,
                    failed_case_ids=[c.case_id for c in report.cases if not c.passed],
                )
            )
    return failures


def format_threshold_failures(failures: list[ThresholdFailure]) -> str:
    """Human-readable multi-line summary for CI logs (stderr)."""
    lines = ["EVAL THRESHOLD FAILURES:"]
    for failure in failures:
        cases = ", ".join(failure.failed_case_ids) or "(none listed)"
        lines.append(
            f"  {failure.pack_id}: pass_rate={failure.pass_rate:.2%} "
            f"< threshold={failure.threshold:.2%} "
            f"(failed cases: {cases})"
        )
    return "\n".join(lines)


def thresholds_to_dict(config: ThresholdConfig) -> dict[str, Any]:
    """JSON-friendly view (tests / debugging)."""
    return {
        "default_pass_rate": config.default_pass_rate,
        "packs": dict(config.packs),
    }
