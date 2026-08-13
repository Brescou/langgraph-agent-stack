"""tests/test_dashboards.py — Grafana PromQL vs prometheus_client.REGISTRY."""

from __future__ import annotations

import json  # noqa: F401 — scaffolding for later tasks
import re
from pathlib import Path
from typing import Any  # noqa: F401 — scaffolding for later tasks

import prometheus_client  # noqa: F401 — fail loudly if extra missing
from prometheus_client import REGISTRY  # noqa: F401 — scaffolding for later tasks

IMPLIED_LABELS = frozenset({"le", "quantile"})
_DASHBOARDS_DIR = Path("infra/grafana/dashboards")

_METRIC_SELECTOR = re.compile(
    r"(?<![A-Za-z0-9_:])([a-zA-Z_:][a-zA-Z0-9_:]*)\s*(?:\{([^}]*)\})?"
)
_BY_WITHOUT = re.compile(r"\b(?:by|without)\s*\(([^)]*)\)", re.IGNORECASE)
_LABEL_KEY = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=")
_PROMQL_KEYWORDS = frozenset(
    {
        "by",
        "without",
        "on",
        "ignoring",
        "group_left",
        "group_right",
        "and",
        "or",
        "unless",
        "sum",
        "min",
        "max",
        "avg",
        "count",
        "stddev",
        "stdvar",
        "rate",
        "irate",
        "increase",
        "histogram_quantile",
        "label_replace",
        "label_join",
        "vector",
        "time",
        "bool",
    }
)


def extract_promql_refs(expr: str) -> list[tuple[str, frozenset[str]]]:
    """Return ``(metric_name, label_names)`` for each metric mention in ``expr``.

    Label names come from ``{key=...}`` selectors plus ``by (...)`` /
    ``without (...)`` clauses on the same expression. Function names are
    skipped. ``le`` / ``quantile`` may appear here; callers whitelist them
    before comparing to ``collector._labelnames``.
    """
    clause_labels: set[str] = set()
    for match in _BY_WITHOUT.finditer(expr):
        for part in match.group(1).split(","):
            name = part.strip()
            if name:
                clause_labels.add(name)

    refs: list[tuple[str, frozenset[str]]] = []
    for match in _METRIC_SELECTOR.finditer(expr):
        metric = match.group(1)
        if metric in _PROMQL_KEYWORDS:
            continue
        selector = match.group(2)
        if metric in clause_labels and selector is None:
            continue
        selector = selector or ""
        labels = set(_LABEL_KEY.findall(selector)) | clause_labels
        refs.append((metric, frozenset(labels)))
    return refs


def test_extract_selector_labels() -> None:
    refs = extract_promql_refs('pack_runs_total{outcome="budget_exceeded"}')
    assert refs == [
        ("pack_runs_total", frozenset({"outcome"})),
    ]


def test_extract_by_clause_on_histogram_bucket() -> None:
    expr = (
        "histogram_quantile(0.95, "
        "sum by (le, pack_id, version) "
        "(rate(pack_run_duration_seconds_bucket[5m])))"
    )
    refs = extract_promql_refs(expr)
    names = {name for name, _ in refs}
    labels: set[str] = set()
    for _, lab in refs:
        labels |= set(lab)
    assert "pack_run_duration_seconds_bucket" in names
    assert {"le", "pack_id", "version"} <= labels
    assert "le" in IMPLIED_LABELS


def test_extract_without_clause() -> None:
    refs = extract_promql_refs("sum without (instance) (http_requests_total)")
    labels: set[str] = set()
    for _, lab in refs:
        labels |= set(lab)
    assert any(name == "http_requests_total" for name, _ in refs)
    assert "instance" in labels


def test_extract_status_code_on_counter_not_confused_with_function() -> None:
    refs = extract_promql_refs("sum by (status_code) (rate(http_requests_total[5m]))")
    assert any(name == "http_requests_total" for name, _ in refs)
    labels: set[str] = set()
    for _, lab in refs:
        labels |= set(lab)
    assert "status_code" in labels
    assert all(name != "rate" for name, _ in refs)
    assert all(name != "sum" for name, _ in refs)


def test_by_without_labels_not_treated_as_metrics() -> None:
    """Aggregation-clause label names must not appear as metric names."""
    histogram_expr = (
        "histogram_quantile(0.95, "
        "sum by (le, pack_id, version) "
        "(rate(pack_run_duration_seconds_bucket[5m])))"
    )
    without_expr = "sum without (instance) (http_requests_total)"
    status_expr = "sum by (status_code) (rate(http_requests_total[5m]))"

    label_only_names = {"instance", "status_code", "le", "pack_id", "version"}
    for expr in (histogram_expr, without_expr, status_expr):
        names = {name for name, _ in extract_promql_refs(expr)}
        assert label_only_names.isdisjoint(names)
    assert "pack_run_duration_seconds_bucket" in {
        name for name, _ in extract_promql_refs(histogram_expr)
    }
    assert "http_requests_total" in {
        name for name, _ in extract_promql_refs(without_expr)
    }
    assert "http_requests_total" in {
        name for name, _ in extract_promql_refs(status_expr)
    }
