"""tests/test_dashboards.py — Grafana PromQL vs prometheus_client.REGISTRY."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import prometheus_client  # noqa: F401 — fail loudly if extra missing
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from agents.models import ResearchResult

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
        if metric.startswith("$") or metric.startswith("__"):
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


def test_tiny_budget_mock_client_is_shared_fixture() -> None:
    import tests.conftest as shared

    assert hasattr(shared, "tiny_budget_mock_client")


def _iter_exprs(node: Any) -> list[str]:
    exprs: list[str] = []
    if isinstance(node, dict):
        expr = node.get("expr")
        if isinstance(expr, str) and expr.strip():
            exprs.append(expr)
        for value in node.values():
            exprs.extend(_iter_exprs(value))
    elif isinstance(node, list):
        for item in node:
            exprs.extend(_iter_exprs(item))
    return exprs


def _dashboard_paths() -> list[Path]:
    return sorted(_DASHBOARDS_DIR.glob("*.json"))


def test_dashboards_dir_contains_exactly_three_json_files() -> None:
    names = {path.name for path in _dashboard_paths()}
    assert names == {
        "cost.json",
        "traffic-latency.json",
        "packs-versions.json",
    }


def test_each_dashboard_uses_datasource_variable() -> None:
    for path in _dashboard_paths():
        payload = json.loads(path.read_text(encoding="utf-8"))
        variables = payload["templating"]["list"]
        assert any(
            item.get("name") == "datasource"
            and item.get("type") == "datasource"
            and item.get("query") == "prometheus"
            for item in variables
        ), path
        blob = json.dumps(payload)
        assert "${DS_PROMETHEUS}" not in blob
        assert "__inputs" not in payload


def test_metrics_endpoint_mounted(test_client: TestClient) -> None:
    response = test_client.get("/metrics")
    assert response.status_code == 200


def _outcome_budget_exceeded_total() -> float:
    total = 0.0
    for metric in REGISTRY.collect():
        # prometheus_client 0.24 strips the _total suffix from the family name.
        if metric.name not in {"pack_runs_total", "pack_runs"}:
            continue
        for sample in metric.samples:
            if (
                sample.name == "pack_runs_total"
                and sample.labels.get("outcome") == "budget_exceeded"
            ):
                total += float(sample.value)
    return total


def test_dashboard_promql_matches_registry(
    test_client: TestClient,
    tiny_budget_mock_client: TestClient,
    mock_research_result: ResearchResult,
) -> None:
    from unittest.mock import patch

    from langgraph.checkpoint.memory import MemorySaver

    from core.config import get_settings
    from core.llm import get_llm
    from domain_packs.research.research_only.pack import ResearchOnlyPack

    before = _outcome_budget_exceeded_total()

    def _noop_init(self, **kwargs):  # type: ignore[no-untyped-def]
        pass

    # test_client patches get_shared_checkpointer to MagicMock; LangGraph
    # rejects that. Mock the pack only for the 200 seed so the overlapping
    # tiny-budget fixture cannot 402 this request.
    with (
        patch.object(ResearchOnlyPack, "__init__", _noop_init),
        patch.object(ResearchOnlyPack, "run", return_value=mock_research_result),
        patch.object(ResearchOnlyPack, "close", return_value=None),
    ):
        success = test_client.post(
            "/packs/research_only/run", json={"query": "dashboard seed"}
        )
    assert success.status_code == 200, success.text

    # Nested patches win over test_client's MagicMock LLM/checkpointer.
    with (
        patch(
            "api.state.get_shared_llm", return_value=get_llm(get_settings().llm_config)
        ),
        patch("api.state.get_shared_checkpointer", return_value=MemorySaver()),
    ):
        budget = tiny_budget_mock_client.post(
            "/packs/research_only/run", json={"query": "dashboard 402"}
        )
    assert budget.status_code == 402, budget.text
    after = _outcome_budget_exceeded_total()
    assert after > before

    collectors = REGISTRY._names_to_collectors  # noqa: SLF001
    for path in _dashboard_paths():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for expr in _iter_exprs(payload):
            for metric_name, labels in extract_promql_refs(expr):
                assert metric_name in collectors, (path, expr, metric_name)
                declared = set(collectors[metric_name]._labelnames)  # noqa: SLF001
                extra = set(labels) - IMPLIED_LABELS - declared
                assert not extra, (path, metric_name, extra, declared)


def test_http_histogram_never_groups_by_status_code() -> None:
    path = _DASHBOARDS_DIR / "traffic-latency.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for expr in _iter_exprs(payload):
        if "http_request_duration_seconds" not in expr:
            continue
        refs = extract_promql_refs(expr)
        for name, labels in refs:
            if name.startswith("http_request_duration_seconds"):
                assert "status_code" not in labels, expr
