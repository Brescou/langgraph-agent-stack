"""
tests/test_evals.py — Golden-dataset evaluation harness.

Covers the deterministic checks, the dataset loader, the runner (happy path,
expected-error cases, crashing packs), version comparison through the
PackRegistry, and the optional LLM judge. Everything is scripted — no
network, no real LLM.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from evals.checks import run_checks
from evals.judge import judge_case
from evals.models import EvalCase
from evals.runner import (
    ScriptedChatModel,
    compare_versions,
    dataset_path_for,
    list_builtin_datasets,
    load_dataset,
    run_pack_eval,
)
from pack_kernel.builtin_packs import register_builtin_packs

register_builtin_packs()


# ---------------------------------------------------------------------------
# Deterministic checks (unit)
# ---------------------------------------------------------------------------


class TestRunChecks:
    def test_required_fields(self) -> None:
        results = run_checks({"a": 1, "b": None}, {"required_fields": ["a", "b", "c"]})
        outcome = {r.name: r.passed for r in results}
        assert outcome == {
            "required_fields:a": True,
            "required_fields:b": False,
            "required_fields:c": False,
        }

    def test_contains_and_not_contains(self) -> None:
        output = {"summary": "quantum computing", "bullets": ["alpha", "beta"]}
        results = run_checks(
            output,
            {
                "contains": {"summary": "quantum", "bullets": "beta"},
                "not_contains": {"summary": "blockchain"},
            },
        )
        assert all(r.passed for r in results)

        failing = run_checks(output, {"contains": {"summary": "blockchain"}})
        assert not failing[0].passed
        assert "blockchain" in failing[0].detail

    def test_min_length(self) -> None:
        output = {"items": ["a", "b"], "scalar": 3}
        results = run_checks(
            output, {"min_length": {"items": 2, "scalar": 1, "missing": 1}}
        )
        outcome = {r.name: r.passed for r in results}
        assert outcome["min_length:items"] is True
        assert outcome["min_length:scalar"] is False  # len() on int → fail
        assert outcome["min_length:missing"] is False

    def test_numeric_range(self) -> None:
        results = run_checks(
            {"confidence": 0.7, "score": "high"},
            {"numeric_range": {"confidence": [0.5, 1.0], "score": [0, 1]}},
        )
        outcome = {r.name: r.passed for r in results}
        assert outcome["numeric_range:confidence"] is True
        assert outcome["numeric_range:score"] is False


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def test_load_builtin_datasets() -> None:
    available = list_builtin_datasets()
    assert {"summariser", "research_analysis", "talent_screening"} <= set(available)
    for pack_id in available:
        cases = load_dataset(dataset_path_for(pack_id))
        assert cases, f"dataset {pack_id} is empty"
        assert all(c.id for c in cases)


def test_load_dataset_rejects_bad_shape(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'cases'"):
        load_dataset(bad)


# ---------------------------------------------------------------------------
# Runner — built-in datasets must pass end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pack_id", ["summariser", "research_analysis", "talent_screening"]
)
def test_builtin_dataset_passes(pack_id: str) -> None:
    cases = load_dataset(dataset_path_for(pack_id))
    report = run_pack_eval(pack_id, cases)
    failed = [c for c in report.cases if not c.passed]
    assert not failed, f"failing cases: {[(c.case_id, c.error) for c in failed]}"
    assert report.pass_rate == 1.0
    assert report.version == "default"


def test_expected_error_case_fails_when_pack_succeeds() -> None:
    """expect_error must FAIL the case when the pack unexpectedly succeeds."""
    case = EvalCase(
        id="should-have-failed",
        input={"text": "hello world"},
        mock_responses=["- a bullet"],
        expect_error="integrity check",
    )
    report = run_pack_eval("summariser", [case])
    assert report.pass_rate == 0.0
    assert "got success" in (report.cases[0].error or "")


def test_crashing_case_is_reported_not_raised() -> None:
    """A pack error becomes a failed case with the error message, not a crash."""
    case = EvalCase(
        id="boom",
        input={"text": "hello"},
        mock_responses=[],  # empty scripted responses → falls back to llm=None
    )
    case = case.model_copy(update={"mock_responses": None})
    report = run_pack_eval("summariser", [case])
    assert report.pass_rate == 0.0
    assert "No LLM available" in (report.cases[0].error or "")


def test_runner_records_latency() -> None:
    cases = load_dataset(dataset_path_for("summariser"))
    report = run_pack_eval("summariser", cases)
    assert all(c.latency_seconds >= 0.0 for c in report.cases)
    assert report.mean_latency_seconds >= 0.0


# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------


def test_compare_versions_diff() -> None:
    """Register a v2 of summariser and compare it against the default."""
    from domain_packs.productivity.summariser.pack import SummariserPack
    from pack_kernel.registry import PackRegistry

    class SummariserV2(SummariserPack):
        version = "2.0-eval-test"

    PackRegistry.register(SummariserV2)
    try:
        cases = load_dataset(dataset_path_for("summariser"))
        comparison = compare_versions(
            "summariser",
            cases,
            baseline_version="1.0",
            candidate_version="2.0-eval-test",
        )
        diff = comparison.diff()
        assert diff["baseline_version"] == "1.0"
        assert diff["candidate_version"] == "2.0-eval-test"
        # Identical implementation → identical pass rate.
        assert diff["pass_rate_delta"] == 0.0
        assert comparison.baseline.pass_rate == 1.0
        assert comparison.candidate.pass_rate == 1.0
    finally:
        PackRegistry._reset()
        register_builtin_packs()


# ---------------------------------------------------------------------------
# LLM judge (optional)
# ---------------------------------------------------------------------------


def test_judge_parses_strict_verdict() -> None:
    judge_llm = MagicMock()
    judge_llm.invoke.return_value = MagicMock(
        content=json.dumps({"score": 0.9, "reasoning": "solid"})
    )
    verdict = judge_case(
        judge_llm,
        rubric="Bullets must be factual.",
        case_input={"text": "x"},
        output={"bullets": ["a"]},
    )
    assert verdict.score == 0.9
    assert verdict.reasoning == "solid"
    prompt = judge_llm.invoke.call_args[0][0]
    assert "UNTRUSTED" in prompt  # judged content is delimiter-wrapped


@pytest.mark.parametrize(
    "bad_response",
    [
        "not json at all",
        json.dumps({"score": 2.0, "reasoning": "out of range"}),
        json.dumps({"score": 0.5, "extra_field": True}),
    ],
)
def test_judge_rejects_invalid_verdicts(bad_response: str) -> None:
    judge_llm = MagicMock()
    judge_llm.invoke.return_value = MagicMock(content=bad_response)
    with pytest.raises(ValueError, match="Invalid judge verdict"):
        judge_case(
            judge_llm, rubric="r", case_input={"text": "x"}, output={"bullets": []}
        )


def test_judge_score_flows_into_case_result() -> None:
    judge_llm = MagicMock()
    judge_llm.invoke.return_value = MagicMock(
        content=json.dumps({"score": 0.75, "reasoning": "ok"})
    )
    case = EvalCase(
        id="judged",
        input={"text": "hello"},
        mock_responses=["- a bullet"],
        checks={"required_fields": ["bullets"]},
        judge="Bullets must be relevant.",
    )
    report = run_pack_eval("summariser", [case], judge_llm=judge_llm)
    assert report.cases[0].passed is True
    assert report.cases[0].judge_score == 0.75


def test_judge_skipped_without_judge_llm() -> None:
    case = EvalCase(
        id="unjudged",
        input={"text": "hello"},
        mock_responses=["- a bullet"],
        judge="Some rubric.",
    )
    report = run_pack_eval("summariser", [case])
    assert report.cases[0].judge_score is None


# ---------------------------------------------------------------------------
# ScriptedChatModel
# ---------------------------------------------------------------------------


def test_scripted_model_supports_bind_tools() -> None:
    model = ScriptedChatModel(responses=["x"])
    assert model.bind_tools([]) is model


def test_eval_cli_configures_logging_and_keeps_json_stdout(
    tmp_path, monkeypatch, capsys
) -> None:
    """The CLI configures stderr logging without contaminating JSON stdout."""
    import evals.__main__ as eval_cli

    configure_logging = MagicMock()
    monkeypatch.setattr(eval_cli, "configure_logging", configure_logging)

    dataset = tmp_path / "demo.yaml"
    dataset.write_text("cases: []\n", encoding="utf-8")
    monkeypatch.setattr(eval_cli, "list_builtin_datasets", lambda: ["demo"])
    monkeypatch.setattr(eval_cli, "dataset_path_for", lambda _: dataset)
    monkeypatch.setattr(eval_cli, "load_dataset", lambda _: [])

    report = MagicMock()
    report.summary.return_value = {
        "pack_id": "demo",
        "version": "default",
        "passed": 0,
        "total": 0,
        "pass_rate": 1.0,
        "mean_latency_seconds": 0.0,
        "total_cost_usd": 0.0,
    }
    report.cases = []
    report.pass_rate = 1.0
    monkeypatch.setattr(eval_cli, "run_pack_eval", lambda *args, **kwargs: report)

    assert eval_cli.main(["--pack", "demo", "--json"]) == 0
    configure_logging.assert_called_once_with(level="WARNING")

    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)[0]["pack_id"] == "demo"


# ---------------------------------------------------------------------------
# CI threshold gate
# ---------------------------------------------------------------------------


def test_load_thresholds_default_and_overrides(tmp_path) -> None:
    from evals.thresholds import load_thresholds

    path = tmp_path / "thresholds.yaml"
    path.write_text(
        "default_pass_rate: 0.9\npacks:\n  summariser: 1.0\n",
        encoding="utf-8",
    )
    config = load_thresholds(path)
    assert config.default_pass_rate == 0.9
    assert config.min_pass_rate("summariser") == 1.0
    assert config.min_pass_rate("research_analysis") == 0.9


def test_evaluate_thresholds_reports_failed_cases() -> None:
    from evals.models import CaseResult, EvalReport
    from evals.thresholds import (
        ThresholdConfig,
        evaluate_thresholds,
        format_threshold_failures,
    )

    report = EvalReport(
        pack_id="summariser",
        version="default",
        cases=[
            CaseResult(case_id="ok", passed=True),
            CaseResult(case_id="broken", passed=False),
        ],
    )
    config = ThresholdConfig(default_pass_rate=1.0, packs={})
    failures = evaluate_thresholds([report], config)
    assert len(failures) == 1
    assert failures[0].failed_case_ids == ["broken"]
    summary = format_threshold_failures(failures)
    assert "summariser" in summary
    assert "broken" in summary


def test_shipped_thresholds_cover_builtin_datasets() -> None:
    from evals.runner import list_builtin_datasets
    from evals.thresholds import DEFAULT_THRESHOLDS_PATH, load_thresholds

    config = load_thresholds(DEFAULT_THRESHOLDS_PATH)
    for pack_id in list_builtin_datasets():
        assert config.min_pass_rate(pack_id) == 1.0


def test_eval_cli_thresholds_fail_with_stderr_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    """Threshold failures keep JSON on stdout and name packs on stderr."""
    import evals.__main__ as eval_cli
    from evals.models import CaseResult, EvalReport

    monkeypatch.setattr(eval_cli, "configure_logging", MagicMock())
    monkeypatch.setattr(eval_cli, "register_builtin_packs", MagicMock())

    dataset = tmp_path / "demo.yaml"
    dataset.write_text("cases: []\n", encoding="utf-8")
    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text("default_pass_rate: 1.0\npacks: {}\n", encoding="utf-8")

    monkeypatch.setattr(eval_cli, "list_builtin_datasets", lambda: ["demo"])
    monkeypatch.setattr(eval_cli, "dataset_path_for", lambda _: dataset)
    monkeypatch.setattr(eval_cli, "load_dataset", lambda _: [])

    report = EvalReport(
        pack_id="demo",
        version="default",
        cases=[
            CaseResult(case_id="a", passed=True),
            CaseResult(case_id="b", passed=False),
        ],
    )
    monkeypatch.setattr(eval_cli, "run_pack_eval", lambda *args, **kwargs: report)

    assert (
        eval_cli.main(["--pack", "demo", "--json", "--thresholds", str(thresholds)])
        == 1
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)[0]["pack_id"] == "demo"
    assert "EVAL THRESHOLD FAILURES" in captured.err
    assert "demo" in captured.err
    assert "b" in captured.err
