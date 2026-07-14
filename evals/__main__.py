"""evals/__main__.py — CLI entry point for the pack evaluation harness.

Usage::

    python -m evals --list
    python -m evals --pack summariser
    python -m evals --pack summariser --version 1.0 --compare 2.0
    python -m evals --all --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.observability import configure_logging
from evals.models import EvalReport
from evals.runner import (
    compare_versions,
    dataset_path_for,
    list_builtin_datasets,
    load_dataset,
    run_pack_eval,
)
from pack_kernel.builtin_packs import register_builtin_packs


def _print_report(report: EvalReport) -> None:
    agg = report.summary()
    print(f"\n=== {agg['pack_id']} (version: {agg['version']}) ===")
    for case in report.cases:
        marker = "PASS" if case.passed else "FAIL"
        line = f"  [{marker}] {case.case_id}  ({case.latency_seconds:.2f}s)"
        if case.judge_score is not None:
            line += f"  judge={case.judge_score:.2f}"
        print(line)
        if not case.passed:
            for check in case.checks:
                if not check.passed:
                    print(f"         check {check.name}: {check.detail}")
            if case.error:
                print(f"         error: {case.error}")
    print(
        f"  -> {agg['passed']}/{agg['total']} passed "
        f"(pass_rate={agg['pass_rate']:.0%}, "
        f"mean_latency={agg['mean_latency_seconds']:.2f}s, "
        f"cost=${agg['total_cost_usd']:.4f})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals", description="Golden-dataset pack evaluation harness."
    )
    parser.add_argument("--pack", help="Pack id to evaluate.")
    parser.add_argument(
        "--all", action="store_true", help="Run every built-in dataset."
    )
    parser.add_argument(
        "--version", help="Specific pack version (default: registry routing)."
    )
    parser.add_argument(
        "--compare",
        metavar="VERSION",
        help="Candidate version to compare against --version.",
    )
    parser.add_argument(
        "--dataset", help="Dataset path (default: evals/datasets/<pack>.yaml)."
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON."
    )
    parser.add_argument(
        "--list", action="store_true", help="List built-in datasets and exit."
    )
    args = parser.parse_args(argv)

    configure_logging(level="WARNING")
    register_builtin_packs()

    if args.list:
        for pack_id in list_builtin_datasets():
            print(pack_id)
        return 0

    pack_ids = list_builtin_datasets() if args.all else [args.pack] if args.pack else []
    if not pack_ids:
        parser.error("provide --pack <id>, --all, or --list")

    exit_code = 0
    json_out: list[dict] = []
    for pack_id in pack_ids:
        dataset = Path(args.dataset) if args.dataset else dataset_path_for(pack_id)
        if not dataset.exists():
            print(
                f"error: no dataset for pack {pack_id!r} at {dataset}", file=sys.stderr
            )
            return 2
        cases = load_dataset(dataset)

        if args.compare:
            comparison = compare_versions(
                pack_id,
                cases,
                baseline_version=args.version,
                candidate_version=args.compare,
            )
            if args.json:
                json_out.append(comparison.diff())
            else:
                _print_report(comparison.baseline)
                _print_report(comparison.candidate)
                print(f"\n  diff: {json.dumps(comparison.diff(), indent=2)}")
            if comparison.candidate.pass_rate < comparison.baseline.pass_rate:
                exit_code = 1
        else:
            report = run_pack_eval(pack_id, cases, version=args.version)
            if args.json:
                json_out.append(
                    {
                        **report.summary(),
                        "cases": [c.model_dump() for c in report.cases],
                    }
                )
            else:
                _print_report(report)
            if report.pass_rate < 1.0:
                exit_code = 1

    if args.json:
        print(json.dumps(json_out, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
