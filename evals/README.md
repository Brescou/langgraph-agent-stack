# evals — Golden-dataset pack evaluation harness

Deterministic, mock-mode evaluations for built-in packs. Zero token cost, no
API key, no network flakiness when `LLM_PROVIDER=mock` (and
`SEARCH_PROVIDER=mock` for research packs).

## What this catches (and what it does not)

Mock-mode evals catch **structural** regressions: broken schemas, parsing
failures, graph wiring mistakes, missing fields, output-guard rejections wired
incorrectly. They do **not** measure semantic prompt quality — that needs a
real provider and stays a deliberate, manual, off-CI activity (`--compare`
against a baseline exists for that).

## Local usage

```bash
# Human-readable report for every built-in dataset
LLM_PROVIDER=mock SEARCH_PROVIDER=mock uv run python -m evals --all

# Same as CI (JSON on stdout + pass_rate floors)
LLM_PROVIDER=mock SEARCH_PROVIDER=mock \
  uv run python -m evals --all --json --thresholds

# One pack
uv run python -m evals --pack summariser
```

Or: `make eval` (human report) / `make eval-ci` (gate mode).

## CI gate

Every PR runs the mock eval suite as a required-style job in `.github/workflows/ci.yml`:

```bash
LLM_PROVIDER=mock SEARCH_PROVIDER=mock \
  uv run python -m evals --all --json --thresholds evals/thresholds.yaml
```

- **Stdout** stays machine-parseable JSON (log-clean; see #89).
- **Stderr** gets a readable per-pack summary when a pack drops below its floor.
- Floors live in [`thresholds.yaml`](thresholds.yaml): `default_pass_rate` plus
  optional per-pack overrides under `packs:`.

A PR that breaks an eval case fails this job with the failing `case_id`s named
in the threshold summary.
