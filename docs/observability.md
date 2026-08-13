# Observability — Prometheus, Grafana, and `/metrics`

Local Compose can scrape the API and provision three Grafana dashboards against the metrics that already exist (`pack_run_cost_usd_total`, HTTP counters/histograms, `pack_runs_total`, `pack_run_duration_seconds`, `active_pipelines`). This is a **local profile**, not a Helm add-on: Grafana is not in the chart. Operators with an existing Grafana import the JSON files below.

Helm `ServiceMonitor` / KEDA still expect `GET /metrics` on the running image. The published GHCR image returns **404** on `/metrics` until [#132](https://github.com/Brescou/langgraph-agent-stack/issues/132) (GHCR `OBS_EXTRAS=observability` build-arg). Do not treat this Compose profile as a substitute for that image.

---

## Prerequisites

Copy `.env.example` to `.env` at the **repository root** before starting Compose. The `app` service sets `env_file: ../.env`; without that file `docker compose --profile observability up` fails.

```bash
cp .env.example .env
# set LLM_PROVIDER=mock (or a real provider + API key)
```

The Compose `app` image is built with `--build-arg OBS_EXTRAS=observability` so `prometheus-client` is installed and `/metrics` is mounted. That arg is a **build** arg only — putting `OBS_EXTRAS` in `environment:` is a no-op.

---

## Local stack

From `infra/`:

```bash
cd infra && docker compose --profile observability up --build
```

| Service | URL | Notes |
|---------|-----|--------|
| API | `http://localhost:8000` | `/health`, `/metrics` (200 with this image) |
| Prometheus | `http://localhost:9090` | scrapes `app:8000/metrics` on `langgraph-net` |
| Grafana | `http://localhost:3000` | login **`admin` / `admin`**. Sign-up is disabled; anonymous Admin is not enabled. |

Dashboards are file-provisioned from `infra/grafana/dashboards/` into Grafana folder **LangGraph**. Prometheus is the default datasource.

---

## Import into an existing Grafana

If you already run Grafana (cluster sidecar, Grafana Cloud, …), do **not** wait for Compose provisioning. Import the three JSON files:

- `infra/grafana/dashboards/cost.json` — **LangGraph / Cost**
- `infra/grafana/dashboards/traffic-latency.json` — **LangGraph / Traffic & latency**
- `infra/grafana/dashboards/packs-versions.json` — **LangGraph / Packs & versions**

Each dashboard declares a `datasource` variable (type `datasource`, query `prometheus`). After import, pick your Prometheus datasource from that dropdown. Panels use `${datasource}` — there is no `${DS_PROMETHEUS}` / `__inputs` block.

Grafana in Helm is out of scope. Keep the JSON in source control and import it into whatever Grafana you already operate.

---

## Panel inventory

### Cost (`cost.json`)

| Panel | Metric |
|-------|--------|
| LLM cost by pack / version / model | `pack_run_cost_usd_total` by `pack_id`, `version`, `model` |
| LLM cost by pack / version / provider | `llm_cost_usd_total` by `pack_id`, `version`, `provider` |
| Budget rejections by pack / version | `pack_runs_total{outcome="budget_exceeded"}` by `pack_id`, `version` |

A mock run at $0 still creates cost series (`on_llm_end` increments unconditionally). The budget panel uses `outcome="budget_exceeded"` only — not `http_requests_total{status_code="402"}`.

### Traffic & latency (`traffic-latency.json`)

| Panel | Metric |
|-------|--------|
| HTTP requests by status | `http_requests_total` with `sum by (status_code)` (and `path` / `method` as needed) |
| HTTP latency p95 by route template | `histogram_quantile` on `http_request_duration_seconds_bucket`. Labels: **`path` only** (route template; all packs share a series) |
| Pack run volume | `pack_runs_total` by `pack_id`, `version`, `outcome` |
| Pack run latency p95 | `histogram_quantile` on `pack_run_duration_seconds_bucket` |

**Do not mix these dimensions:**

- HTTP `status_code` is used **only** on `http_requests_total`. Never `sum by (status_code)` on `http_request_duration_seconds` (that histogram is labelled `path` only).
- Pack latency is **only** on `pack_run_duration_seconds`. The HTTP histogram collapses every pack onto the route template, so HTTP p95 is **by route template, not pack**.

### Packs & versions (`packs-versions.json`)

| Panel | Metric |
|-------|--------|
| Run volume by pack version | `pack_runs_total` by `pack_id`, `version`, `outcome` (canary split) |
| Pack latency p95 by version | `histogram_quantile` on `pack_run_duration_seconds_bucket` |
| `active_pipelines` (KEDA scaling signal) | `active_pipelines` as a **time series** (gauge returns to 0 after runs) |

---

## Seed metrics (visual authoring)

Pytest does **not** start Compose or Grafana. After the stack is up, seed the scrape target with mock POSTs:

```bash
# .env: LLM_PROVIDER=mock
curl -X POST http://localhost:8000/packs/research_only/run \
  -H "Content-Type: application/json" \
  -d '{"query": "What is LangGraph?"}'
```

Repeat a few times so rate/increase windows have samples. Then open Grafana at `http://localhost:3000` and confirm Cost / Traffic / Packs panels populate.

To exercise the **budget rejections** panel, force a **402** (mock is $0 by default, so you need a tiny budget **and** a cost table that prices the mock model high — see `LLM_COST_TABLE_PATH` in `.env.example` and `PACK_DEFAULT_BUDGET_USD`):

```bash
# with PACK_DEFAULT_BUDGET_USD small enough that one mock call exceeds it
curl -X POST http://localhost:8000/packs/research_only/run \
  -H "Content-Type: application/json" \
  -d '{"query": "budget seed"}'
# expect HTTP 402
```

---

## GHCR, Helm, and KEDA (#132)

| Surface | `/metrics` today |
|---------|------------------|
| Local Compose (`OBS_EXTRAS=observability` build-arg) | **200** |
| `uv sync --extra observability` + uvicorn | **200** |
| Published **GHCR** image | **404** until [#132](https://github.com/Brescou/langgraph-agent-stack/issues/132) |

Helm `ServiceMonitor` and KEDA (`active_pipelines`) need an image that actually serves `/metrics`. Until #132 lands, do **not** expect cluster scrape or HPA-from-Prometheus to work against GHCR. This PR does not add Grafana to Helm or GHCR `build-args`.
