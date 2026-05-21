# langgraph-agent-stack

> Production-grade **template** for multi-agent LangGraph systems — hardened FastAPI API, domain packs, Helm, Terraform stubs, CI, and observability. Multi-tenant identity is **your** layer ([Security guide](docs/security.md)).

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)](https://github.com/astral-sh/uv)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/brescou/langgraph-agent-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/brescou/langgraph-agent-stack/actions/workflows/ci.yml)

## What is this?

A deployable starting point for ML / data engineers who want a **real** agent stack—not a notebook demo. The default pipeline is Research → Analysis (`ResearchAgent` + `AnalystAgent`), exposed over FastAPI with SSE streaming, session history, cost tracking, and pack-based routing.

**Included:** Docker, Helm chart, Terraform entry points (GKE / EKS / AKS), rate limiting, input validation, structured logging, Prometheus metrics, CI security scans, GHCR images with SBOM + Cosign on `main`.

**Not included:** OAuth2/OIDC, per-tenant API keys, or billing. The built-in `API_KEY` is a single shared Bearer secret for internal / single-tenant use.

## Quick start

```bash
git clone https://github.com/brescou/langgraph-agent-stack.git
cd langgraph-agent-stack
uv sync --extra anthropic
cp .env.example .env   # set ANTHROPIC_API_KEY
uv run uvicorn api.main:app --reload
```

```bash
# Legacy default pipeline (research_analysis pack)
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest advances in quantum computing?"}'

# Pack registry
curl http://localhost:8000/packs
curl -X POST http://localhost:8000/packs/meeting_prep/run \
  -H "Content-Type: application/json" \
  -d '{"company": "Acme", "person": "Jane", "meeting_goal": "discovery"}'
```

Interactive API docs: `http://localhost:8000/docs` (disabled when `ENVIRONMENT=production`).

## Architecture

```
Client → FastAPI (auth · rate limit · validation)
              → PackRegistry / control_plane policies
              → domain_packs/* (LangGraph workflows)
              → agents/* (reusable agent nodes)
              → core/* (LLM · memory · security · cost · observability)
              → connectors/* (optional retrieval)
```

| Layer | Path | Role |
|-------|------|------|
| HTTP | `api/` | FastAPI app, middlewares, legacy + pack routes |
| Kernel | `pack_kernel/` | `BaseDomainPack`, `PackRegistry`, versioning, traffic split |
| Workflows | `domain_packs/` | Built-in and vertical packs — see [domain_packs/README.md](domain_packs/README.md) |
| Agents | `agents/` | Reusable LangGraph agents (`ResearchAgent`, `AnalystAgent`, …) |
| Policies | `control_plane/` | Per-pack limits (query size, budget, stream timeout) — [control_plane/README.md](control_plane/README.md) |
| Connectors | `connectors/` + `core/connectors.py` | Retrieval adapters — [connectors/README.md](connectors/README.md) |
| Foundation | `core/` | Config, LLM factory, memory, security, cost, observability |
| Ops | `infra/` | Dockerfile, Compose, Helm, Terraform |

`core/graph.py` is a **compatibility shim** (`MultiAgentGraph` → `ResearchAnalysisPack`). New orchestration belongs in a domain pack.

## Domain packs

15 built-in packs registered in `pack_kernel/builtin_packs.py`:

| Category | Examples |
|----------|----------|
| Core pipelines | `research_analysis`, `research_only`, `analysis_only`, `summariser` |
| Vertical workflows | `meeting_prep`, `rfp_assistant`, `support_triage`, `executive_brief`, `contract_reviewer`, `financial_memo` |
| HR (`domain_packs/rh/`) | `talent_screening`, `job_description_writer`, `hr_policy_qa` |

Each pack gets typed `POST /packs/{pack_id}/run` and `/run/stream` when schemas are declared. Versioning, traffic weights, and sticky sessions: `GET /packs`, `GET /packs/{id}/versions`, headers `X-Pack-Version` / `X-Pack-Version-Used`.

Full catalogue and authoring guide: **[domain_packs/README.md](domain_packs/README.md)**.

## API (summary)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/packs` | List registered packs and metadata |
| `POST` | `/packs/{pack_id}/run` | Run a pack (typed body per pack schema) |
| `POST` | `/packs/{pack_id}/run/stream` | SSE stream for a pack |
| `POST` | `/run`, `/run/stream` | Legacy routes → `DEFAULT_PACK_ID` (`research_analysis`) |
| `POST` | `/research` | Research phase only |
| `GET` | `/health`, `/ready` | Probes |
| `GET` | `/sessions/{id}/history` | Session run history |
| `GET` | `/metrics` | Prometheus (with `observability` extra) |

Responses include `cost_usd` when cost tracking is active; HTTP **402** on budget exceed. See `/docs` for request/response schemas.

## LLM providers

Set `LLM_PROVIDER` and install the matching extra. Details and gateway overrides: `.env.example`.

| Provider | Value | Extra | Key env vars |
|----------|-------|-------|--------------|
| Anthropic | `anthropic` | `--extra anthropic` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `--extra openai` | `OPENAI_API_KEY` |
| Google | `google` | `--extra google` | `GOOGLE_API_KEY` |
| Bedrock | `bedrock` | `--extra bedrock` | `AWS_REGION`, `BEDROCK_MODEL` |
| Azure OpenAI | `azure` | `--extra openai` | `AZURE_OPENAI_*` |
| Ollama | `ollama` | `--extra ollama` | `OLLAMA_BASE_URL` |
| Mock (CI/dev) | `mock` | *(none)* | — |

## Run & deploy

**Docker Compose**

```bash
docker compose -f infra/docker-compose.yml up
docker compose -f infra/docker-compose.yml --profile redis up   # Redis memory backend
```

**Helm**

```bash
helm install langgraph ./infra/helm/langgraph-agent-stack \
  -f infra/helm/langgraph-agent-stack/values.prod.yaml \
  --namespace langgraph-agents --create-namespace
```

Production: set `secrets.existingSecret` (External Secrets Operator), `config.environment=production`, `networkPolicy.enabled=true`. Autoscaling defaults to **KEDA** on `active_pipelines` (not CPU). See chart `values.yaml` / `values.prod.yaml`.

**Terraform** — entry points under `infra/terraform/{gke,eks,aks}/` (no shared root module). Configure a remote backend before production apply. GKE module expects [External Secrets Operator](docs/security.md#3-secret-management) installed before `ClusterSecretStore` resources.

**Infra CI locally:** `make infra-check` (template Checkov profile). Before production hardening: `make infra-check-prod` ([checklist](docs/security.md#before-going-to-production-checkov)).

## Security

Rate limiting (memory or Redis), request body cap, prompt-injection / SSRF input validation, security headers, optional `API_KEY` Bearer auth, graceful shutdown drain. Full model, env vars, K8s hardening, and scanning pipeline: **[docs/security.md](docs/security.md)**.

## Development

```bash
make help          # all targets
make check         # ruff + pyright (CI lint)
make test          # pytest (mocked; use -m integration for testcontainers)
make infra-check   # helm lint + kubeconform + checkov
```

Contributor workflow, pre-commit, and PR expectations: **[CONTRIBUTING.md](CONTRIBUTING.md)**.

**LangGraph patterns** (standalone scripts, not served by the API): [examples/README.md](examples/README.md).

## Project structure

```
langgraph-agent-stack/
├── api/                 # FastAPI application
├── pack_kernel/         # Pack contract + PackRegistry
├── domain_packs/        # Workflows (core + vertical + rh/)
├── agents/              # Reusable LangGraph agents
├── connectors/          # Retrieval connector implementations
├── control_plane/       # Pack policies
├── core/                # Config, LLM, memory, security, cost, observability
├── infra/               # Dockerfile, compose, helm/, terraform/
├── examples/            # LangGraph pattern demos
├── tests/
├── docs/                # security.md, …
└── scripts/             # infra-devsecops.sh, …
```

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/security.md](docs/security.md) | Auth, secrets, K8s hardening, CI scans, supply chain, Checkov prod gate |
| [domain_packs/README.md](domain_packs/README.md) | Pack catalogue and authoring |
| [connectors/README.md](connectors/README.md) | Connector contract and wiring |
| [control_plane/README.md](control_plane/README.md) | Policy registry |
| [examples/README.md](examples/README.md) | Sequential, parallel, supervisor, human-in-loop |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup, tests, PR process |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

## License

MIT © [brescou](https://github.com/brescou)
