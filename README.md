# langgraph-agent-stack

> Production-ready template for deploying multi-agent LangGraph systems on Kubernetes — skip two weeks of boilerplate and ship your first agent pipeline today.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)](https://github.com/astral-sh/uv)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/your-org/langgraph-agent-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/langgraph-agent-stack/actions/workflows/ci.yml)

## What is this?

Setting up a production-grade multi-agent system from scratch means wiring together an LLM SDK, a graph orchestrator, a persistent memory backend, a hardened API layer, containerization, and Kubernetes manifests — before you write a single line of domain logic. This template does all of that for you. It is aimed at ML and Data Engineers who want a correct, deployable starting point rather than a toy notebook.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  FastAPI  (rate limiting · security headers · CORS)  │
│                                                       │
│   POST /run ──────────────────────────────────────┐  │
│   POST /research ─────────────────────────────┐   │  │
│   GET  /health                                │   │  │
└───────────────────────────────────────────────┼───┼──┘
                                                │   │
                    ┌───────────────────────────┘   │
                    ▼                               ▼
         ┌─────────────────┐             ┌─────────────────┐
         │  ResearchAgent  │             │  ResearchAgent  │
         │  (LangGraph)    │             │  only           │
         └────────┬────────┘             └─────────────────┘
                  │ ResearchResult
                  ▼
         ┌─────────────────┐
         │  AnalystAgent   │
         │  (LangGraph)    │
         └────────┬────────┘
                  │ AnalysisReport
                  ▼
         ┌─────────────────┐
         │  Memory Backend │
         │  SQLite / Redis │
         └─────────────────┘
```

**Key components**

| Component | Path | Responsibility |
|-----------|------|----------------|
| `ResearchAgent` | `agents/researcher.py` | Expands queries into sub-queries, retrieves information snippets, validates quality |
| `AnalystAgent` | `agents/analyst.py` | Consumes research findings, extracts insights, identifies patterns, produces a structured report |
| `MultiAgentGraph` | `core/graph.py` | LangGraph orchestrator that sequences the two agents with shared state |
| `core/memory.py` | `core/memory.py` | Pluggable checkpoint backend (SQLite for development, Redis for production) |
| `core/security.py` | `core/security.py` | Input validation, per-IP rate limiting, API key format checks |
| `api/main.py` | `api/main.py` | FastAPI application with lifespan management and thread pool offloading |

## Quick Start

**Prerequisites**

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker (optional, for containerized runs)
- An [Anthropic API key](https://console.anthropic.com)

**1. Clone and install dependencies**

```bash
git clone https://github.com/your-org/langgraph-agent-stack.git
cd langgraph-agent-stack
uv sync
```

**2. Configure environment**

```bash
cp .env.example .env
```

Open `.env` and set your provider and API key:

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

All other values have working defaults for local development.

**3. Start the API server**

```bash
uv run uvicorn api.main:app --reload
```

The server starts on `http://localhost:8000`. The interactive API docs are at `http://localhost:8000/docs`.

**4. Send your first request**

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest advances in quantum computing?"}'
```

You will receive a structured `AnalysisReport` with an executive summary, key insights, identified patterns, and a confidence score.

## LLM Providers

The template supports any LangChain-compatible LLM provider. Set `LLM_PROVIDER` in your `.env` and install the matching extra.

| Provider | `LLM_PROVIDER` | Install | Key variable |
|----------|---------------|---------|--------------|
| Anthropic (Claude) | `anthropic` | `uv sync --extra anthropic` | `ANTHROPIC_API_KEY` |
| OpenAI (GPT) | `openai` | `uv sync --extra openai` | `OPENAI_API_KEY` |
| Google (Gemini) | `google` | `uv sync --extra google` | `GOOGLE_API_KEY` |
| AWS Bedrock | `bedrock` | `uv sync --extra bedrock` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |
| Azure OpenAI | `azure` | `uv sync --extra openai` | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` |
| Ollama (local) | `ollama` | `uv sync --extra ollama` | None — runs locally |

### Switching providers

**OpenAI:**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```
`uv sync --extra openai`

**Ollama (no API key required):**
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```
`uv sync --extra ollama`

> **Note:** The agents never import provider-specific code directly. The `LLMFactory` in `core/llm.py` resolves the provider at startup, so switching is a one-line `.env` change.

## Running with Docker

Build and start the full stack (SQLite backend):

```bash
docker compose -f infra/docker-compose.yml up
```

Start with Redis as the memory backend:

```bash
docker compose -f infra/docker-compose.yml --profile redis up
```

The compose file reads your `.env` file automatically. Make sure it exists before running.

The application is available at `http://localhost:8000` after the health check passes (about 15 seconds on first startup).

## Kubernetes Deployment

The chart lives in `infra/helm/langgraph-agent-stack/`. It requires Helm 3 and a running Kubernetes cluster.

### Install

```bash
helm install langgraph ./infra/helm/langgraph-agent-stack \
  --namespace langgraph-agents \
  --create-namespace \
  --set secrets.anthropicApiKey=$ANTHROPIC_API_KEY
```

### Custom values

```bash
# Development
helm install langgraph ./infra/helm/langgraph-agent-stack \
  -f infra/helm/langgraph-agent-stack/values.dev.yaml \
  --set secrets.anthropicApiKey=$ANTHROPIC_API_KEY

# Production (with External Secrets Operator — no key in CLI)
helm install langgraph ./infra/helm/langgraph-agent-stack \
  -f infra/helm/langgraph-agent-stack/values.prod.yaml
```

### Upgrade / Uninstall

```bash
helm upgrade langgraph ./infra/helm/langgraph-agent-stack
helm uninstall langgraph -n langgraph-agents
```

### Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `ingress.enabled` | `false` | Create an Ingress resource |
| `autoscaling.enabled` | `false` | Enable HorizontalPodAutoscaler |
| `secrets.existingSecret` | `""` | Use an existing Secret (External Secrets Operator) |
| `serviceAccount.create` | `true` | Create a dedicated ServiceAccount |

### Production secrets

In production, set `secrets.existingSecret` to point to a secret managed by the [External Secrets Operator](https://external-secrets.io) or Sealed Secrets instead of passing keys via `--set`.

```yaml
# values.prod.yaml (already configured)
secrets:
  existingSecret: langgraph-secrets  # chart will not create a Secret
```

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/run` | Run the full Research + Analysis pipeline. Returns a structured `AnalysisReport`. |
| `POST` | `/research` | Run the Research phase only. Returns a `ResearchResult` without downstream analysis. |
| `GET` | `/health` | Liveness and readiness probe. Returns service status, version, uptime, and environment. |

**POST /run**

```json
// Request
{ "query": "string (max 2000 characters)" }

// Response
{
  "run_id": "uuid",
  "summary": "string",
  "key_insights": ["string"],
  "patterns": ["string"],
  "implications": ["string"],
  "confidence": 0.87,
  "query": "string",
  "timestamp": "ISO 8601"
}
```

**POST /research**

```json
// Request
{ "query": "string (max 2000 characters)" }

// Response
{
  "run_id": "uuid",
  "summary": "string",
  "findings": ["string"],
  "sources": ["string"],
  "confidence": 0.91,
  "query": "string",
  "timestamp": "ISO 8601"
}
```

**GET /health**

```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_seconds": 142.3,
  "environment": "development"
}
```

Rate limit: 60 requests per minute per IP. Exceeding the limit returns `429 Too Many Requests` with a `Retry-After` header. The `/health` endpoint is exempt from rate limiting so Kubernetes probes are never blocked.

## Configuration

All configuration is loaded from environment variables. Copy `.env.example` to `.env` to get started.

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LLM_PROVIDER` | LLM provider to use | `anthropic` | Required |
| `ANTHROPIC_API_KEY` | Anthropic API key | — | Required if `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL` | Claude model name | `claude-3-5-sonnet-20241022` | Optional |
| `MAX_TOKENS` | Maximum tokens per LLM call | `4096` | No |
| `MEMORY_BACKEND` | Checkpoint backend: `sqlite` or `redis` | `sqlite` | No |
| `SQLITE_PATH` | Path to the SQLite database file | `./data/agent_memory.db` | No |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` | Only when `MEMORY_BACKEND=redis` |
| `API_HOST` | Host the server binds to | `0.0.0.0` | No |
| `API_PORT` | TCP port the server listens on | `8000` | No |
| `LOG_LEVEL` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` | No |
| `ENVIRONMENT` | Deployment environment label: `development`, `staging`, `production` | `development` | No |

See `.env.example` for all provider-specific variables (OpenAI, Google, Bedrock, Azure, Ollama).

## Development

**Run the test suite**

```bash
uv run pytest
```

87 tests across API endpoints, memory backends, security primitives, and tools. Async tests run automatically via `pytest-asyncio`.

**Lint and format**

```bash
uv run ruff check .
uv run black --check .
```

Both checks run automatically in CI on every push and pull request.

**Project structure**

```
langgraph-agent-stack/
├── agents/
│   ├── base_agent.py       # Abstract base class, error types, retry logic
│   ├── researcher.py       # ResearchAgent implementation
│   └── analyst.py          # AnalystAgent implementation
├── core/
│   ├── config.py           # Pydantic-settings configuration model
│   ├── graph.py            # MultiAgentGraph — LangGraph orchestrator
│   ├── memory.py           # SQLite / Redis checkpoint backend
│   ├── security.py         # InputValidator, RateLimiter, key validation
│   └── tools.py            # LangChain tools available to agents
├── api/
│   ├── main.py             # FastAPI application and endpoints
│   └── models.py           # Pydantic request/response models
├── infra/
│   ├── Dockerfile          # Multi-stage build (builder + runtime, non-root)
│   ├── docker-compose.yml  # Local stack with optional Redis profile
│   └── k8s/
│       ├── configmap.yaml
│       ├── deployment.yaml # 2 replicas, liveness/readiness probes, resource limits
│       ├── secret.yaml
│       └── service.yaml
├── tests/
│   ├── test_api.py
│   ├── test_memory.py
│   ├── test_security.py
│   └── test_tools.py
├── .github/workflows/
│   ├── ci.yml              # ruff + black + pytest on push/PR
│   └── security.yml        # Security scanning
├── pyproject.toml
└── .env.example
```

## Extending the Template

### Add a new agent

1. Create `agents/my_agent.py` inheriting from `BaseAgent` in `agents/base_agent.py`. Implement the `run` and `run_structured` methods.
2. Add your agent as a node in `core/graph.py` and connect its edges in the LangGraph state graph.
3. Expose it via a new endpoint in `api/main.py` following the pattern used by `/research`.

### Change the LLM provider

Set `LLM_PROVIDER` in `.env` to one of: `anthropic`, `openai`, `google`, `bedrock`, `azure`, `ollama`.
Install the matching extra: `uv sync --extra <provider>`.
No code changes required — the factory in `core/llm.py` handles instantiation.

### Enable Redis for production

1. Set `MEMORY_BACKEND=redis` and `REDIS_URL=redis://your-host:6379/0` in your environment.
2. Install the optional Redis extras: `uv sync --extra redis`.
3. When deploying with Docker Compose, start with `--profile redis` to bring up the Redis service alongside the application.

## License

MIT © [your-org](https://github.com/your-org)
