# langgraph-agent-stack

> Production-ready template for deploying multi-agent LangGraph systems on Kubernetes — skip two weeks of boilerplate and ship your first agent pipeline today.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)](https://github.com/astral-sh/uv)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/brescou/langgraph-agent-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/brescou/langgraph-agent-stack/actions/workflows/ci.yml)

## What is this?

Setting up a production-grade multi-agent system from scratch means wiring together an LLM SDK, a graph orchestrator, a persistent memory backend, a hardened API layer, containerization, and Kubernetes manifests — before you write a single line of domain logic. This template does all of that for you. It is aimed at ML and Data Engineers who want a correct, deployable starting point rather than a toy notebook.

The source code implements a Research + Analysis pipeline where two LangGraph agents (`ResearchAgent` and `AnalystAgent`) are orchestrated by a shared graph and served over a FastAPI REST API with SSE streaming.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  FastAPI  (rate limiting · auth · security headers)  │
│                                                       │
│   POST /run ──────────────────────────────────────┐  │
│   POST /run/stream ────────────────────────────┐  │  │
│   POST /research ─────────────────────────┐   │  │  │
│   GET  /health                            │   │  │  │
│   GET  /ready                             │   │  │  │
│   GET  /sessions/{id}/history             │   │  │  │
└───────────────────────────────────────────┼───┼──┼──┘
                                            │   │  │
                    ┌───────────────────────┘   │  │
                    ▼                           │  │
         ┌─────────────────┐                   │  │
         │  ResearchAgent  │◄──────────────────┘  │
         │  (LangGraph)    │◄─────────────────────┘
         └────────┬────────┘
                  │ ResearchResult
                  ▼
         ┌─────────────────┐
         │  AnalystAgent   │
         │  (LangGraph)    │
         └────────┬────────┘
                  │ AnalysisReport
                  ▼
         ┌─────────────────────────┐
         │  Memory Backend         │
         │  SQLite / Redis / PG    │
         └─────────────────────────┘
```

**Key components**

| Component | Path | Responsibility |
|-----------|------|----------------|
| `ResearchAgent` | `agents/researcher.py` | Expands queries into sub-queries, retrieves information snippets, validates quality |
| `AnalystAgent` | `agents/analyst.py` | Consumes research findings, extracts insights, identifies patterns, produces a structured report |
| `MultiAgentGraph` | `core/graph.py` | LangGraph orchestrator that sequences the two agents with shared state |
| `ConversationMemory` | `core/memory.py` | Pluggable checkpoint backend (SQLite, Redis, or PostgreSQL) |
| `core/security.py` | `core/security.py` | Input validation, per-IP rate limiting, log sanitization |
| `api/main.py` | `api/main.py` | FastAPI application with lifespan management and thread pool offloading |

## Quick Start

**Prerequisites**

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker (optional, for containerized runs)
- An API key for your chosen LLM provider

**1. Clone and install dependencies**

```bash
git clone https://github.com/brescou/langgraph-agent-stack.git
cd langgraph-agent-stack
uv sync --extra anthropic
```

**2. Configure environment**

```bash
cp .env.example .env
```

Minimum required variables for the default Anthropic provider:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

All other variables have working defaults for local development.

**3. Start the API server**

```bash
uv run uvicorn api.main:app --reload
```

The server starts on `http://localhost:8000`. Interactive API docs are at `http://localhost:8000/docs`.

**4. Send your first request**

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest advances in quantum computing?"}'
```

You will receive a structured `AnalysisReport` with an executive summary, key insights, identified patterns, and a confidence score.

## LLM Providers

Set `LLM_PROVIDER` in your `.env` and install the matching extra. Provider-specific packages are imported lazily — only the package you install is required.

| Provider | `LLM_PROVIDER` value | Install extra | Required env vars |
|----------|----------------------|---------------|-------------------|
| Anthropic (Claude) | `anthropic` | `uv sync --extra anthropic` | `ANTHROPIC_API_KEY` |
| OpenAI (GPT) | `openai` | `uv sync --extra openai` | `OPENAI_API_KEY` |
| Google (Gemini) | `google` | `uv sync --extra google` | `GOOGLE_API_KEY` |
| AWS Bedrock | `bedrock` | `uv sync --extra bedrock` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| Azure OpenAI | `azure` | `uv sync --extra openai` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| Ollama (local) | `ollama` | `uv sync --extra ollama` | None — runs locally |

### Switching providers

**OpenAI:**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```
```bash
uv sync --extra openai
```

**Ollama (no API key required):**
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```
```bash
uv sync --extra ollama
```

Agents never import provider-specific code directly. `core/llm.py` resolves the provider at startup via `get_llm()`, so switching is a one-line `.env` change with no code modifications.

## Running with Docker

Build and start the full stack (SQLite backend):

```bash
docker compose -f infra/docker-compose.yml up
```

Start with Redis as the memory backend:

```bash
docker compose -f infra/docker-compose.yml --profile redis up
```

The Dockerfile supports three independent build args for extras:

| Arg | Default | Description |
|-----|---------|-------------|
| `LLM_EXTRAS` | `anthropic` | LLM provider (`openai`, `google`, `bedrock`, `ollama`, `all-providers`) |
| `INFRA_EXTRAS` | *(empty)* | Storage backend (`redis`, `postgres`) |
| `OBS_EXTRAS` | *(empty)* | Observability stack (`observability` — includes Prometheus + OTel) |

```bash
docker build \
  --build-arg LLM_EXTRAS=anthropic \
  --build-arg INFRA_EXTRAS=redis \
  --build-arg OBS_EXTRAS=observability \
  -f infra/Dockerfile .
```

The compose file reads your `.env` file automatically. The application is available at `http://localhost:8000` after the health check passes (about 15 seconds on first startup).

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

In production, set `secrets.existingSecret` to point to a secret managed by the [External Secrets Operator](https://external-secrets.io) or Sealed Secrets instead of passing keys via `--set`.

## Infrastructure as Code

Terraform modules for GKE (Autopilot) and EKS in `infra/terraform/`.

> **Important:** By default, Terraform state is stored locally. For production or team use, configure a remote backend (GCS or S3) in `infra/terraform/versions.tf` before running `terraform apply`. See the comments in that file for instructions.

```bash
cd infra/terraform
terraform init

# GKE (default)
terraform apply -var-file=environments/dev.tfvars \
  -var="project_id=my-gcp-project" \
  -var="anthropic_api_key=$ANTHROPIC_API_KEY"

# EKS
terraform apply -var-file=environments/dev.tfvars \
  -var="cloud_provider=eks" \
  -var="anthropic_api_key=$ANTHROPIC_API_KEY"
```

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/run` | Run the full Research + Analysis pipeline. Returns a structured `AnalysisReport`. |
| `POST` | `/run/stream` | Same pipeline streamed as Server-Sent Events. |
| `POST` | `/research` | Run the Research phase only. Returns a `ResearchResult` without downstream analysis. |
| `GET` | `/health` | Liveness probe. Returns service status, version, uptime, and environment. |
| `GET` | `/ready` | Readiness probe. Returns 200 when LLM and checkpointer are initialised; 503 otherwise. |
| `GET` | `/sessions/{session_id}/history` | Retrieve all run records for a session, ordered newest-first. |

---

**POST /run**

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"query": "Latest advances in quantum computing"}'
```

```json
{
  "query": "Latest advances in quantum computing",
  "executive_summary": "string",
  "key_insights": ["string"],
  "patterns": ["string"],
  "implications": ["string"],
  "confidence": 0.87,
  "research_summary": "string",
  "metadata": {},
  "session_id": "uuid"
}
```

---

**POST /run/stream**

```bash
curl -X POST http://localhost:8000/run/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"query": "Latest AI advances"}'
```

Events are delivered as Server-Sent Events:

```
data: {"type": "status", "message": "Starting research phase..."}
data: {"type": "agent_switch", "from": "researcher", "to": "analyst"}
data: {"type": "status", "message": "Starting analysis phase..."}
data: {"type": "done", "session_id": "...", "confidence": 0.87, ...}
```

The stream enforces a wall-clock timeout controlled by `STREAM_TIMEOUT_SECONDS` (default 120s). On timeout, a final `{"type": "error", "message": "Stream timed out after 120s"}` event is emitted.

---

**POST /research**

```json
// Request
{ "query": "string (max 2000 characters)", "session_id": "optional string (max 128 chars, alphanumeric/dash/underscore)" }

// Response
{
  "query": "string",
  "summary": "string",
  "findings": ["string"],
  "sources": ["string"],
  "confidence": 0.91,
  "metadata": {},
  "session_id": "uuid"
}
```

---

**GET /health**

```json
{
  "status": "ok",
  "version": "0.3.0",
  "uptime_seconds": 142.3,
  "environment": "development",
  "components": {
    "llm": { "status": "ok", "detail": "anthropic (initialised)" },
    "memory": { "status": "ok", "detail": ":memory:" },
    "checkpointer": { "status": "ok", "detail": "sqlite" }
  }
}
```

This endpoint is exempt from rate limiting so Kubernetes probes are never blocked.

> **Note on LLM health:** The `llm` component reports whether the LLM client is initialised, not whether the upstream provider is reachable. A real provider call on every probe would add latency, consume tokens, and risk hitting rate limits. An expired API key or provider outage will surface on the first pipeline request, not on the liveness probe. The `memory` and `checkpointer` components perform actual connectivity checks (SQLite `SELECT 1`, Redis `PING`, Postgres `SELECT 1`).

---

**GET /sessions/{session_id}/history**

```json
{
  "session_id": "uuid",
  "total": 3,
  "entries": [
    {
      "run_id": "uuid",
      "query": "string",
      "result_summary": "string",
      "created_at": "ISO 8601",
      "metadata": {}
    }
  ]
}
```

## Security

**Bearer token authentication**

Set `API_KEY` in your environment to enable authentication. When set, all requests to pipeline endpoints must include an `Authorization: Bearer <token>` header. The `/health`, `/ready`, `/docs`, `/redoc`, and `/openapi.json` endpoints are always exempt.

```bash
curl -X POST http://localhost:8000/run \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "..."}'
```

Leave `API_KEY` unset to disable authentication (suitable for internal deployments behind a gateway).

**CORS**

The template ships with `allow_origins=["*"]` for maximum compatibility during development. **For production deployments**, restrict this to your trusted frontend origins:

```python
allow_origins=["https://your-frontend.example.com"]
```

Never combine `allow_origins=["*"]` with `allow_credentials=True`.

**Rate limiting**

60 requests per minute per IP, enforced by a per-IP sliding-window limiter. Exceeding the limit returns `429 Too Many Requests` with a `Retry-After` header.

The rate limiter backend is controlled by `RATE_LIMIT_BACKEND`:

| Value | Scope | Use case |
|-------|-------|----------|
| `memory` (default) | Per-process | Local development, single-replica deployments |
| `redis` | Shared across replicas | Production with multiple pods behind a load balancer |

When using `redis`, the limiter uses a Lua-scripted sliding window on Redis sorted sets, providing a consistent view of request counts across all replicas. Set `REDIS_URL` in your environment to point to a shared Redis instance.

**Fail-open behaviour:** If the Redis instance becomes unreachable, the rate limiter fails open — requests are allowed through and a warning is logged (`Redis rate limiter unreachable — failing open`). This is intentional: rate limiting is a non-critical function and should never block legitimate traffic during a Redis outage. Operators should monitor this log message and the rate of `429` responses to detect degraded rate limiting.

**Input validation**

All queries are validated by `InputValidator` before reaching agent code. Queries exceeding 2000 characters or matching dangerous patterns are rejected with `400 Bad Request`.

**Security headers**

Every response includes `X-Content-Type-Options`, `X-Frame-Options`, `Content-Security-Policy: default-src 'self'`, `Referrer-Policy`, and `Cache-Control: no-store`. The `Server` header is stripped to avoid advertising the runtime stack.

**Log sanitization**

`sanitize_log_data()` in `core/security.py` recursively redacts sensitive keys (passwords, tokens, URLs with credentials) before structured log output.

## Configuration

All configuration is loaded from environment variables. Copy `.env.example` to `.env` to get started.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | LLM provider: `anthropic`, `openai`, `google`, `bedrock`, `azure`, `ollama` |
| `MEMORY_BACKEND` | `sqlite` | Checkpoint backend: `sqlite`, `redis`, `postgres` |
| `ENVIRONMENT` | `development` | Deployment label: `development`, `staging`, `production` |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `API_HOST` | `0.0.0.0` | Host the FastAPI server binds to |
| `API_PORT` | `8000` | TCP port the FastAPI server listens on |
| `API_KEY` | — | Bearer token for API auth. Leave unset to disable auth. |

### LLM providers

| Variable | Default | Used when |
|----------|---------|-----------|
| `ANTHROPIC_API_KEY` | — | `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | `LLM_PROVIDER=anthropic` |
| `OPENAI_API_KEY` | — | `LLM_PROVIDER=openai` |
| `OPENAI_MODEL` | `gpt-4o` | `LLM_PROVIDER=openai` |
| `GOOGLE_API_KEY` | — | `LLM_PROVIDER=google` |
| `GOOGLE_MODEL` | `gemini-1.5-pro` | `LLM_PROVIDER=google` |
| `AWS_ACCESS_KEY_ID` | — | `LLM_PROVIDER=bedrock` |
| `AWS_SECRET_ACCESS_KEY` | — | `LLM_PROVIDER=bedrock` |
| `AWS_REGION` | `us-east-1` | `LLM_PROVIDER=bedrock` |
| `BEDROCK_MODEL` | `anthropic.claude-3-5-sonnet-20241022-v2:0` | `LLM_PROVIDER=bedrock` |
| `AZURE_OPENAI_API_KEY` | — | `LLM_PROVIDER=azure` |
| `AZURE_OPENAI_ENDPOINT` | — | `LLM_PROVIDER=azure` |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o` | `LLM_PROVIDER=azure` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | `LLM_PROVIDER=ollama` |
| `OLLAMA_MODEL` | `llama3.2` | `LLM_PROVIDER=ollama` |
| `MAX_TOKENS` | `4096` | All providers |

### Memory and storage

| Variable | Default | Description |
|----------|---------|-------------|
| `SQLITE_PATH` | `./data/agent_memory.db` | SQLite database file path |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL (required when `MEMORY_BACKEND=redis`) |
| `POSTGRES_URL` | — | PostgreSQL DSN (required when `MEMORY_BACKEND=postgres`) |
| `RAG_ENABLED` | `false` | Enable vector store infrastructure (not yet wired to agents — planned for v0.3.0) |

### Agent behaviour

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_RESEARCH_ITERATIONS` | `3` | Safety cap on research loop iterations (max 10) |
| `MAX_STEP_COUNT` | `20` | Hard limit on total graph steps per run (max 100) |
| `STREAM_TIMEOUT_SECONDS` | `120` | Wall-clock timeout for SSE streaming runs |
| `THREAD_POOL_MAX_WORKERS` | `4` | Size of the thread pool for blocking agent calls (1–64) |

### Observability

The stack includes three complementary observability pillars:

| Feature | Package | Install |
|---------|---------|---------|
| **Structured JSON logging** | `python-json-logger` | `uv sync --extra observability` |
| **Prometheus metrics** | `prometheus-client` | `uv sync --extra observability` |
| **OpenTelemetry tracing** | `opentelemetry-sdk` | `uv sync --extra observability` |

**Logging** — Every request is logged as a single JSON line containing
`timestamp`, `level`, `logger`, `message`, `request_id`, and all `extra`
fields. Sensitive keys (passwords, tokens, API keys) are automatically
redacted by a `SanitizingFilter` before serialisation.

**Prometheus metrics** — When `prometheus-client` is installed, the
`/metrics` endpoint exposes:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `http_requests_total` | Counter | method, path, status_code | Total HTTP requests |
| `http_request_duration_seconds` | Histogram | path | Request latency (buckets: 0.1s–30s) |
| `llm_requests_total` | Counter | provider, status | LLM API calls per attempt (success / retryable_error / fatal_error) |
| `active_pipelines` | Gauge | — | Currently running agent pipelines |
| `server_shutting_down` | Gauge | — | 1 during graceful drain, 0 otherwise |
| `requests_rejected_during_shutdown_total` | Counter | — | Requests rejected with 503 during shutdown |

To include metrics in your Docker image, build with:

```bash
docker build --build-arg LLM_EXTRAS=anthropic --build-arg OBS_EXTRAS=observability .
```

**OpenTelemetry** — Set `OTEL_ENABLED=true` and configure `OTEL_EXPORTER_OTLP_ENDPOINT`
to ship traces to Jaeger, Tempo, or any OTLP-compatible backend.

## Development

### Running tests

```bash
# Full test suite
uv run pytest

# With coverage report
uv run pytest --cov=. --cov-report=term-missing

# Single file
uv run pytest tests/test_api.py -v
```

The default test suite is fully mocked — no external API calls or databases are needed. `conftest.py` patches LLM calls and memory backends automatically. Real backend integration tests (Postgres, Redis via testcontainers) live in `tests/test_integration_real.py` and are marked with `@pytest.mark.integration`:

```bash
# Skip integration tests (fast CI path)
uv run pytest -m "not integration"

# Run only integration tests (requires Docker)
uv run pytest -m integration -v
```

### Test files

| File | What it covers |
|------|----------------|
| `tests/conftest.py` | Shared fixtures, mocks, and test settings used by all test files |
| `tests/test_api.py` | FastAPI endpoint tests, auth middleware, SSE streaming, session history |
| `tests/test_agents.py` | `ResearchAgent` and `AnalystAgent` unit tests |
| `tests/test_graph.py` | `MultiAgentGraph` orchestration and state transition tests |
| `tests/test_llm.py` | `get_llm()` provider switching tests covering all 6 providers |
| `tests/test_memory.py` | `ConversationMemory` backend tests (SQLite, Redis, PostgreSQL) |
| `tests/test_security.py` | `InputValidator` and `RateLimiter` unit tests |
| `tests/test_tools.py` | LangChain agent tool tests |
| `tests/test_vectorstore.py` | Vector store integration and RAG tests |
| `tests/test_config.py` | `get_settings()` caching, `llm_config` property, cross-field validators |
| `tests/test_observability.py` | Structured logging and OpenTelemetry tracing tests |

### Lint and format

```bash
uv run ruff check .          # lint
uv run ruff check . --fix    # lint + auto-fix
uv run black .               # format
uv run black --check .       # format check (CI mode)
```

Both checks run automatically in CI on every push and pull request via `.github/workflows/ci.yml`. Security scanning (gitleaks, bandit, dependency audit) runs via `.github/workflows/security.yml`.

### Makefile shortcuts

Run `make help` to list all targets. Key commands:

```bash
# Development
make install       # uv sync --all-extras
make run           # Start API server with hot reload
make run-ollama    # Start API server using Ollama provider

# Quality
make test          # Run test suite with verbose output
make test-cov      # Run tests with coverage report
make lint          # Check code style with ruff
make format        # Format source code with black
make check         # Lint + format check without modifying (CI mode)

# Docker
make docker-build  # Build the Docker image
make docker-run    # Start all services with docker compose
make docker-redis  # Start all services including Redis profile
make docker-down   # Stop and remove all containers

# Helm
make helm-lint     # Lint the Helm chart
make helm-dev      # Deploy to dev environment
make helm-prod     # Deploy to production environment
make helm-dry-run  # Simulate a Helm install
make helm-uninstall # Uninstall the Helm release

# Terraform
make tf-init       # Initialize Terraform working directory
make tf-plan       # Generate Terraform execution plan
make tf-apply      # Apply the Terraform plan

# Utilities
make clean         # Remove build artifacts and caches
```

### Project structure

```
langgraph-agent-stack/
├── agents/
│   ├── base_agent.py       # Abstract BaseAgent, error types, retry logic
│   ├── models.py           # ResearchResult and AnalysisReport dataclasses
│   ├── researcher.py       # ResearchAgent — query expansion, retrieval, quality checks
│   └── analyst.py          # AnalystAgent — insight extraction, pattern detection, reporting
├── core/
│   ├── config.py           # Pydantic-settings Settings model; use get_settings() not Settings()
│   ├── graph.py            # MultiAgentGraph — LangGraph orchestrator and state definition
│   ├── llm.py              # get_llm() — provider-agnostic LLM instantiation
│   ├── memory.py           # ConversationMemory — SQLite / Redis / PostgreSQL checkpointing
│   ├── observability.py    # Structured JSON logging and optional OpenTelemetry tracing
│   ├── security.py         # InputValidator, RateLimiter, sanitize_log_data
│   ├── tools.py            # LangChain tools shared across agents
│   └── vectorstore.py      # Optional RAG vector store integration
├── api/
│   ├── main.py             # FastAPI app, lifespan, endpoints
│   └── models.py           # Pydantic request/response models
├── infra/
│   ├── Dockerfile          # Multi-stage build (builder + non-root runtime)
│   ├── docker-compose.yml  # Local stack with optional Redis profile
│   ├── helm/               # Helm chart for Kubernetes deployment
│   └── terraform/          # Terraform modules for GKE Autopilot and EKS
├── examples/
│   ├── sequential/         # Linear Research → Analysis pipeline
│   ├── parallel/           # Three analysts running simultaneously
│   ├── supervisor/         # Dynamic routing to specialist agents
│   └── human_in_loop/      # Pause graph execution for human approval
├── tests/                  # Unit tests (mocked) + real backend integration tests
├── .github/workflows/
│   ├── ci.yml              # ruff + black + pytest on push/PR
│   └── security.yml        # gitleaks, bandit, dependency audit
├── .dockerignore           # Excludes .env, .git, tests, docs from Docker context
├── pyproject.toml
└── .env.example
```

## Extending the Template

### Add a new agent

1. Create `agents/my_agent.py` inheriting from `BaseAgent` in `agents/base_agent.py`. Implement `build_graph()` and `run()`. Add `run_structured()` if you need typed output (it is not abstract).
2. Add your agent as a node in `core/graph.py` and connect its edges in the LangGraph state graph.
3. Expose it via a new endpoint in `api/main.py` following the pattern used by `/research`.

### Change the LLM provider

Set `LLM_PROVIDER` in `.env` to one of: `anthropic`, `openai`, `google`, `bedrock`, `azure`, `ollama`.
Install the matching extra: `uv sync --extra <provider>`.
No code changes required — `core/llm.py` handles instantiation.

### Enable Redis for production

1. Set `MEMORY_BACKEND=redis` and `REDIS_URL=redis://your-host:6379/0` in your environment.
2. Install the Redis extras: `uv sync --extra redis`.
3. When deploying with Docker Compose, start with `--profile redis` to bring up the Redis service.

### Enable PostgreSQL for production

1. Set `MEMORY_BACKEND=postgres` and `POSTGRES_URL=postgresql+psycopg://user:pass@host:5432/dbname`.
2. Install the PostgreSQL extras: `uv sync --extra postgres`.

### Enable RAG (experimental)

> **Note:** Setting `RAG_ENABLED=true` currently provisions the vector store
> infrastructure (ChromaDB or PGVector) but does **not** wire it into agent
> pipelines. Full RAG integration is planned for **v0.3.0**.

1. Set `RAG_ENABLED=true` in your environment.
2. Install the RAG extras: `uv sync --extra rag`.

## Examples

Four ready-to-run multi-agent patterns in `examples/`:

| Pattern | Description | Run |
|---------|-------------|-----|
| Sequential | Linear Research → Analysis pipeline | `uv run python examples/sequential/graph.py` |
| Parallel | Three analysts run simultaneously | `uv run python examples/parallel/graph.py` |
| Supervisor | Dynamic routing to specialist agents | `uv run python examples/supervisor/graph.py` |
| Human-in-loop | Pause for human approval | `uv run python examples/human_in_loop/graph.py` |

See `examples/README.md` for architecture details.

## License

MIT © [brescou](https://github.com/brescou)
