# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `/ready` readiness probe endpoint — separate from `/health` liveness probe
- Fail-fast checkpointer in production (`_fallback_or_raise` in `core/memory.py`)
- OTel `insecure=True` conditional on `OTEL_EXPORTER_OTLP_INSECURE` or localhost endpoint
- LLM retry with backoff activated in all 6 agent graph nodes
- `Content-Security-Policy: default-src 'self'` header (replaces deprecated `X-XSS-Protection`)
- Tests: SSE done event validation, timeout error event, shutdown 503 guard, rate limiting on /research, auth exempt paths, Redis/Postgres checkpointer mocks, populated session history
- Tests: all 5 security headers asserted (CSP, Referrer-Policy, Cache-Control, X-Frame-Options, X-Content-Type-Options)

### Changed
- `_node_validate` defaults to `is_sufficient=False` on error (fail-close instead of fail-open)
- `/research` endpoint uses `session_id` as `thread_id` (unified with `/run/stream`)
- `GET /sessions/{id}/history` now runs `list_runs_by_session` via `_run_in_executor`
- `ResearchRequest.session_id` gains `max_length=128` + alphanumeric pattern validation
- `MultiAgentGraph` executor uses `get_settings().thread_pool_max_workers`
- Helm `readinessProbe` points to `/ready` instead of `/health`
- CI Docker cache key content-based (`hashFiles`) instead of commit SHA
- Security workflow `pip-audit` syncs with `--extra anthropic` to match production image
- `analyst.py` except blocks now log `exc_info=True` for debugging
- `_extract_text_content` used consistently in all fallback paths

### Fixed
- `_APP_VERSION` and `Chart.yaml` aligned to `0.2.0`
- `Chart.yaml` maintainer `your-name` → `brescou`
- `docs/security.md` link `your-org` → `brescou`
- Terraform secrets: `data` → `string_data` (GKE and EKS modules)
- Helm ConfigMap: conditional `REDIS_URL` when `memoryBackend=redis`
- `CLAUDE.md` directory tree: added `core/observability.py`
- Dockerfile: documented `--build-arg LLM_EXTRAS` for Redis/Postgres production builds

## [0.2.0] - 2026-04-01

### Added
- Provider-agnostic LLM factory (`core/llm.py`) supporting Anthropic, OpenAI, Google, AWS Bedrock, Azure OpenAI, and Ollama
- `get_settings()` with `@lru_cache` replacing module-level singleton
- `ConversationMemory.list_runs_by_session()` with SQL-side session filtering
- Optional Bearer token API authentication (`API_KEY` env var)
- SSE stream timeout enforcement via `asyncio.timeout`
- Shared LLM and checkpointer instances pre-warmed at FastAPI lifespan startup
- Helm chart for Kubernetes deployment (`infra/helm/`)
- Terraform modules for GKE and EKS (`infra/terraform/`)
- Multi-agent example patterns: sequential, parallel, supervisor, human-in-the-loop
- `agents/models.py` — decoupled `ResearchResult` and `AnalysisReport` dataclasses
- `_extract_text_content()` helper for safe json.loads on multi-modal LLM responses
- `MultiAgentGraph` context manager protocol (`__enter__`/`__exit__`)
- Graceful shutdown guard — endpoints return 503 when `_shutting_down` is set
- `session_id` validation: `max_length=128`, `pattern=^[a-zA-Z0-9_-]+$` on `RunRequest`
- `THREAD_POOL_MAX_WORKERS` setting (default 4) for configurable thread pool sizing
- `.dockerignore` to exclude .env, .git, tests, docs from Docker build context
- PodDisruptionBudget template for Helm chart
- `@model_validator` on Settings enforcing `POSTGRES_URL` when `memory_backend=postgres`
- Exponent guard (`max 1000`) on calculator `ast.Pow` operator
- `math.isfinite()` guard before `int(result)` in calculator tool
- `tests/test_config.py` — settings caching, llm_config, cross-field validators
- `tests/test_observability.py` — structured logging and OTel tracing coverage

### Changed
- `MultiAgentGraph` now uses the configured memory backend instead of hardcoded `MemorySaver`
- All LLM providers now honour `max_tokens` (or equivalent) from settings
- `sanitize_log_data` now recurses into list values and checks sensitive key before type
- Middleware registration order: rate limiter now executes before auth (brute-force protection)
- `_stream_pipeline` uses thread-safe `get_shared_llm()`/`get_shared_checkpointer()` accessors
- SSE error events return generic messages instead of leaking `str(exc)` internals
- ConversationMemory read operations (`get_run`, `list_runs`, `list_runs_by_session`) protected by `_lock`
- SQLite isolation level changed from `None` to `DEFERRED` with `threading.Lock` protection
- `_input_validator` renamed to `input_validator` (cross-module import convention)
- Conftest fixtures use real `ResearchResult`/`AnalysisReport` instances (function-scope)
- LLM/checkpointer initialization uses double-checked locking with `threading.Lock`
- Agents return partial state updates instead of full `{**state, ...}` copies
- `run()`/`run_structured()` DRY-ed via shared `_execute()` method in both agents
- Retry logic uses jitter (`0.5 + random.random()`) in exponential backoff
- Dockerfile comment corrected: `--frozen` → `--locked`
- Docker Compose Redis healthcheck uses `-a ${REDIS_PASSWORD:-changeme}`
- Helm `replicas` field conditional on `autoscaling.enabled`
- CI workflow: `permissions: contents: read`, Python matrix (3.12+3.13), parallel lint/test
- `pyproject.toml` coverage threshold raised from 50% to 70%

### Fixed
- `dir()` antipattern in `ResearchAgent._node_summarize` replaced with proper scoping
- Redis URL no longer logged in plain text (credentials stripped)
- `_is_sensitive_key` now uses word-boundary regex to avoid false positives
- Redis `requirepass` now compatible with healthcheck (`-a` flag added)
- Terraform GKE `deletion_protection` condition: `"production"` → `"prod"`
- SSTI regex refined to avoid false positives on `{{config}}` while catching real injections

## [0.1.0] - 2026-03-01

### Added
- Initial release of langgraph-agent-stack template
- `ResearchAgent` and `AnalystAgent` with LangGraph state machines
- `MultiAgentGraph` orchestrator with conditional routing
- FastAPI REST API with SSE streaming
- SQLite / Redis / PostgreSQL checkpointing via `ConversationMemory`
- Security: `InputValidator`, `RateLimiter`, bandit SAST, gitleaks scanning
- CI/CD with GitHub Actions (lint, test, Docker build)
- Docker multi-stage build and docker-compose for local development
