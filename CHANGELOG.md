# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **MCP server** (#92) — optional streamable-HTTP endpoint at `/mcp` (`MCP_SERVER_ENABLED`, `uv sync --extra mcp`) that auto-exposes one MCP tool per registered domain pack. Tools share the typed REST execution path (validation, budgets, compliance gating); regulated packs are omitted from the tool list when gated off.

### Fixed
- **Inconsistent `cost_usd` in mock mode** (#88) — typed pack routes returned `null` while legacy `POST /run` returned `0.0`. `MockProviderChatModel` now emits deterministic `usage_metadata`, `mock-provider` is priced at $0 in the cost table, and `StructuredLLMPack` wires `CostTracker` so every route (including SSE finals) reports `0.0`. Budget `402` is now exercisable in mock via `LLM_COST_TABLE_PATH` override.
- **Vector store TODOs** (#90) — every RAG backend now receives an explicit embeddings instance via `core/embeddings.py` (factory parallel to `get_llm`; `EMBEDDING_PROVIDER=mock` is deterministic and keyless). PGVector uses maintained `langchain_postgres.PGVector` (new `postgres` extra dependency) instead of the deprecated community class.

## [0.6.2] - 2026-07-09

### Fixed
- **Invalid API key returned HTTP 200 with degraded placeholder content** (launch-audit MAJEUR) — with a wrong or placeholder `ANTHROPIC_API_KEY`, the provider's 401 was swallowed by node-level parse fallbacks and `POST /run` answered 200 with `"Summary unavailable."` / `"structured extraction failed"` and `cost_usd: 0.0`. New `AgentAuthenticationError` (raised from `BaseAgent._invoke_llm_with_retry` and every raw pack `llm.invoke` via `is_auth_llm_error` on HTTP 401/403 or typed SDK auth exceptions) now propagates through agents and all built-in packs to the API, which returns **HTTP 502** with an actionable detail ("Check ANTHROPIC_API_KEY in your .env … or set LLM_PROVIDER=mock"). SSE streams emit the same message as an explicit `error` event. Auth errors are never retried. A `find_auth_cause` net on the pack routes also converts third-party plugin packs' wrapped 401s into the same 502.
- **Same bug, non-auth case: any fatal LLM error was swallowed identically** — every node in `ResearchAgent` and `AnalystAgent` wrapped its `_invoke_llm_with_retry` call in a broad `except Exception`, so a persistent 5xx (retries exhausted), a 404 on a bad model name, or a content-policy refusal produced the same silent placeholder output as the auth case, just without the 502 mapping. Each node's LLM call now runs unguarded — only the JSON-parsing step below it is wrapped, catching just the specific parsing exceptions (`json.JSONDecodeError`, and `AttributeError`/`TypeError`/`ValueError` where a malformed-but-valid-JSON payload could bomb on `.get()`/`float()`) — so any other exception propagates as a real `AgentExecutionError` (mapped to HTTP 500, or 502 if it's an auth error) instead of a masked 200.
- **Typed pack SSE streams failed on the default sqlite config** — `summariser` and every `StructuredLLMPack` vertical drove a sync `graph.invoke` on the event loop thread from their stream generators; `AsyncSqliteSaver` rejects that ("Synchronous calls … only allowed from a different thread"), so `/packs/{id}/run/stream` emitted a generic error event on a fresh clone (the v0.6.1 B1 fix only covered `/run/stream`). `run_from_input` now runs via `asyncio.to_thread`; regression tests added in `tests/test_sse_integration.py` for `summariser` and `meeting_prep`.
- **Stale model reference in `docs/security.md`** — the env var table documented `ANTHROPIC_MODEL`'s default as `claude-3-5-sonnet-20241022`; corrected to `claude-sonnet-5` (matches `core/config.py` and `.env.example`).

### Changed
- **README** — new Troubleshooting section (invalid key → 502, mock output, `/metrics` 404 without the observability extra, 402 budget, 422-vs-403 on regulated packs, sqlite scaling); tests badge updated 758+ → 790+; regulated-pack note clarifies that schema validation (`422`) runs before the compliance gate (`403`).
- **Language hygiene** — translated the remaining French docstrings/comments to English (`examples/custom_pack/pack.py`, `examples/custom_pack/schemas.py`, `agents/researcher.py`, `core/security.py`).
- **`core/cost.py`** — documented Claude Sonnet 5's introductory pricing ($2/$10 per MTok through 2026-08-31) next to the sticker-price entry.
- **Dev dependencies** — pinned `pip>=26.1.2` (PYSEC-2026-196).
- **`tests/test_integration_backends.py`** — the full-pipeline E2E mock LLM fixture provided only 5 canned responses for a run that needs 6; it "passed" only because the old swallowing bug masked the resulting `StopIteration`. Replaced with one well-formed, call-site-matched response per node (no looped re-validation), so the test now exercises the real happy path instead of degraded fallback text.

## [0.6.1] - 2026-07-08

### Fixed
- **SSE streaming broken on default sqlite config (B1)** — `stream_events()` failed with `NotImplementedError` because the lifespan used sync `SqliteSaver`. Async checkpointers (`AsyncSqliteSaver`, `AsyncRedisSaver`, `AsyncPostgresSaver`) now back streaming/async paths; sync savers remain for `run()`. Added `tests/test_sse_integration.py` exercising real checkpointers over `/run/stream` and pack stream routes.
- **Four structured packs returned HTTP 500 in mock mode (B2)** — `meeting_prep`, `executive_brief`, `rfp_assistant`, and `support_triage` received generic mock JSON that failed Pydantic validation. New `core/mock_llm.py` provides schema-aware mock responses; added parametrized `tests/test_mock_packs_api.py` covering all 13 built-in packs (200 or 403 for regulated).
- **`docker compose up` failed on REDIS_PASSWORD interpolation (B3)** — `${REDIS_PASSWORD:?}` was evaluated even without the redis profile. Compose now fails explicitly only when the redis profile is active without a password; `.env.example` documents `REDIS_PASSWORD`.
- **CI `readme-smoke` job** — capture SSE via a temp file instead of `curl | head` so SIGPIPE does not fail the job under `pipefail`; ruff lint/format fixes on `core/llm.py`, `core/mock_llm.py`, and `tests/test_mock_packs_api.py`.

### Changed
- **`APP_VERSION` derived from package metadata** (M1) — `api/state.py` reads `importlib.metadata.version("langgraph-agent-stack")`; `pyproject.toml` bumped to `0.6.1`.
- **Bedrock unit tests skip without `botocore`** (M2) — `pytest.importorskip("botocore")` on Bedrock test classes so `make test` passes with `--extra anthropic` only.
- **Dependency upgrades for known CVEs** (M3) — `aiohttp`, `langgraph-sdk`, `pydantic-settings`, `msgpack`, `cryptography`, `chromadb`.
- **README badges** — Python 3.13, coverage ~86%.

### Added
- **CI job `readme-smoke`** (T4) — runs `scripts/readme-smoke.sh`, which executes the README quickstart verbatim (mock server, curl assertions, SSE event, `docker compose config`). Coupled to README by design.

## [0.6.0] - 2026-07-07

### Fixed
- **`LLM_PROVIDER=mock` returned HTTP 500 on every request** — `BaseAgent.__init__` unconditionally called `llm.bind_tools(self.tools)`, which `FakeListChatModel` (the mock provider's chat model) does not implement. Now falls back to the unbound LLM when `bind_tools()` raises `NotImplementedError`. Also fixed the mock LLM's canned response ordering, which did not match the real 6-call sequence of the `research_analysis` pipeline and produced incoherent output (empty insights, raw JSON as summary) even once the 500 was gone. Added a regression test that exercises the real `/run` endpoint end-to-end with `LLM_PROVIDER=mock` (no pack/agent mocking).
- **`cp .env.example .env` crashed on startup** — `.env.example` ships `PACK_DEFAULT_BUDGET_USD=` uncommented with an empty value ("empty = no limit"), but `Settings()` raised a `float_parsing` error on the empty string. Added a field validator that treats a blank string as unset for this field.

### Changed — Sprint highlights (models, security, README, release)
- **Default LLM models updated** to current (July 2026) releases: Anthropic `claude-sonnet-5`, OpenAI `gpt-5.5`, Google `gemini-3.5-flash`, Bedrock `anthropic.claude-sonnet-5` — across `core/config.py`, `.env.example`, and the `core/cost.py` pricing table.
- **Security Scanning workflow green on `main`** — added `.gitleaks.toml` allowlisting confirmed test-fixture false positives; upgraded `starlette` and `langsmith` to patch known CVEs.
- **Docs/code cleanup** — removed AI planning scaffolding (`docs/superpowers/`) from the repo; fixed stale `RAG`/`platform/`-package references and translated remaining French comments in the Helm `values.yaml` to English.
- **README rewritten for public launch**: a non-defensive "Why this instead of…" section positioning this template against LangGraph Platform (self-hosted, MIT, own observability/cost data — both are legitimate choices) and generic agent-service-toolkit templates (per-run budgets with HTTP 402, pack versioning with canary weights, Cosign + SBOM supply chain); a bulletproof quickstart with a `LLM_PROVIDER=mock` "try without an API key" callout, a `uv` install link, a mock-search note, and a cost estimate derived from the `core/cost.py` pricing table; credibility signals (758+ tests / ~87% coverage badges, a SQLite-dev-vs-Redis/Postgres-prod scaling callout, and a note that the regulated HR/legal/finance packs are gated behind `REGULATED_PACKS_ENABLED=false`, returning HTTP 403 until opted in).
- **Repo hygiene** — added `.github/PULL_REQUEST_TEMPLATE.md` (description / tests / lint / docs checklist) and a root `SECURITY.md` pointing to `docs/security.md`'s full vulnerability reporting procedure.

### Changed — Platform kernel and pack catalogue
- **API package split** — monolithic `api/main.py` refactored into `api/app.py`, `lifespan.py`, `middleware.py`, `dependencies.py`, `router_factory.py`, and `api/endpoints/*`.
- **Domain packs namespaced by business domain** — `domain_packs/research/`, `productivity/`, `hr/`, `finance/`, `legal/`, `common/` (HR packs moved from `domain_packs/rh/` to `domain_packs/hr/`).
- **Built-in pack registration** — consolidated in `pack_kernel/builtin_packs.py` (called from `api/lifespan.py`).
- Documentation updated for the new layout (`README.md`, `domain_packs/README.md`, `docs/architecture.md`, `CONTRIBUTING.md`, `connectors/README.md`).

### Added
- **Third and fourth domain packs** — `summariser` (`SummariserPack`, single-agent bullet summary) and `analysis_only` (`AnalysisOnlyPack`, AnalystAgent on pre-supplied research). Registered in `pack_kernel/builtin_packs.py` with control-plane policies and typed API routes (`POST /packs/summariser/run`, `POST /packs/analysis_only/run`).
- **`domain_packs/README.md`** — catalogue of built-in packs and authoring guide.
- API helpers `_pack_primary_text`, `_invoke_pack_run`, `_iter_pack_stream_events`, `_serialize_pack_result` so typed bodies with `text` (not only `query`) work on pack routes.
- **Nine vertical domain packs** — `meeting_prep`, `rfp_assistant`, `support_triage`, `executive_brief`, `contract_reviewer`, `financial_memo`, plus HR packs under `domain_packs/hr/`: `talent_screening`, `job_description_writer`, `hr_policy_qa`. Shared base: `domain_packs/common/structured_llm.py` (`StructuredLLMPack`).
- **`pack_kernel/builtin_packs.py`** — single registration source for all built-in packs.

### Changed
- **Platform kernel renamed to `pack_kernel/`** — the former top-level `platform/` package shadowed Python's stdlib `platform` module (required fragile `conftest.py` bootstrap and a `sys.path` scan hack). All imports now use `pack_kernel`; the old `platform/` path is removed (no compat shim — any shim would still shadow the stdlib).
- **Proxy-aware rate limiting** — honour `X-Forwarded-For` / `Forwarded` only from trusted peers (`TRUST_PROXY_HEADERS`, `FORWARDED_ALLOW_IPS`); per-Bearer-token buckets when `API_KEY` is set; Helm prod values and uvicorn `--forwarded-allow-ips` documented.
- **Regulated vertical pack compliance scaffolding** — `PackPolicy.human_review_required`, server-injected mandatory `disclaimer` + `human_review_required` on outputs for HR/legal/finance packs; `COMPLIANCE.md` per regulated pack; `tests/test_compliance.py`.
- `api/dependencies.py` — injects shared connector into any pack whose `__init__` accepts `connector=` (not only `research_analysis`).
- `control_plane/__init__.py` — policy table covers all registered packs.

### Security
- **`validate_pack_body`** — scans every string field on typed pack requests for injection/SSRF patterns (closes gap where only the primary `query`/`text` label was validated).
- **`InputValidator.check_content_safety`** — per-field length + pattern checks for document-sized inputs.
- **Bandit CI scope** extended to `platform/`, `domain_packs/`, `connectors/`, `control_plane/`.
- **LLM JSON cap** — `StructuredLLMPack` rejects parsed responses over 512 KiB.
- **Prompt injection guards** — `domain_packs/common/prompt_safety.py` wraps untrusted content in delimiters; vertical packs use `format_vertical_prompt`.
- **Production auth** — `Settings` rejects `ENVIRONMENT=production` without `API_KEY`.
- **Bandit clean-up** — retry jitter via tenacity `wait_random_exponential` (stdlib `random`, not `secrets`); pack traffic split uses `random.choices` with `# nosec B311`; no bare `except: pass` on connection close.
- **Pyright** — `domain_packs/` included in typecheck scope; agent init narrowed for optional members.

## [0.5.0] - 2026-05-19

### Added
- **Platform kernel** — `platform/` (`BaseDomainPack`, `PackRegistry`), `domain_packs/research_analysis/` (`ResearchAnalysisPack`); `core/graph.py` shim (`MultiAgentGraph` alias); `DEFAULT_PACK_ID`; contract and API tests.
- **Per-run cost attribution** (`core/cost.py`): `CostTracker` callback handler accumulates token costs per model run using a configurable pricing table; `BudgetExceededError` and `AgentBudgetExceededError` raised when `budget_usd` is exceeded; `pack_run_cost_usd_total` Prometheus counter emitted per run. `cost_usd` exposed on agent, pack, and API responses; HTTP 402 returned when budget is exceeded. New settings: `PACK_DEFAULT_BUDGET_USD`, `LLM_COST_TABLE_PATH`.
- **Typed pack schemas + auto API wiring** (`platform/base_pack.py`, `platform/registry.py`): `input_schema`/`output_schema`/`version` ClassVars on `BaseDomainPack`; `get_schemas()` and `list_packs_with_metadata()` on `PackRegistry`; dynamic `_build_pack_router()` generates per-pack endpoints at startup. New endpoints: `GET /packs`, `GET /packs/{pack_id}/versions`, `PATCH /packs/{pack_id}/versions/{version}/weight`. `ResearchAnalysisInput`/`ResearchAnalysisOutput` Pydantic schemas in `domain_packs/research_analysis/schemas.py`.
- **Pack versioning + traffic split** (`platform/registry.py`): `PackVersion` dataclass; `_registry` refactored to `dict[str, list[PackVersion]]`; `set_weights()` for traffic-split configuration. `X-Pack-Version` request header pins to a specific version; `X-Pack-Version-Used` response header reports the actual version used. Sticky session support via `get_pack_version_for_session()` (SQLite backend). `save_run` stores `pack_version` metadata.
- **Second domain pack** `research_only` (`ResearchOnlyPack`) with typed `/packs/research_only/run` routes.
- **Retrieval connectors** — `example_memory`, `http` (`CONNECTOR_HTTP_URL`), and `rag` (`RAG_ENABLED`); API injection via `CONNECTOR_ENABLED` into `ResearchAnalysisPack`; optional `connector=` on the pack constructor.
- **Control plane enforcement** — `PolicyRegistry`, `control_plane/enforce.py` (per-pack query limits, budget ceiling, stream timeout); foundation types in `connectors/` and `control_plane/`.
- **Sticky pack versions** on Redis and Postgres run-history backends (`get_pack_version_for_session`).
- `examples/custom_pack/`: `SummariserPack` reference implementation showing how to author a third-party domain pack.

### Changed
- `agents/base_agent.py`: `budget_usd` constructor parameter added; `cost_usd` property reads from attached `CostTracker`.
- `domain_packs/research_analysis/pack.py`: cost propagation through the pipeline; `cost_usd` property on the pack instance; optional connector merge in the research phase.
- `api/models.py`: `cost_usd` field added to `RunResponse` and `ResearchResponse`.
- `infra/Dockerfile`: copies `platform/`, `domain_packs/`, `connectors/`, `control_plane/` into the runtime image.

## [0.4.0] - 2026-04-08

### Added
- **Mock LLM provider** (`LLM_PROVIDER=mock`) — run the full pipeline without any
  API key using deterministic `FakeListChatModel` responses from langchain-core
- **Real-time SSE streaming** via `MultiAgentGraph.stream_events()` using LangGraph's
  `astream_events(version="v2")` — replaces the batch `phase_completed` events with
  true `phase_started`, `phase_completed`, and `token` events streamed as nodes execute
- **Redis run history backend** (`RedisRunHistory`) — stores run history in Redis
  hashes with sorted sets for chronological and session-based ordering; eliminates
  the need for a SQLite PVC when `MEMORY_BACKEND=redis`
- **PostgreSQL run history backend** (`PostgresRunHistory`) — stores run history in a
  `run_history` table with native JSONB session filtering
- `create_run_history(settings)` factory in `core/memory.py` — auto-selects the
  matching backend with graceful fallback to SQLite
- `RunHistoryStore` protocol for type-safe backend interchangeability
- Terraform: private cluster endpoints for all three clouds
  - EKS: `endpoint_private_access = true`, configurable `public_access_cidrs`
  - GKE: `private_cluster_config` + `master_authorized_networks_config`
  - AKS: `api_server_access_profile` with `authorized_ip_ranges`
- Terraform: dedicated VPC for EKS (replaces default VPC) with private/public
  subnets, NAT Gateway, and proper route tables
- Prometheus metrics: `llm_request_duration_seconds` (Histogram) and
  `llm_tokens_total` (Counter by direction) in `_invoke_llm_with_retry`
- `contextvars.copy_context()` in `_run_in_executor` — request_id now propagates
  into thread pool workers for log correlation
- Helm NetworkPolicy: namespace-scoped ingress + egress rules (DNS + HTTPS),
  enabled by default in `values.prod.yaml`
- 7 unit tests for `extract_text_content` (multi-modal, list, fallback paths)
- 6 `pytest-asyncio` tests for SSE streaming with `httpx.AsyncClient`
- 2 tests for Mock LLM provider
- Terraform module for Azure AKS (`infra/terraform/modules/aks/`) with Log Analytics
  workspace, auto-scaling node pool, System-Assigned Managed Identity, and Helm chart
  deployment
- AKS entry point (`infra/terraform/aks/`) with `subscription_id` (mandatory since
  AzureRM 4.x) and `redis_url` variables
- `redis_url` variable added to EKS and GKE modules/entry points for secret parity
  with AKS

### Changed
- **BREAKING (SSE):** `/run/stream` event types changed:
  - `agent_switch` → `phase_started` + `phase_completed` (emitted in real time)
  - New `token` event type for LLM token-level streaming
- **BREAKING (Terraform):** Provider version bumps across all modules:
  - `hashicorp/azurerm` `~> 3.0` → `~> 4.0`
  - `hashicorp/aws` `~> 5.0` → `~> 6.0`
  - `hashicorp/google` `~> 5.0` → `~> 7.0`
  - `hashicorp/helm` `~> 2.12` → `~> 3.1`
  - `hashicorp/kubernetes` `~> 2.25` → `~> 3.0`
- CORS fail-closed in production: no wildcard unless `CORS_ORIGINS` is explicitly set
- `/docs` and `/redoc` disabled when `ENVIRONMENT=production`
- Rate limiter initialisation moved from import time to lifespan (no Redis
  connection on import)
- `thread_pool_max_workers` default raised from 4 to 8
- `_extract_text_content` renamed to `extract_text_content` (public API)
- `to_dict()` on dataclasses uses `dataclasses.asdict()` instead of manual dict
- `vars(report)` replaced with `report.to_dict()` for consistency
- `except AgentError` narrowed to `except (AgentExecutionError, AgentTimeoutError,
  AgentValidationError)` — `AgentConfigurationError` now propagates to the caller
- `MultiAgentGraph._get_executor()` uses double-checked locking
- `MultiAgentGraph.close()` uses `wait=True` and resets `_executor = None`
- `_NoOpTracer` singleton cached at module level
- `_safe_eval` return type narrowed from `Any` to `int | float`
- Trivy CI action pinned to SHA (`@6e7b7d1f...`) instead of `@master`
- Redis default password removed from `docker-compose.yml` — fails if
  `REDIS_PASSWORD` not set
- Terraform root directory converted to documentation-only (per-cloud entry points)

### Fixed
- **CRITICAL:** `request_id` lost in thread pool — `contextvars.copy_context()` now
  propagates context variables into executor threads
- **CRITICAL:** Race condition in `MultiAgentGraph._get_executor()` — concurrent
  `arun()` calls could create multiple thread pools
- **CRITICAL:** `_run_in_executor` silently fell back to unbounded default executor
  when `_executor is None` — now raises `RuntimeError`
- **CRITICAL:** Resource leak in `_stream_pipeline` — `MultiAgentGraph` was not
  closed via context manager, leaking the internal `ThreadPoolExecutor`
- `type: ignore[union-attr]` in `analyst.py` replaced with proper
  `extract_text_content()` call for multi-modal LLM content
- ConversationMemory warning for non-SQLite backends changed to `logger.info` with
  actionable persistence guidance

## [0.3.0] - 2026-04-02

### Added
- CI job `integration`: `pytest -m integration` with Docker + `uv sync --extra redis --extra postgres`
- E2E tests (`tests/test_integration_real.py`): full `MultiAgentGraph.run()` with real SqliteSaver, PostgresSaver, and RedisSaver (redis-stack image for RedisJSON)
- README: `RATE_LIMIT_BACKEND`, Redis rate limiter fail-open policy, test suite layout (mocked vs integration), `/health` LLM semantics, Prometheus metrics reference table
- `/ready` readiness probe endpoint — separate from `/health` liveness probe
- Fail-fast checkpointer in production (`_fallback_or_raise` in `core/memory.py`)
- OTel `insecure=True` conditional on `OTEL_EXPORTER_OTLP_INSECURE` or localhost endpoint
- LLM retry with backoff activated in all 6 agent graph nodes
- `Content-Security-Policy: default-src 'self'` header (replaces deprecated `X-XSS-Protection`)
- Helm NetworkPolicy template (opt-in via `networkPolicy.enabled`)
- Tests: SSE done event validation, timeout error event, shutdown 503 guard, rate limiting on /research, auth exempt paths, Redis/Postgres checkpointer mocks, populated session history
- Tests: all 5 security headers asserted (CSP, Referrer-Policy, Cache-Control, X-Frame-Options, X-Content-Type-Options)
- Tests: `/ready` endpoint (200 OK, 503 LLM not init, 503 shutting down)
- Tests: `/health` with LLM initialised — regression test for `llm_provider.value` bug
- Tests: vectorstore happy path (Chroma + PGVector with mocked imports)
- Tests: `recall_history` tool (empty, populated, error, no-summary-key)
- Tests: Tavily/SerpAPI search provider branches (success + ImportError)
- CORS production hardening documented in README Security section
- Terraform remote state backend warning and instructions in `versions.tf`
- All 20+ Makefile targets documented in README

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
- Rate limit middleware now excludes `/ready` alongside `/health`

### Fixed
- **CRITICAL**: `settings.llm_provider.value` → `settings.llm_provider` — `LLMProvider` is `Literal` (str), not Enum; `.value` caused `AttributeError` on `/health`
- Release versioning: `pyproject.toml`, Helm `Chart.yaml`, and API `_APP_VERSION` set to `0.3.0`
- `Chart.yaml` maintainer `your-name` → `brescou`
- `docs/security.md` link `your-org` → `brescou`
- Terraform `kubernetes_secret`: use provider-valid `data` map (invalid `string_data` removed in GKE and EKS modules)
- CI infra-lint: validate Terraform modules only; root module uses legacy child providers + `count` (not validatable in CI without credentials)
- Helm: moved `REDIS_URL` from ConfigMap to Secret (passwords must not be in cleartext ConfigMaps)
- `CLAUDE.md` directory tree: added `core/observability.py`
- Dockerfile: pinned base images by SHA digest; documented `--build-arg LLM_EXTRAS`

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
