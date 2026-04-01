# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Changed
- `MultiAgentGraph` now uses the configured memory backend instead of hardcoded `MemorySaver`
- All LLM providers now honour `max_tokens` (or equivalent) from settings
- `sanitize_log_data` now recurses into list values and checks sensitive key before type

### Fixed
- `dir()` antipattern in `ResearchAgent._node_summarize` replaced with proper scoping
- Redis URL no longer logged in plain text (credentials stripped)
- `_is_sensitive_key` now uses word-boundary regex to avoid false positives

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
