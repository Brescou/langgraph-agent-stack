"""
core/config.py — Application settings and configuration management.

Loads all configuration from environment variables / .env file using
pydantic-settings.  Never hard-code secrets here; use a .env file or the
shell environment instead.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.llm import LLMProvider

if TYPE_CHECKING:
    from core.llm import LLMConfig


class MemoryBackend(StrEnum):
    """Supported memory/persistence backends."""

    SQLITE = "sqlite"
    REDIS = "redis"
    POSTGRES = "postgres"


class LogLevel(StrEnum):
    """Standard Python log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """
    Central application settings loaded from environment variables.

    All secrets (API keys, connection strings) MUST be provided via the
    environment or a ``.env`` file — never committed to version control.

    Attributes:
        anthropic_api_key: Anthropic API key for ChatAnthropic.
        llm_provider: Active LLM provider name.
        anthropic_model: Anthropic model identifier.
        max_tokens: Maximum tokens per LLM response.
        memory_backend: Storage backend for agent memory/checkpoints.
        redis_url: Redis connection URL (required when memory_backend=redis).
        sqlite_path: File path for SQLite database (used in dev mode).
        log_level: Python logging level for the application.
        api_host: Host the FastAPI server binds to.
        api_port: Port the FastAPI server listens on.
        max_research_iterations: Safety cap on research loop iterations.
        max_step_count: Hard limit on total agent steps per run.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
        # .env.example documents several optional fields as "empty = disabled"
        # (e.g. PACK_DEFAULT_BUDGET_USD, API_KEY). Without this, `cp
        # .env.example .env` breaks non-str optional fields: pydantic-settings
        # otherwise tries to parse "" as e.g. a float and raises.
        env_parse_none_str="",
    )

    # --- LLM ---
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key — required when llm_provider='anthropic'.",
    )
    llm_provider: LLMProvider = Field(
        default="anthropic",
        validation_alias="LLM_PROVIDER",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-5",
        validation_alias="ANTHROPIC_MODEL",
    )
    anthropic_base_url: str | None = Field(
        default=None,
        validation_alias="ANTHROPIC_BASE_URL",
        description=(
            "Optional Anthropic API base URL (LiteLLM, Helicone, internal gateway)."
        ),
    )
    openai_api_key: str | None = Field(default=None)
    openai_model: str = Field(default="gpt-5.5", validation_alias="OPENAI_MODEL")
    openai_base_url: str | None = Field(
        default=None,
        validation_alias="OPENAI_BASE_URL",
        description=(
            "Optional OpenAI-compatible base URL (LiteLLM, OpenRouter, Together AI)."
        ),
    )
    google_api_key: str | None = Field(default=None)
    google_model: str = Field(
        default="gemini-3.5-flash", validation_alias="GOOGLE_MODEL"
    )
    google_base_url: str | None = Field(
        default=None,
        validation_alias="GOOGLE_BASE_URL",
        description="Optional Google Generative AI API base URL override.",
    )
    aws_access_key_id: str | None = Field(
        default=None,
        description=(
            "Optional static AWS access key for Bedrock (local dev). "
            "Leave unset on EKS to use IRSA / Pod Identity via the default "
            "boto3 credential chain."
        ),
    )
    aws_secret_access_key: str | None = Field(
        default=None,
        description=(
            "Optional static AWS secret key for Bedrock (local dev). "
            "Must be set together with aws_access_key_id when used."
        ),
    )
    aws_region: str = Field(default="us-east-1", validation_alias="AWS_REGION")
    bedrock_model: str = Field(
        default="anthropic.claude-sonnet-5",
        validation_alias="BEDROCK_MODEL",
    )
    bedrock_endpoint_url: str | None = Field(
        default=None,
        validation_alias="BEDROCK_ENDPOINT_URL",
        description="Optional Bedrock Runtime endpoint URL (VPC endpoint, proxy).",
    )
    azure_openai_api_key: str | None = Field(default=None)
    azure_openai_endpoint: str | None = Field(default=None)
    azure_openai_base_url: str | None = Field(
        default=None,
        validation_alias="AZURE_OPENAI_BASE_URL",
        description="Optional Azure OpenAI HTTP base URL override (gateway / proxy).",
    )
    azure_openai_deployment: str = Field(
        default="gpt-4o",
        validation_alias="AZURE_OPENAI_DEPLOYMENT",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(default="llama3.2", validation_alias="OLLAMA_MODEL")
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=32768,
        description="Maximum tokens to generate per LLM call.",
    )
    llm_request_timeout_seconds: float = Field(
        default=120.0,
        ge=1.0,
        description=(
            "Wall-clock timeout for a single synchronous LLM HTTP request. "
            "Distinct from STREAM_TIMEOUT_SECONDS (whole SSE pipeline)."
        ),
        validation_alias="LLM_REQUEST_TIMEOUT_SECONDS",
    )

    default_pack_id: str = Field(
        default="research_analysis",
        validation_alias="DEFAULT_PACK_ID",
        description="Pack ID to use when no pack is specified. Must be registered in PackRegistry.",
    )

    pack_default_budget_usd: float | None = Field(
        default=None,
        validation_alias="PACK_DEFAULT_BUDGET_USD",
        description=(
            "Optional default USD cost budget applied to every agent run. "
            "Overridden per-agent by passing budget_usd= to BaseAgent.__init__. "
            "Set to None (the default) to disable budget enforcement globally."
        ),
    )

    llm_cost_table_path: Path | None = Field(
        default=None,
        validation_alias="LLM_COST_TABLE_PATH",
        description="Path to JSON file with custom LLM pricing. See core/cost.py for format.",
    )

    # --- Memory / Persistence ---
    memory_backend: MemoryBackend = Field(
        default=MemoryBackend.SQLITE,
        description="Backend used for agent state checkpointing.",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL (only used when memory_backend=redis).",
    )
    sqlite_path: str = Field(
        default="./data/agent_memory.db",
        description="Path to the SQLite database file (dev/test only).",
    )
    postgres_url: str | None = Field(
        default=None,
        validation_alias="POSTGRES_URL",
        description="PostgreSQL DSN (required when memory_backend=postgres). "
        "Example: postgresql+psycopg://user:pass@localhost:5432/dbname",
    )
    rag_enabled: bool = Field(
        default=False,
        validation_alias="RAG_ENABLED",
        description="Enable RAG (Retrieval-Augmented Generation) via a vector store.",
    )
    connector_enabled: bool = Field(
        default=False,
        validation_alias="CONNECTOR_ENABLED",
        description=(
            "When true, inject a retrieval connector into ResearchAnalysisPack "
            "runs (legacy /run and /packs/research_analysis/*)."
        ),
    )
    connector_id: str = Field(
        default="example_memory",
        validation_alias="CONNECTOR_ID",
        description="Built-in connector id when CONNECTOR_ENABLED=true (see core/connectors.py).",
    )
    connector_http_url: str | None = Field(
        default=None,
        validation_alias="CONNECTOR_HTTP_URL",
        description="Base URL for CONNECTOR_ID=http (query param ``q``, ``limit``).",
    )
    connector_http_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        validation_alias="CONNECTOR_HTTP_TIMEOUT_SECONDS",
        description="HTTP client timeout for CONNECTOR_ID=http.",
    )
    connector_http_max_response_bytes: int = Field(
        default=1_048_576,
        ge=1024,
        validation_alias="CONNECTOR_HTTP_MAX_RESPONSE_BYTES",
        description="Max response body size for CONNECTOR_ID=http (bytes).",
    )
    connector_http_max_redirects: int = Field(
        default=5,
        ge=0,
        le=20,
        validation_alias="CONNECTOR_HTTP_MAX_REDIRECTS",
        description="Max redirects for CONNECTOR_ID=http (each hop SSRF-validated).",
    )

    # SharePoint (Microsoft Graph) connector — CONNECTOR_ID=sharepoint.
    # ACLs are enforced by the service principal: grant Sites.Selected on the
    # sites the agent may read.
    connector_sharepoint_tenant_id: str | None = Field(
        default=None,
        validation_alias="CONNECTOR_SHAREPOINT_TENANT_ID",
        description="Entra ID tenant for the Graph client-credentials flow.",
    )
    connector_sharepoint_client_id: str | None = Field(
        default=None,
        validation_alias="CONNECTOR_SHAREPOINT_CLIENT_ID",
        description="App registration client id (application permissions).",
    )
    connector_sharepoint_client_secret: str | None = Field(
        default=None,
        validation_alias="CONNECTOR_SHAREPOINT_CLIENT_SECRET",
        description="App registration client secret.",
    )

    # Google Drive connector — CONNECTOR_ID=gdrive.
    # Token is minted by the environment (Workload Identity / sidecar) so the
    # stack stays SDK-free; ACLs are enforced by the credentials' visibility.
    connector_gdrive_access_token: str | None = Field(
        default=None,
        validation_alias="CONNECTOR_GDRIVE_ACCESS_TOKEN",
        description="Short-lived Drive bearer token supplied by the environment.",
    )
    connector_gdrive_drive_id: str | None = Field(
        default=None,
        validation_alias="CONNECTOR_GDRIVE_DRIVE_ID",
        description="Optional shared-drive id to scope the search corpus.",
    )

    # --- Logging ---
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Application-wide logging verbosity.",
    )

    # --- API server ---
    api_host: str = Field(
        default="0.0.0.0",  # noqa: S104  # nosec B104 — required for container networking
        description="Host the FastAPI application binds to.",
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="TCP port the FastAPI application listens on.",
    )
    cors_origins_raw: str = Field(
        default="",
        validation_alias="CORS_ORIGINS",
        exclude=True,
        description="Raw comma-separated CORS origins string from the environment.",
    )

    @property
    def cors_origins(self) -> list[str]:
        """Return the list of allowed CORS origins parsed from ``CORS_ORIGINS``."""
        return [
            origin.strip()
            for origin in self.cors_origins_raw.split(",")
            if origin.strip()
        ]

    # --- Agent behaviour ---
    max_research_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of research iterations before forced completion.",
    )
    max_step_count: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Hard cap on total graph steps per agent run.",
    )
    stream_timeout_seconds: int = Field(
        default=120,
        ge=1,
        description="Maximum wall-clock seconds allowed for a streaming SSE pipeline run.",
        validation_alias="STREAM_TIMEOUT_SECONDS",
    )
    max_request_body_bytes: int = Field(
        default=1_048_576,
        ge=1024,
        le=52_428_800,
        description=(
            "Maximum inbound HTTP request body size in bytes. "
            "Enforced before JSON parsing."
        ),
        validation_alias="MAX_REQUEST_BODY_BYTES",
    )
    thread_pool_max_workers: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Size of the ThreadPoolExecutor used for blocking agent calls.",
        validation_alias="THREAD_POOL_MAX_WORKERS",
    )

    # --- Environment tag (informational) ---
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment tag used for log correlation.",
    )

    rate_limit_backend: Literal["memory", "redis"] = Field(
        default="memory",
        validation_alias="RATE_LIMIT_BACKEND",
        description=(
            "Rate-limiter backend. 'memory' (default) is per-process; "
            "'redis' shares state across replicas (requires REDIS_URL)."
        ),
    )

    session_registry_backend: Literal["memory", "redis"] = Field(
        default="memory",
        validation_alias="SESSION_REGISTRY_BACKEND",
        description=(
            "In-flight session registry backend. 'memory' (default) dedupes "
            "concurrent runs per session within a single process; 'redis' "
            "shares the locks across replicas (requires REDIS_URL)."
        ),
    )
    session_lock_ttl_seconds: int = Field(
        default=300,
        ge=10,
        validation_alias="SESSION_LOCK_TTL_SECONDS",
        description=(
            "Lifetime of Redis session locks. Must exceed the longest "
            "expected run (STREAM_TIMEOUT_SECONDS plus a margin) so a "
            "crashed replica cannot block a session forever."
        ),
    )

    review_store_backend: Literal["memory", "sqlite"] = Field(
        default="memory",
        validation_alias="REVIEW_STORE_BACKEND",
        description=(
            "Human-review queue backend for regulated pack outputs. 'memory' "
            "(default) is per-process and lost on restart; 'sqlite' persists "
            "decisions across restarts."
        ),
    )
    review_store_path: str = Field(
        default="./data/review_store.db",
        validation_alias="REVIEW_STORE_PATH",
        description="SQLite file path used when REVIEW_STORE_BACKEND=sqlite.",
    )

    pack_plugins_enabled: bool = Field(
        default=False,
        validation_alias="PACK_PLUGINS_ENABLED",
        description=(
            "Opt-in loading of third-party packs from the "
            "'langgraph_agent_stack.packs' entry-point group. Loading plugins "
            "executes third-party code: keep disabled unless the allowlist "
            "below names packages you trust."
        ),
    )
    pack_plugins_allowlist: str = Field(
        default="",
        validation_alias="PACK_PLUGINS_ALLOWLIST",
        description=(
            "Comma-separated entry-point names to load when "
            "PACK_PLUGINS_ENABLED=true. Empty means none (deny by default)."
        ),
    )

    @property
    def resolved_pack_plugins_allowlist(self) -> tuple[str, ...]:
        """Allowlisted plugin entry-point names (deduplicated, blanks removed)."""
        seen: dict[str, None] = {}
        for name in self.pack_plugins_allowlist.split(","):
            cleaned = name.strip()
            if cleaned:
                seen.setdefault(cleaned)
        return tuple(seen)

    api_key: str | None = Field(
        default=None,
        validation_alias="API_KEY",
        description=(
            "Optional single shared Bearer secret for API authentication. "
            "When set, all requests except health/docs/metrics must include "
            "'Authorization: Bearer <token>'. Not multi-tenant: no rotation, "
            "scopes, per-tenant keys, or caller audit — rate limits are per "
            "client IP (not per token). Use OAuth/OIDC or gateway auth for SaaS. "
            "Leave empty to disable (or when auth is upstream)."
        ),
    )

    api_keys: str | None = Field(
        default=None,
        validation_alias="API_KEYS",
        description=(
            "Optional comma-separated list of accepted Bearer secrets for "
            "multi-tenant API authentication (spaces around commas tolerated). "
            "Merged with the legacy single API_KEY: a request is authenticated "
            "when its token matches ANY configured key. Enables zero-downtime "
            "rotation (list old + new keys during the transition) and per-token "
            "rate limiting (active as soon as 2+ distinct keys are configured)."
        ),
    )

    @property
    def resolved_api_keys(self) -> tuple[str, ...]:
        """All configured Bearer secrets (legacy ``API_KEY`` + ``API_KEYS``).

        Merges the single legacy ``api_key`` with the comma-separated
        ``api_keys`` list, strips whitespace, drops empty entries, and
        de-duplicates while preserving order. Authentication is enabled as
        soon as this tuple is non-empty.
        """
        candidates: list[str] = []
        if self.api_key:
            candidates.append(self.api_key.strip())
        if self.api_keys:
            candidates.extend(part.strip() for part in self.api_keys.split(","))
        return tuple(dict.fromkeys(key for key in candidates if key))

    trust_proxy_headers: bool = Field(
        default=False,
        validation_alias="TRUST_PROXY_HEADERS",
        description=(
            "When true, honour X-Forwarded-For / Forwarded for client IP "
            "resolution and rate limiting, but only if the direct TCP peer "
            "matches FORWARDED_ALLOW_IPS."
        ),
    )

    forwarded_allow_ips: str = Field(
        default="",
        validation_alias="FORWARDED_ALLOW_IPS",
        description=(
            "Comma-separated IPs/CIDRs of trusted reverse proxies (Ingress, "
            "ALB, GCP LB). Required when TRUST_PROXY_HEADERS=true. Also passed "
            "to uvicorn --forwarded-allow-ips in the container entrypoint."
        ),
    )

    regulated_packs_enabled: bool = Field(
        default=False,
        validation_alias="REGULATED_PACKS_ENABLED",
        description=(
            "When false (default), HR/legal/finance regulated vertical packs "
            "(talent_screening, contract_reviewer, etc.) return HTTP 403. "
            "Enable only after completing the compliance programme documented "
            "in each pack's COMPLIANCE.md."
        ),
    )

    @property
    def llm_config(self) -> LLMConfig:
        """Build an :class:`~core.llm.LLMConfig` from the current settings."""
        from core.llm import LLMConfig

        return LLMConfig(
            provider=self.llm_provider,
            anthropic_api_key=self.anthropic_api_key,
            anthropic_model=self.anthropic_model,
            anthropic_base_url=self.anthropic_base_url,
            max_tokens=self.max_tokens,
            openai_api_key=self.openai_api_key,
            openai_model=self.openai_model,
            openai_base_url=self.openai_base_url,
            google_api_key=self.google_api_key,
            google_model=self.google_model,
            google_base_url=self.google_base_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_region=self.aws_region,
            bedrock_model=self.bedrock_model,
            bedrock_endpoint_url=self.bedrock_endpoint_url,
            azure_openai_api_key=self.azure_openai_api_key,
            azure_openai_endpoint=self.azure_openai_endpoint,
            azure_openai_base_url=self.azure_openai_base_url,
            azure_openai_deployment=self.azure_openai_deployment,
            ollama_base_url=self.ollama_base_url,
            ollama_model=self.ollama_model,
            request_timeout_seconds=self.llm_request_timeout_seconds,
        )

    @model_validator(mode="after")
    def _validate_backend_urls(self) -> Settings:
        if self.memory_backend.value == "postgres" and not self.postgres_url:
            raise ValueError("POSTGRES_URL must be set when MEMORY_BACKEND=postgres")
        if self.environment == "production" and not self.resolved_api_keys:
            raise ValueError(
                "API_KEY or API_KEYS must be set when ENVIRONMENT=production. "
                "Disable auth only in development/staging."
            )
        if self.connector_enabled:
            from connectors.resolver import list_connector_ids

            if self.connector_id not in list_connector_ids():
                raise ValueError(
                    f"CONNECTOR_ID {self.connector_id!r} is not supported. "
                    f"Use one of: {', '.join(list_connector_ids())}"
                )
            if self.connector_id == "http" and not self.connector_http_url:
                raise ValueError(
                    "CONNECTOR_HTTP_URL must be set when CONNECTOR_ID=http"
                )
            if self.connector_id == "rag" and not self.rag_enabled:
                raise ValueError("RAG_ENABLED must be true when CONNECTOR_ID=rag")
        return self

    @field_validator("redis_url")
    @classmethod
    def redis_url_scheme(cls, v: str) -> str:
        """Ensure the Redis URL uses a recognised scheme."""
        if not (v.startswith("redis://") or v.startswith("rediss://")):
            raise ValueError(
                f"redis_url must start with 'redis://' or 'rediss://' — got: {v!r}"
            )
        return v


# ---------------------------------------------------------------------------
# Settings factory — use get_settings() everywhere instead of a bare singleton.
# The @lru_cache ensures only one Settings instance is created per process.
# In tests, call get_settings.cache_clear() to force re-instantiation.
# ---------------------------------------------------------------------------


@lru_cache
def get_settings() -> Settings:
    """Return the cached application Settings instance."""
    return Settings()  # type: ignore[call-arg]  # key comes from env
