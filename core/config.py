"""
core/config.py — Application settings and configuration management.

Loads all configuration from environment variables / .env file using
pydantic-settings.  Never hard-code secrets here; use a .env file or the
shell environment instead.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator
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
        default="claude-3-5-sonnet-20241022",
        validation_alias="ANTHROPIC_MODEL",
    )
    openai_api_key: str | None = Field(default=None)
    openai_model: str = Field(default="gpt-4o", validation_alias="OPENAI_MODEL")
    google_api_key: str | None = Field(default=None)
    google_model: str = Field(default="gemini-1.5-pro", validation_alias="GOOGLE_MODEL")
    aws_access_key_id: str | None = Field(default=None)
    aws_secret_access_key: str | None = Field(default=None)
    aws_region: str = Field(default="us-east-1", validation_alias="AWS_REGION")
    bedrock_model: str = Field(
        default="anthropic.claude-3-5-sonnet-20241022-v2:0",
        validation_alias="BEDROCK_MODEL",
    )
    azure_openai_api_key: str | None = Field(default=None)
    azure_openai_endpoint: str | None = Field(default=None)
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

    # --- Logging ---
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Application-wide logging verbosity.",
    )

    # --- API server ---
    api_host: str = Field(
        default="0.0.0.0",
        description="Host the FastAPI application binds to.",
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="TCP port the FastAPI application listens on.",
    )

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

    # --- Environment tag (informational) ---
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment tag used for log correlation.",
    )

    api_key: str | None = Field(
        default=None,
        validation_alias="API_KEY",
        description=(
            "Optional Bearer token for API authentication. "
            "When set, all requests except health/docs must include "
            "'Authorization: Bearer <token>'. Leave empty to disable."
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
            max_tokens=self.max_tokens,
            openai_api_key=self.openai_api_key,
            openai_model=self.openai_model,
            google_api_key=self.google_api_key,
            google_model=self.google_model,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_region=self.aws_region,
            bedrock_model=self.bedrock_model,
            azure_openai_api_key=self.azure_openai_api_key,
            azure_openai_endpoint=self.azure_openai_endpoint,
            azure_openai_deployment=self.azure_openai_deployment,
            ollama_base_url=self.ollama_base_url,
            ollama_model=self.ollama_model,
        )

    @field_validator("redis_url")
    @classmethod
    def redis_url_scheme(cls, v: str) -> str:
        """Ensure the Redis URL uses a recognised scheme."""
        if not (v.startswith("redis://") or v.startswith("rediss://")):
            raise ValueError(
                "redis_url must start with 'redis://' or 'rediss://' — " f"got: {v!r}"
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
