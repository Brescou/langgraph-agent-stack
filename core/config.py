"""
core/config.py — Application settings and configuration management.

Loads all configuration from environment variables / .env file using
pydantic-settings.  Never hard-code secrets here; use a .env file or the
shell environment instead.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

import re

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MemoryBackend(str, Enum):
    """Supported memory/persistence backends."""

    SQLITE = "sqlite"
    REDIS = "redis"


class LogLevel(str, Enum):
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
        model_name: LLM model identifier.
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
    anthropic_api_key: str = Field(
        ...,
        description="Anthropic API key — must be set in the environment or .env.",
    )
    model_name: str = Field(
        default="claude-sonnet-4-5",
        description="Anthropic model identifier passed to ChatAnthropic.",
    )
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

    # --- Environment tag (informational) ---
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment tag used for log correlation.",
    )

    @field_validator("anthropic_api_key")
    @classmethod
    def anthropic_key_format(cls, v: str) -> str:
        """
        Warn early when the API key does not match the expected Anthropic format.

        The key must begin with ``sk-ant-`` followed by at least 10 characters
        of alphanumerics, hyphens, or underscores.  A misconfigured placeholder
        (e.g. the example value from ``.env.example``) will be rejected at
        settings-load time rather than producing a cryptic Anthropic SDK error
        at runtime.
        """
        _pattern = re.compile(r"^sk-ant-[A-Za-z0-9\-_]{10,}$")
        if not _pattern.match(v):
            raise ValueError(
                "anthropic_api_key does not match the expected Anthropic format "
                "(must start with 'sk-ant-' followed by at least 10 characters). "
                "Check that ANTHROPIC_API_KEY is set correctly in your environment."
            )
        return v

    @field_validator("redis_url")
    @classmethod
    def redis_url_scheme(cls, v: str) -> str:
        """Ensure the Redis URL uses a recognised scheme."""
        if not (v.startswith("redis://") or v.startswith("rediss://")):
            raise ValueError(
                "redis_url must start with 'redis://' or 'rediss://' — "
                f"got: {v!r}"
            )
        return v


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere instead of re-instantiating.
# ---------------------------------------------------------------------------

settings = Settings()  # type: ignore[call-arg]  # key comes from env
