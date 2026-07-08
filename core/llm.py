"""
core/llm.py — Provider-agnostic LLM factory for the LangGraph agent stack.

Usage:
    from core.llm import get_llm, LLMConfig
    llm = get_llm(settings.llm_config)
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

LLMProvider = Literal[
    "anthropic", "openai", "google", "bedrock", "azure", "ollama", "mock"
]


class LLMConfig(BaseModel):
    """Configuration for the LLM provider. All fields have sensible defaults."""

    provider: LLMProvider = "anthropic"
    request_timeout_seconds: float = Field(
        default=120.0,
        ge=1.0,
        description="Wall-clock timeout per LLM HTTP request (seconds).",
    )
    # Anthropic
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"
    anthropic_base_url: str | None = Field(
        default=None,
        description=(
            "Optional Anthropic API base URL (LiteLLM, Helicone, internal gateway). "
            "Maps to LangChain ``base_url`` / ``anthropic_api_url``."
        ),
    )
    max_tokens: int = 4096
    # OpenAI
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.5"
    openai_base_url: str | None = Field(
        default=None,
        description=(
            "Optional OpenAI-compatible API base URL (LiteLLM, OpenRouter, "
            "Together AI, Anyscale, internal gateway)."
        ),
    )
    # Google
    google_api_key: str | None = None
    google_model: str = "gemini-3.5-flash"
    google_base_url: str | None = Field(
        default=None,
        description="Optional Google Generative AI API base URL override.",
    )
    # AWS Bedrock
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "us-east-1"
    bedrock_model: str = "anthropic.claude-sonnet-5"
    bedrock_endpoint_url: str | None = Field(
        default=None,
        description=(
            "Optional Bedrock Runtime endpoint URL (VPC endpoints, custom proxy)."
        ),
    )
    # Azure OpenAI
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_base_url: str | None = Field(
        default=None,
        description=(
            "Optional Azure OpenAI HTTP base URL override (gateway / private link proxy)."
        ),
    )
    azure_openai_deployment: str = "gpt-4o"
    # Ollama (no API key required — local only)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"


# Vendor SDK auto-retries are disabled in :func:`get_llm`. Transient LLM failures
# are retried once in ``BaseAgent._invoke_llm_with_retry`` (tenacity) so workers
# are not blocked by stacked SDK × application backoff (e.g. Anthropic 2 × app 3).
_SDK_MAX_RETRIES = 0


def _set_optional_url(kwargs: dict[str, Any], key: str, value: str | None) -> None:
    """Attach a URL override when the caller provided a non-empty string."""
    if value:
        kwargs[key] = value


def _bedrock_chat_kwargs(config: LLMConfig) -> dict[str, Any]:
    """Build ``ChatBedrock`` kwargs using the default AWS credential chain when possible.

    On EKS, omit static keys so boto3 resolves IRSA / Pod Identity via
    ``AWS_WEB_IDENTITY_TOKEN_FILE``. Static ``AWS_ACCESS_KEY_ID`` /
    ``AWS_SECRET_ACCESS_KEY`` remain supported for local development.
    """
    from botocore.config import Config

    if config.aws_access_key_id and not config.aws_secret_access_key:
        raise ValueError(
            "AWS_SECRET_ACCESS_KEY must be set when AWS_ACCESS_KEY_ID is provided "
            "for the 'bedrock' provider."
        )
    if config.aws_secret_access_key and not config.aws_access_key_id:
        raise ValueError(
            "AWS_ACCESS_KEY_ID must be set when AWS_SECRET_ACCESS_KEY is provided "
            "for the 'bedrock' provider."
        )

    kwargs: dict[str, Any] = {
        "model_id": config.bedrock_model,
        "region_name": config.aws_region,
        "max_tokens": config.max_tokens,
        "credentials_profile_name": None,
        "config": Config(
            read_timeout=int(config.request_timeout_seconds),
            connect_timeout=min(10, int(config.request_timeout_seconds)),
            retries={"max_attempts": 1, "mode": "standard"},
        ),
    }
    if config.aws_access_key_id and config.aws_secret_access_key:
        kwargs["aws_access_key_id"] = config.aws_access_key_id
        kwargs["aws_secret_access_key"] = config.aws_secret_access_key
    _set_optional_url(kwargs, "endpoint_url", config.bedrock_endpoint_url)
    return kwargs


def get_llm(config: LLMConfig) -> BaseChatModel:
    """Instantiate and return a LangChain chat model for the configured provider.

    Each provider's integration package is imported lazily so that only the
    package actually used needs to be installed.  Install extras with:

        uv sync --extra anthropic
        uv sync --extra openai
        uv sync --extra google
        uv sync --extra bedrock
        uv sync --extra openai   (Azure uses the openai extra)
        uv sync --extra ollama
        uv sync   (mock requires no extra — included in langchain-core)

    Args:
        config: An :class:`LLMConfig` instance describing which provider and
            model to use together with the associated credentials.

    Returns:
        A :class:`~langchain_core.language_models.BaseChatModel` ready for
        invocation.

    Raises:
        ImportError: If the required integration package is not installed.
        ValueError: If a required API key is missing or the provider is unknown.
    """
    match config.provider:
        case "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError as e:
                raise ImportError("Install with: uv sync --extra anthropic") from e
            if not config.anthropic_api_key:
                raise ValueError(
                    "anthropic_api_key is required for the 'anthropic' provider."
                )
            return ChatAnthropic(
                model=config.anthropic_model,
                api_key=config.anthropic_api_key,
                max_tokens=config.max_tokens,
                max_retries=_SDK_MAX_RETRIES,
                default_request_timeout=config.request_timeout_seconds,
                **(
                    {"base_url": config.anthropic_base_url}
                    if config.anthropic_base_url
                    else {}
                ),
            )

        case "openai":
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as e:
                raise ImportError("Install with: uv sync --extra openai") from e
            if not config.openai_api_key:
                raise ValueError(
                    "openai_api_key is required for the 'openai' provider."
                )
            return ChatOpenAI(
                model=config.openai_model,
                api_key=config.openai_api_key,
                max_tokens=config.max_tokens,
                max_retries=_SDK_MAX_RETRIES,
                request_timeout=config.request_timeout_seconds,
                **(
                    {"base_url": config.openai_base_url}
                    if config.openai_base_url
                    else {}
                ),
            )

        case "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as e:
                raise ImportError("Install with: uv sync --extra google") from e
            if not config.google_api_key:
                raise ValueError(
                    "google_api_key is required for the 'google' provider."
                )
            return ChatGoogleGenerativeAI(
                model=config.google_model,
                google_api_key=config.google_api_key,
                max_output_tokens=config.max_tokens,
                max_retries=_SDK_MAX_RETRIES,
                timeout=config.request_timeout_seconds,
                **(
                    {"base_url": config.google_base_url}
                    if config.google_base_url
                    else {}
                ),
            )

        case "bedrock":
            try:
                from langchain_aws import ChatBedrock
            except ImportError as e:
                raise ImportError("Install with: uv sync --extra bedrock") from e
            return ChatBedrock(**_bedrock_chat_kwargs(config))

        case "azure":
            try:
                from langchain_openai import AzureChatOpenAI
            except ImportError as e:
                raise ImportError("Install with: uv sync --extra openai") from e
            if not config.azure_openai_api_key:
                raise ValueError(
                    "azure_openai_api_key is required for the 'azure' provider."
                )
            if not config.azure_openai_endpoint:
                raise ValueError(
                    "azure_openai_endpoint is required for the 'azure' provider."
                )
            return AzureChatOpenAI(
                azure_deployment=config.azure_openai_deployment,
                api_key=config.azure_openai_api_key,
                azure_endpoint=config.azure_openai_endpoint,
                max_tokens=config.max_tokens,
                max_retries=_SDK_MAX_RETRIES,
                request_timeout=config.request_timeout_seconds,
                **(
                    {"base_url": config.azure_openai_base_url}
                    if config.azure_openai_base_url
                    else {}
                ),
            )

        case "ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError as e:
                raise ImportError("Install with: uv sync --extra ollama") from e
            return ChatOllama(
                model=config.ollama_model,
                base_url=config.ollama_base_url,
                num_predict=config.max_tokens,
                sync_client_kwargs={"timeout": config.request_timeout_seconds},
                async_client_kwargs={"timeout": config.request_timeout_seconds},
            )

        case "mock":
            from core.mock_llm import MockProviderChatModel

            return MockProviderChatModel()

        case _:
            raise ValueError(
                f"Unknown LLM provider: {config.provider!r}. "
                "Valid providers: anthropic | openai | google | bedrock | azure | ollama | mock"
            )


# LangChain constructors diverge from pyright stubs across releases.
# pyright: reportCallIssue=false
