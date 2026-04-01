"""
core/llm.py — Provider-agnostic LLM factory for the LangGraph agent stack.

Usage:
    from core.llm import get_llm, LLMConfig
    llm = get_llm(settings.llm_config)
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

LLMProvider = Literal["anthropic", "openai", "google", "bedrock", "azure", "ollama"]


class LLMConfig(BaseModel):
    """Configuration for the LLM provider. All fields have sensible defaults."""

    provider: LLMProvider = "anthropic"
    # Anthropic
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4096
    # OpenAI
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    # Google
    google_api_key: str | None = None
    google_model: str = "gemini-1.5-pro"
    # AWS Bedrock
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "us-east-1"
    bedrock_model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    # Azure OpenAI
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_deployment: str = "gpt-4o"
    # Ollama (no API key required — local only)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"


def get_llm(config: LLMConfig) -> BaseChatModel:
    """Instantiate and return a LangChain chat model for the configured provider.

    Each provider's integration package is imported lazily so that only the
    package actually used needs to be installed.  Install extras with:

        uv sync --extra anthropic
        uv sync --extra openai
        uv sync --extra google
        uv sync --extra bedrock
        uv sync --extra azure
        uv sync --extra ollama

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
            )

        case "bedrock":
            try:
                from langchain_aws import ChatBedrock
            except ImportError as e:
                raise ImportError("Install with: uv sync --extra bedrock") from e
            if not config.aws_access_key_id:
                raise ValueError(
                    "aws_access_key_id is required for the 'bedrock' provider."
                )
            if not config.aws_secret_access_key:
                raise ValueError(
                    "aws_secret_access_key is required for the 'bedrock' provider."
                )
            return ChatBedrock(
                model_id=config.bedrock_model,
                region_name=config.aws_region,
                max_tokens=config.max_tokens,
            )

        case "azure":
            try:
                from langchain_openai import AzureChatOpenAI
            except ImportError as e:
                raise ImportError("Install with: uv sync --extra azure") from e
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
            )

        case _:
            raise ValueError(
                f"Unknown LLM provider: {config.provider!r}. "
                "Valid providers: anthropic | openai | google | bedrock | azure | ollama"
            )
