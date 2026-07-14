"""core/embeddings.py — Provider-agnostic embeddings factory for RAG.

Mirrors :mod:`core.llm`: callers never import provider SDKs directly.
Selection is driven by ``EMBEDDING_PROVIDER`` (or ``auto`` derived from
``LLM_PROVIDER``).
"""

from __future__ import annotations

from typing import Literal

from langchain_core.embeddings import Embeddings

from core.config import Settings

EmbeddingProvider = Literal["auto", "openai", "azure", "google", "ollama", "mock"]

_DEFAULT_MODELS: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "azure": "text-embedding-3-small",
    "google": "models/text-embedding-004",
    "ollama": "nomic-embed-text",
}


def resolve_embedding_provider(settings: Settings) -> str:
    """Resolve the effective embedding provider from settings.

    ``auto`` follows ``LLM_PROVIDER`` when that provider has a first-party
    embeddings integration. Anthropic / Bedrock have none in this stack —
    in non-production we fall back to ``mock`` so local RAG experiments work
    without an API key; production requires an explicit ``EMBEDDING_PROVIDER``.
    """
    configured = settings.embedding_provider
    if configured != "auto":
        return configured

    llm = settings.llm_provider
    if llm == "mock":
        return "mock"
    if llm in ("openai", "azure", "google", "ollama"):
        return llm
    if settings.environment == "production":
        raise ValueError(
            f"EMBEDDING_PROVIDER=auto cannot derive embeddings from "
            f"LLM_PROVIDER={llm!r}. Set EMBEDDING_PROVIDER to openai, azure, "
            "google, ollama, or mock."
        )
    return "mock"


def get_embeddings(settings: Settings) -> Embeddings:
    """Return an embeddings model for the configured provider.

    Args:
        settings: Application settings (reads embedding_* and LLM credentials).

    Returns:
        A LangChain ``Embeddings`` instance.

    Raises:
        ValueError: When provider resolution fails or required credentials
            are missing.
        ImportError: When the provider package is not installed.
    """
    provider = resolve_embedding_provider(settings)
    model = settings.embedding_model or _DEFAULT_MODELS.get(provider)
    dimensions = settings.embedding_dimensions

    match provider:
        case "mock":
            from langchain_core.embeddings import DeterministicFakeEmbedding

            return DeterministicFakeEmbedding(size=dimensions)

        case "openai":
            try:
                from langchain_openai import OpenAIEmbeddings
            except ImportError as exc:
                raise ImportError(
                    "OpenAI embeddings require langchain-openai. "
                    "Install with: uv sync --extra openai"
                ) from exc
            if not settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
                )
            kwargs: dict = {
                "api_key": settings.openai_api_key,
                "model": model or _DEFAULT_MODELS["openai"],
            }
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            return OpenAIEmbeddings(**kwargs)

        case "azure":
            try:
                from langchain_openai import AzureOpenAIEmbeddings
            except ImportError as exc:
                raise ImportError(
                    "Azure embeddings require langchain-openai. "
                    "Install with: uv sync --extra openai"
                ) from exc
            if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
                raise ValueError(
                    "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are required "
                    "when EMBEDDING_PROVIDER=azure"
                )
            return AzureOpenAIEmbeddings(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                azure_deployment=settings.azure_openai_deployment,
                model=model or _DEFAULT_MODELS["azure"],
            )

        case "google":
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
            except ImportError as exc:
                raise ImportError(
                    "Google embeddings require langchain-google-genai. "
                    "Install with: uv sync --extra google"
                ) from exc
            if not settings.google_api_key:
                raise ValueError(
                    "GOOGLE_API_KEY is required when EMBEDDING_PROVIDER=google"
                )
            return GoogleGenerativeAIEmbeddings(
                google_api_key=settings.google_api_key,
                model=model or _DEFAULT_MODELS["google"],
            )

        case "ollama":
            try:
                from langchain_ollama import OllamaEmbeddings
            except ImportError as exc:
                raise ImportError(
                    "Ollama embeddings require langchain-ollama. "
                    "Install with: uv sync --extra ollama"
                ) from exc
            return OllamaEmbeddings(
                model=model or _DEFAULT_MODELS["ollama"],
                base_url=settings.ollama_base_url,
            )

        case _:
            raise ValueError(f"Unknown embedding provider: {provider!r}")
