"""tests/test_embeddings.py — Unit tests for core/embeddings.py."""

from __future__ import annotations

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding


def test_resolve_auto_openai() -> None:
    from core.config import Settings
    from core.embeddings import resolve_embedding_provider

    settings = Settings(
        llm_provider="openai",
        openai_api_key="sk-test",
        embedding_provider="auto",
        memory_backend="sqlite",
        sqlite_path=":memory:",
        environment="development",
    )
    assert resolve_embedding_provider(settings) == "openai"


def test_resolve_auto_anthropic_falls_back_to_mock_outside_production() -> None:
    from core.config import Settings
    from core.embeddings import resolve_embedding_provider

    settings = Settings(
        llm_provider="anthropic",
        anthropic_api_key="sk-ant-test123456789012345",
        embedding_provider="auto",
        memory_backend="sqlite",
        sqlite_path=":memory:",
        environment="development",
    )
    assert resolve_embedding_provider(settings) == "mock"


def test_resolve_auto_anthropic_raises_in_production() -> None:
    from core.config import Settings
    from core.embeddings import resolve_embedding_provider

    settings = Settings(
        llm_provider="anthropic",
        anthropic_api_key="sk-ant-test123456789012345",
        embedding_provider="auto",
        memory_backend="sqlite",
        sqlite_path=":memory:",
        environment="production",
        api_key="test-api-key-for-production",
    )
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        resolve_embedding_provider(settings)


def test_get_embeddings_mock_returns_deterministic() -> None:
    from core.config import Settings
    from core.embeddings import get_embeddings

    settings = Settings(
        llm_provider="mock",
        embedding_provider="mock",
        embedding_dimensions=16,
        memory_backend="sqlite",
        sqlite_path=":memory:",
        environment="development",
    )
    emb = get_embeddings(settings)
    assert isinstance(emb, DeterministicFakeEmbedding)
    assert len(emb.embed_query("x")) == 16
