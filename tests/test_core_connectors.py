"""tests/test_core_connectors.py — Connector factory and settings integration."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from connectors.examples.example_connector import ExampleMemoryConnector
from connectors.http_connector import HttpConnector
from connectors.rag_connector import RagConnector
from connectors.resolver import create_connector, resolve_connector
from core.config import Settings, get_settings


def _settings_env(**env: str) -> Settings:
    base = {"LLM_PROVIDER": "mock", **env}
    with patch.dict(os.environ, base, clear=False):
        get_settings.cache_clear()
        return Settings()


def test_create_connector_example_memory() -> None:
    connector = create_connector("example_memory", _settings_env())
    assert isinstance(connector, ExampleMemoryConnector)


def test_create_connector_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown CONNECTOR_ID"):
        create_connector("does_not_exist", _settings_env())


def test_create_connector_http_requires_url() -> None:
    with pytest.raises(ValueError, match="CONNECTOR_HTTP_URL"):
        create_connector("http", _settings_env())


def test_create_connector_http() -> None:
    settings = _settings_env(CONNECTOR_HTTP_URL="https://api.example.com/search")
    with patch(
        "core.security.socket.getaddrinfo",
        return_value=[
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ],
    ):
        connector = create_connector("http", settings)
    assert isinstance(connector, HttpConnector)


def test_resolve_connector_disabled() -> None:
    assert resolve_connector(_settings_env(CONNECTOR_ENABLED="false")) is None


def test_resolve_connector_enabled() -> None:
    settings = _settings_env(
        CONNECTOR_ENABLED="true",
        CONNECTOR_ID="example_memory",
    )
    connector = resolve_connector(settings)
    assert isinstance(connector, ExampleMemoryConnector)


def test_settings_rejects_unknown_connector_id() -> None:
    with pytest.raises(ValidationError, match="CONNECTOR_ID"):
        _settings_env(
            CONNECTOR_ENABLED="true",
            CONNECTOR_ID="unknown_connector",
        )


def test_settings_rag_connector_requires_rag_enabled() -> None:
    with pytest.raises(ValidationError, match="RAG_ENABLED"):
        _settings_env(
            CONNECTOR_ENABLED="true",
            CONNECTOR_ID="rag",
            RAG_ENABLED="false",
        )


def test_create_rag_connector() -> None:
    settings = _settings_env(
        RAG_ENABLED="true",
        CONNECTOR_ENABLED="true",
        CONNECTOR_ID="rag",
    )
    connector = create_connector("rag", settings)
    assert isinstance(connector, RagConnector)


@pytest.mark.asyncio
async def test_rag_connector_fetch() -> None:
    mock_doc = MagicMock()
    mock_doc.page_content = "vector hit"
    mock_doc.metadata = {"source": "doc-1"}
    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [mock_doc]

    settings = _settings_env(RAG_ENABLED="true")
    with patch("core.vectorstore.get_vectorstore", return_value=mock_store):
        connector = RagConnector(settings=settings)
        from connectors.base import ConnectorRequest

        result = await connector.fetch(ConnectorRequest(query="quantum"))

    assert result.records[0]["snippet"] == "vector hit"
