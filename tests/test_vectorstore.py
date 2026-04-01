"""tests/test_vectorstore.py — Unit tests for core/vectorstore.py."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestGetVectorstoreDisabled:
    def test_raises_when_rag_disabled(self, test_settings):
        from core.vectorstore import get_vectorstore

        # test_settings has rag_enabled=False by default
        with pytest.raises(RuntimeError, match="RAG"):
            get_vectorstore(test_settings)


class TestGetVectorstoreDisabledVariants:
    def test_rag_disabled_raises_runtime_error(self):
        from core.config import Settings
        from core.vectorstore import get_vectorstore

        settings = Settings(
            llm_provider="anthropic",
            anthropic_api_key="sk-ant-test123456789012345",
            memory_backend="sqlite",
            sqlite_path=":memory:",
            environment="development",
            rag_enabled=False,
        )
        with pytest.raises(RuntimeError):
            get_vectorstore(settings)

    def test_error_message_mentions_rag_enabled(self):
        from core.config import Settings
        from core.vectorstore import get_vectorstore

        settings = Settings(
            llm_provider="anthropic",
            anthropic_api_key="sk-ant-test123456789012345",
            memory_backend="sqlite",
            sqlite_path=":memory:",
            environment="development",
            rag_enabled=False,
        )
        with pytest.raises(RuntimeError, match="RAG_ENABLED"):
            get_vectorstore(settings)


class TestGetVectorstoreChromaImportError:
    def test_raises_import_error_when_chroma_missing(self):
        """When rag_enabled=True but langchain_chroma is not installed, ImportError is raised."""
        import sys
        from unittest.mock import patch

        from core.config import Settings
        from core.vectorstore import get_vectorstore

        settings = Settings(
            llm_provider="anthropic",
            anthropic_api_key="sk-ant-test123456789012345",
            memory_backend="sqlite",
            sqlite_path=":memory:",
            environment="development",
            rag_enabled=True,
        )

        # Remove langchain_chroma from sys.modules so the import inside get_vectorstore fails
        original = sys.modules.pop("langchain_chroma", None)
        try:
            with patch.dict("sys.modules", {"langchain_chroma": None}):
                with pytest.raises((ImportError, RuntimeError)):
                    get_vectorstore(settings)
        finally:
            if original is not None:
                sys.modules["langchain_chroma"] = original


class TestGetVectorstorePGVectorMissingUrl:
    def test_raises_validation_error_when_postgres_url_missing(self):
        """When memory_backend=postgres but postgres_url is not set, Settings validation fails."""
        from pydantic import ValidationError

        from core.config import Settings, get_settings

        env_override = {
            "RAG_ENABLED": "true",
            "MEMORY_BACKEND": "postgres",
            "LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "sk-ant-test123456789012345",
            "SQLITE_PATH": ":memory:",
            "ENVIRONMENT": "development",
        }
        with patch.dict(os.environ, env_override, clear=False):
            os.environ.pop("POSTGRES_URL", None)
            get_settings.cache_clear()
            try:
                with pytest.raises(ValidationError, match="POSTGRES_URL"):
                    Settings()  # type: ignore[call-arg]
            finally:
                get_settings.cache_clear()


class TestGetVectorstoreChromaSuccess:
    """Happy path: ChromaDB vector store created successfully with mocked import."""

    def test_returns_chroma_instance(self):
        from unittest.mock import MagicMock

        from core.vectorstore import get_vectorstore

        env_override = {
            "RAG_ENABLED": "true",
            "MEMORY_BACKEND": "sqlite",
            "LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "sk-ant-test123456789012345",
            "SQLITE_PATH": ":memory:",
            "ENVIRONMENT": "development",
        }

        mock_chroma_instance = MagicMock()
        mock_chroma_module = MagicMock()
        mock_chroma_module.Chroma.return_value = mock_chroma_instance

        with patch.dict(os.environ, env_override, clear=False):
            from core.config import Settings

            settings = Settings()  # type: ignore[call-arg]

            with patch.dict("sys.modules", {"langchain_chroma": mock_chroma_module}):
                result = get_vectorstore(settings)

        assert result is mock_chroma_instance
        mock_chroma_module.Chroma.assert_called_once_with(
            collection_name="langgraph_rag"
        )


class TestGetVectorstorePGVectorSuccess:
    """Happy path: PGVector store created successfully with mocked import."""

    _PG_ENV = {
        "RAG_ENABLED": "true",
        "MEMORY_BACKEND": "postgres",
        "POSTGRES_URL": "postgresql://user:pass@localhost:5432/db",
        "LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "sk-ant-test123456789012345",
        "SQLITE_PATH": ":memory:",
        "ENVIRONMENT": "development",
    }

    def test_returns_pgvector_instance(self):
        from unittest.mock import MagicMock

        from core.vectorstore import get_vectorstore

        mock_pgvector_instance = MagicMock()
        mock_pgvector_module = MagicMock()
        mock_pgvector_module.PGVector.return_value = mock_pgvector_instance
        mock_community = MagicMock()
        mock_community.vectorstores = mock_pgvector_module

        with patch.dict(os.environ, self._PG_ENV, clear=False):
            from core.config import Settings

            settings = Settings()  # type: ignore[call-arg]

            with patch.dict(
                "sys.modules",
                {
                    "langchain_community": mock_community,
                    "langchain_community.vectorstores": mock_pgvector_module,
                },
            ):
                result = get_vectorstore(settings)

        assert result is mock_pgvector_instance
        mock_pgvector_module.PGVector.assert_called_once_with(
            collection_name="langgraph_rag",
            connection_string="postgresql://user:pass@localhost:5432/db",
        )

    def test_raises_runtime_error_when_postgres_url_empty(self):
        """_get_pgvector raises RuntimeError when postgres_url is None."""
        from core.vectorstore import _get_pgvector

        with patch.dict(os.environ, self._PG_ENV, clear=False):
            from core.config import Settings

            settings = Settings()  # type: ignore[call-arg]
            settings.postgres_url = None  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="POSTGRES_URL"):
            _get_pgvector(settings)

    def test_raises_import_error_when_langchain_community_missing(self):
        """_get_pgvector raises ImportError when langchain-community is not installed."""
        from core.vectorstore import _get_pgvector

        with patch.dict(os.environ, self._PG_ENV, clear=False):
            from core.config import Settings

            settings = Settings()  # type: ignore[call-arg]

        with patch.dict(
            "sys.modules",
            {
                "langchain_community": None,
                "langchain_community.vectorstores": None,
            },
        ):
            with pytest.raises(ImportError, match="langchain-community"):
                _get_pgvector(settings)
