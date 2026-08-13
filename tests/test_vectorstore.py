"""tests/test_vectorstore.py — Unit tests for core/vectorstore.py."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding


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

        from core.config import Settings
        from core.vectorstore import get_vectorstore

        settings = Settings(
            llm_provider="mock",
            memory_backend="sqlite",
            sqlite_path=":memory:",
            environment="development",
            rag_enabled=True,
            embedding_provider="mock",
        )

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
            "LLM_PROVIDER": "mock",
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
    """Happy path: ChromaDB vector store created with an explicit embedding function."""

    def test_returns_chroma_instance_with_explicit_embeddings(self):
        from core.vectorstore import _ProtocolVectorStore, get_vectorstore

        env_override = {
            "RAG_ENABLED": "true",
            "MEMORY_BACKEND": "sqlite",
            "LLM_PROVIDER": "mock",
            "EMBEDDING_PROVIDER": "mock",
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

        assert isinstance(result, _ProtocolVectorStore)
        assert result._inner is mock_chroma_instance
        mock_chroma_module.Chroma.assert_called_once()
        kwargs = mock_chroma_module.Chroma.call_args.kwargs
        assert kwargs["collection_name"] == "langgraph_rag"
        assert kwargs["persist_directory"] == str(settings.rag_persist_dir)
        assert isinstance(kwargs["embedding_function"], DeterministicFakeEmbedding)


class TestGetVectorstorePGVectorSuccess:
    """Happy path: langchain_postgres.PGVector with explicit embeddings."""

    _PG_ENV = {
        "RAG_ENABLED": "true",
        "MEMORY_BACKEND": "postgres",
        "POSTGRES_URL": "postgresql+psycopg://user:pass@localhost:5432/db",
        "LLM_PROVIDER": "mock",
        "EMBEDDING_PROVIDER": "mock",
        "SQLITE_PATH": ":memory:",
        "ENVIRONMENT": "development",
    }

    def test_returns_pgvector_instance(self):
        from core.vectorstore import _ProtocolVectorStore, get_vectorstore

        mock_pgvector_instance = MagicMock()
        mock_pg_module = MagicMock()
        mock_pg_module.PGVector.return_value = mock_pgvector_instance

        with patch.dict(os.environ, self._PG_ENV, clear=False):
            from core.config import Settings

            settings = Settings()  # type: ignore[call-arg]

            with patch.dict(
                "sys.modules",
                {"langchain_postgres": mock_pg_module},
            ):
                result = get_vectorstore(settings)

        assert isinstance(result, _ProtocolVectorStore)
        assert result._inner is mock_pgvector_instance
        mock_pg_module.PGVector.assert_called_once()
        kwargs = mock_pg_module.PGVector.call_args.kwargs
        assert kwargs["collection_name"] == "langgraph_rag"
        assert (
            kwargs["connection"] == "postgresql+psycopg://user:pass@localhost:5432/db"
        )
        assert isinstance(kwargs["embeddings"], DeterministicFakeEmbedding)

    def test_raises_runtime_error_when_postgres_url_empty(self):
        """_get_pgvector raises RuntimeError when postgres_url is None."""
        from core.embeddings import get_embeddings
        from core.vectorstore import _get_pgvector

        with patch.dict(os.environ, self._PG_ENV, clear=False):
            from core.config import Settings

            settings = Settings()  # type: ignore[call-arg]
            settings.postgres_url = None  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="POSTGRES_URL"):
            _get_pgvector(settings, get_embeddings(settings))

    def test_raises_import_error_when_langchain_postgres_missing(self):
        """_get_pgvector raises ImportError when langchain-postgres is not installed."""
        from core.embeddings import get_embeddings
        from core.vectorstore import _get_pgvector

        with patch.dict(os.environ, self._PG_ENV, clear=False):
            from core.config import Settings

            settings = Settings()  # type: ignore[call-arg]

        with patch.dict("sys.modules", {"langchain_postgres": None}):
            with pytest.raises(ImportError, match="langchain-postgres"):
                _get_pgvector(settings, get_embeddings(settings))


class TestMockEmbeddingsDeterminism:
    def test_mock_provider_embeddings_are_deterministic(self):
        from core.config import Settings
        from core.embeddings import get_embeddings

        settings = Settings(
            llm_provider="mock",
            embedding_provider="mock",
            embedding_dimensions=32,
            memory_backend="sqlite",
            sqlite_path=":memory:",
            environment="development",
            rag_enabled=True,
        )
        emb = get_embeddings(settings)
        assert emb.embed_query("alpha") == emb.embed_query("alpha")
        assert emb.embed_query("alpha") != emb.embed_query("beta")

    def test_llm_provider_mock_auto_selects_mock_embeddings(self):
        from core.config import Settings
        from core.embeddings import resolve_embedding_provider

        settings = Settings(
            llm_provider="mock",
            embedding_provider="auto",
            memory_backend="sqlite",
            sqlite_path=":memory:",
            environment="development",
        )
        assert resolve_embedding_provider(settings) == "mock"

    def test_no_todo_comments_remain_in_vectorstore(self):
        from pathlib import Path

        source = Path("core/vectorstore.py").read_text(encoding="utf-8")
        assert "TODO" not in source


class _CollectionIdColumn:
    """Stand-in for SQLAlchemy ``EmbeddingStore.collection_id`` equality."""

    def __eq__(self, other: object) -> object:
        return ("collection_id_eq", other)


class _FakeEmbeddingStore:
    collection_id = _CollectionIdColumn()


class _SessionContext:
    """Context manager returned by ``session_maker()``."""

    def __init__(self, session: object) -> None:
        self._session = session

    def __enter__(self) -> object:
        return self._session

    def __exit__(self, *args: object) -> bool:
        return False


class _FakePGVectorInner:
    """Plain PGVector-shaped object: no ``_collection``, no ``get``."""

    EmbeddingStore = _FakeEmbeddingStore

    def __init__(self, session: object, collection: object | None) -> None:
        self._session = session
        self._pg_collection = collection

    def session_maker(self) -> _SessionContext:
        return _SessionContext(self._session)

    def get_collection(self, session: object) -> object | None:
        return self._pg_collection


class TestProtocolVectorStoreDocumentCount:
    def test_document_count_uses_pgvector_session_when_no_collection_attr(self):
        from core.vectorstore import _ProtocolVectorStore

        session = MagicMock()
        session.query.return_value.filter.return_value.count.return_value = 7
        collection = type("Collection", (), {"uuid": "col-1"})()
        inner = _FakePGVectorInner(session=session, collection=collection)
        store = _ProtocolVectorStore(inner)

        assert store.document_count() == 7
        session.query.assert_called_once_with(inner.EmbeddingStore)
        session.query.return_value.filter.assert_called_once_with(
            inner.EmbeddingStore.collection_id == collection.uuid
        )

    def test_document_count_returns_zero_when_pgvector_collection_missing(self):
        from core.vectorstore import _ProtocolVectorStore

        session = MagicMock()
        inner = _FakePGVectorInner(session=session, collection=None)
        store = _ProtocolVectorStore(inner)

        assert store.document_count() == 0
        session.query.assert_not_called()

    def test_document_count_uses_chroma_collection_count_when_present(self):
        from core.vectorstore import _ProtocolVectorStore

        inner = type("ChromaLike", (), {})()
        inner._collection = MagicMock()
        inner._collection.count.return_value = 3
        store = _ProtocolVectorStore(inner)

        assert store.document_count() == 3
