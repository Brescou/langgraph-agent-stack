"""
core/vectorstore.py — Pluggable vector store abstraction for RAG.

Dev backend  : ChromaDB in-memory (no external service required).
Prod backend : pgvector via ``langchain_postgres.PGVector``
               (requires MEMORY_BACKEND=postgres).

Only active when RAG_ENABLED=true. Raises RuntimeError otherwise.

Usage::

    from core.vectorstore import get_vectorstore
    from core.config import get_settings

    vs = get_vectorstore(get_settings())
    docs = vs.similarity_search("quantum computing", k=5)

Backend selection
-----------------
+-------------------------------+--------------------------------------+
| Condition                     | Backend returned                     |
+===============================+======================================+
| ``rag_enabled=False``         | RuntimeError raised                  |
+-------------------------------+--------------------------------------+
| ``rag_enabled=True``          | ChromaDB in-memory (default)         |
+-------------------------------+--------------------------------------+
| ``rag_enabled=True`` +        | PGVector (langchain-postgres)        |
| ``memory_backend=postgres``   |                                      |
+-------------------------------+--------------------------------------+
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core.config import MemoryBackend, Settings
from core.embeddings import get_embeddings


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """
    Minimal interface that every vector store backend must satisfy.

    Both ChromaDB and PGVector implement this surface, so callers can
    treat them interchangeably through this protocol.
    """

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Index a list of LangChain ``Document`` objects.

        Args:
            documents: Documents to embed and store.

        Returns:
            A list of string IDs assigned to the stored documents.
        """
        ...

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """
        Return the ``k`` most semantically similar documents for ``query``.

        Args:
            query: Free-text search string.
            k: Number of results to return (default: 5).

        Returns:
            Ordered list of the closest ``Document`` objects.
        """
        ...


def get_vectorstore(settings: Settings) -> VectorStoreProtocol:
    """
    Factory that returns the appropriate vector store based on ``settings``.

    The function inspects ``settings.rag_enabled`` and
    ``settings.memory_backend`` to select the correct backend:

    * ``rag_enabled=False`` → raises ``RuntimeError`` immediately; callers
      must check the flag before calling this function.
    * ``rag_enabled=True`` + default/sqlite → ChromaDB in-memory collection
      named ``langgraph_rag``.  Requires the ``langchain-chroma`` and
      ``chromadb`` packages (installed via ``uv sync --extra rag``).
    * ``rag_enabled=True`` + ``memory_backend=postgres`` → PGVector backed by
      ``settings.postgres_url``.  Requires ``langchain-postgres``
      (installed via ``uv sync --extra postgres``).

    Every backend receives an explicit embeddings instance from
    :func:`core.embeddings.get_embeddings` (never a library default).

    Args:
        settings: The application ``Settings`` instance.  The function reads
            ``rag_enabled``, ``memory_backend``, embedding settings, and
            optionally ``postgres_url``.

    Returns:
        A configured object that satisfies :class:`VectorStoreProtocol`.

    Raises:
        RuntimeError: When ``settings.rag_enabled`` is ``False``.
        ImportError: When the required backend package is not installed,
            with a hint for the correct install command.
    """
    if not settings.rag_enabled:
        raise RuntimeError(
            "RAG is disabled. Set RAG_ENABLED=true in your .env to enable."
        )

    embeddings = get_embeddings(settings)
    use_postgres = settings.memory_backend == MemoryBackend.POSTGRES

    if use_postgres:
        return _get_pgvector(settings, embeddings)

    return _get_chromadb(embeddings)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


def _get_chromadb(embeddings: Embeddings) -> VectorStoreProtocol:
    """
    Return a ChromaDB in-memory vector store with an explicit embedding model.

    Uses the ``langchain-chroma`` integration and the default in-process
    ephemeral client.  Data is lost when the process exits — suitable for
    development and testing.

    Args:
        embeddings: Explicit embeddings instance (never rely on Chroma defaults).

    Returns:
        A :class:`langchain_chroma.Chroma` instance.

    Raises:
        ImportError: When ``langchain-chroma`` or ``chromadb`` is not installed.
    """
    try:
        from langchain_chroma import Chroma  # type: ignore[import]

        return Chroma(  # type: ignore[return-value]
            collection_name="langgraph_rag",
            embedding_function=embeddings,
        )

    except ImportError as exc:
        raise ImportError(
            "RAG support requires chromadb. Install with: uv sync --extra rag"
        ) from exc


def _get_pgvector(settings: Settings, embeddings: Embeddings) -> VectorStoreProtocol:
    """
    Return a PGVector store connected to ``settings.postgres_url``.

    Uses ``langchain_postgres.PGVector`` (maintained successor to the
    community class).

    Args:
        settings: Application settings; ``postgres_url`` must be a valid DSN.
        embeddings: Explicit embeddings instance.

    Returns:
        A ``langchain_postgres.PGVector`` instance.

    Raises:
        RuntimeError: When ``settings.postgres_url`` is ``None`` or empty.
        ImportError: When ``langchain-postgres`` is not installed.
    """
    postgres_url: str | None = settings.postgres_url
    if not postgres_url:
        raise RuntimeError(
            "PGVector requires POSTGRES_URL to be set. "
            "Provide a valid PostgreSQL DSN in your .env file."
        )

    try:
        from langchain_postgres import PGVector  # type: ignore[import]

        return PGVector(  # type: ignore[return-value]
            embeddings=embeddings,
            connection=postgres_url,
            collection_name="langgraph_rag",
        )

    except ImportError as exc:
        raise ImportError(
            "PGVector support requires langchain-postgres. "
            "Install with: uv sync --extra postgres"
        ) from exc
