"""
connectors/rag_connector.py — Vector-store retrieval via core.vectorstore.

Requires ``RAG_ENABLED=true`` and the ``rag`` optional dependencies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from connectors.base import BaseConnector, ConnectorRequest, ConnectorResult

if TYPE_CHECKING:
    from core.config import Settings

logger = logging.getLogger(__name__)


class RagConnector(BaseConnector):
    """Runs similarity search against the configured RAG vector store."""

    connector_id: ClassVar[str] = "rag"
    name: ClassVar[str] = "RAG vector store connector"
    description: ClassVar[str] = (
        "Similarity search via get_vectorstore() when RAG_ENABLED=true."
    )

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def fetch(self, request: ConnectorRequest) -> ConnectorResult:
        from core.vectorstore import get_vectorstore

        store = get_vectorstore(self._settings)
        documents = store.similarity_search(request.query, k=request.limit)
        records: list[dict[str, Any]] = []
        for index, doc in enumerate(documents):
            meta = dict(doc.metadata) if doc.metadata else {}
            records.append(
                {
                    "source": str(meta.get("source", f"rag:{index}")),
                    "snippet": doc.page_content,
                    "score": meta.get("score"),
                }
            )
        metadata: dict[str, Any] = {"backend": "rag", "empty_store": False}
        if not records:
            count = store.document_count()
            metadata["count"] = count
            metadata["empty_store"] = count == 0
            if count == 0:
                logger.warning(
                    "RAG vector store is empty; ingest documents with "
                    "`python -m ingest <path>`"
                )
        return ConnectorResult(records=tuple(records), metadata=metadata)
