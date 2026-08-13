"""tests/test_ingest.py — RAG ingest pipeline and vector-store upsert."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

langchain_chroma = pytest.importorskip("langchain_chroma")

from core.config import Settings
from core.vectorstore import get_vectorstore


def _rag_settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_provider="mock",
        embedding_provider="mock",
        memory_backend="sqlite",
        sqlite_path=":memory:",
        environment="development",
        rag_enabled=True,
        rag_persist_dir=tmp_path / "chroma",
    )


def test_add_documents_with_ids_upserts_without_duplicating(tmp_path: Path) -> None:
    settings = _rag_settings(tmp_path)
    store = get_vectorstore(settings)
    docs = [
        Document(page_content="quantum qubits", metadata={"source": "a.md"}),
    ]
    store.add_documents(docs, ids=["abc123:0"])
    store.add_documents(
        [Document(page_content="quantum qubits updated", metadata={"source": "a.md"})],
        ids=["abc123:0"],
    )
    assert store.document_count() == 1
    hits = store.similarity_search("quantum qubits", k=1)
    assert hits[0].page_content == "quantum qubits updated"
