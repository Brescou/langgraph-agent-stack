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


def test_chunk_id_uses_resolved_absolute_path(tmp_path: Path) -> None:
    from ingest.ids import chunk_id

    nested = tmp_path / "corpus" / "sub" / "doc.md"
    nested.parent.mkdir(parents=True)
    nested.write_text("x", encoding="utf-8")
    via_file = chunk_id(nested, 0)
    via_same = chunk_id(tmp_path / "corpus" / "sub" / ".." / "sub" / "doc.md", 0)
    assert via_file == via_same
    assert via_file.endswith(":0")
    assert ":" in via_file


def test_ingest_path_indexes_markdown_and_is_idempotent(tmp_path: Path) -> None:
    from ingest.pipeline import ingest_path

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "qubits.md").write_text(
        "Superconducting qubits are a leading quantum hardware approach.\n",
        encoding="utf-8",
    )
    (corpus / "ignore.bin").write_bytes(b"\x00\x01")
    settings = _rag_settings(tmp_path)
    first = ingest_path(corpus, settings)
    second = ingest_path(corpus, settings)
    assert first.files_read == 1
    assert first.chunks_upserted >= 1
    assert first.skipped == 1
    assert second.chunks_upserted == first.chunks_upserted
    store = get_vectorstore(settings)
    assert store.document_count() == first.chunks_upserted
    hits = store.similarity_search("superconducting qubits", k=1)
    assert "qubits" in hits[0].page_content.lower()


def test_ingest_file_and_parent_dir_do_not_duplicate(tmp_path: Path) -> None:
    from ingest.pipeline import ingest_path

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    doc = corpus / "qubits.md"
    doc.write_text("Ion trap qubits need ultra-high vacuum.\n", encoding="utf-8")
    settings = _rag_settings(tmp_path)
    ingest_path(corpus, settings)
    ingest_path(doc, settings)
    store = get_vectorstore(settings)
    assert store.document_count() == 1
