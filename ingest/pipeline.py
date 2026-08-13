"""Load, chunk, and upsert local documents into get_vectorstore()."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import Settings
from core.vectorstore import get_vectorstore
from ingest.ids import chunk_id

logger = logging.getLogger(__name__)

_TEXT_SUFFIXES = {".md", ".txt"}


@dataclass(frozen=True, slots=True)
class IngestReport:
    files_read: int
    chunks_upserted: int
    skipped: int
    backend: str


def ingest_path(
    path: Path,
    settings: Settings,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> IngestReport:
    target = path.expanduser()
    if not target.exists():
        raise FileNotFoundError(target)

    size = chunk_size if chunk_size is not None else settings.rag_chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.rag_chunk_overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    files, skipped = _collect_files(target)
    documents: list[Document] = []
    ids: list[str] = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        resolved = str(file_path.resolve())
        for index, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": resolved,
                        "path": resolved,
                        "chunk_index": index,
                    },
                )
            )
            ids.append(chunk_id(file_path, index))

    store = get_vectorstore(settings)
    if documents:
        store.add_documents(documents, ids=ids)
    backend = "pgvector" if settings.memory_backend.value == "postgres" else "chroma"
    return IngestReport(
        files_read=len(files),
        chunks_upserted=len(documents),
        skipped=skipped,
        backend=backend,
    )


def _collect_files(target: Path) -> tuple[list[Path], int]:
    skipped = 0
    files: list[Path] = []
    candidates = [target] if target.is_file() else sorted(target.rglob("*"))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() in _TEXT_SUFFIXES:
            files.append(candidate)
        else:
            skipped += 1
            logger.info("Skipping unsupported file %s", candidate)
    return files, skipped
