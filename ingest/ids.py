"""Stable chunk identifiers for RAG ingest upserts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def chunk_id(path: Path, chunk_index: int) -> str:
    """Return `{sha256(resolved_absolute_posix)[:16]}:{chunk_index}`."""
    resolved = path.resolve().as_posix()
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return f"{digest}:{chunk_index}"
