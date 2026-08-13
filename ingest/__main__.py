"""ingest/__main__.py — CLI for RAG document ingestion.

Usage::

    python -m ingest examples/rag/corpus
    python -m ingest path/to/file.md --chunk-size 1000 --chunk-overlap 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.config import get_settings
from core.observability import configure_logging
from ingest.pipeline import ingest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingest",
        description="Ingest .txt/.md files into the configured RAG vector store.",
    )
    parser.add_argument("path", help="File or directory to ingest.")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args(argv)

    configure_logging(level="INFO")
    target = Path(args.path)
    if not target.exists():
        print(f"error: path not found: {target}", file=sys.stderr)
        return 2

    settings = get_settings()
    report = ingest_path(
        target,
        settings,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(
        f"files={report.files_read} chunks={report.chunks_upserted} "
        f"skipped={report.skipped} backend={report.backend}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
