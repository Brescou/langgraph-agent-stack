"""Operator tool: ingest local .txt/.md files into the RAG vector store."""

from ingest.pipeline import IngestReport, ingest_path

__all__ = ["IngestReport", "ingest_path"]
