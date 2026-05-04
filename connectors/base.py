"""
connectors/base.py — Minimal contract for external data / tool adapters.

Connectors are **not** LangGraph nodes: they are optional building blocks that
a domain pack (or an agent) can call to pull structured snippets from outside
the LLM (SQL, HTTP APIs, file stores, search indices, etc.).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass(frozen=True, slots=True)
class ConnectorRequest:
    """One invocation: a query string plus optional caps and opaque filters.

    * ``query`` — Free text or a pseudo-query; SQL/API connectors interpret it.
    * ``limit`` — Upper bound on rows/snippets (best-effort).
    * ``filters`` — Backend-specific hints (e.g. column filters, collection id).
    * ``context`` — Call-site metadata (e.g. ``session_id``, ``tenant``).
    """

    query: str
    limit: int = 10
    filters: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConnectorResult:
    """Normalized outcome: tabular-ish rows for downstream prompt formatting.

    Each record is a flat ``dict`` (e.g. ``{"title": "...", "snippet": "..."}``).
    Semantics are connector-specific; packs remain responsible for validation.
    """

    records: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseConnector(abc.ABC):
    """Abstract adapter: one async entry point, optional lifecycle hook.

    Subclasses declare stable ``connector_id`` / ``name`` / ``description``
    for discovery and logging. They implement :meth:`fetch` only; connection
    setup belongs in ``__init__`` or lazy properties — no global registry here.
    """

    connector_id: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str] = ""

    @abc.abstractmethod
    async def fetch(self, request: ConnectorRequest) -> ConnectorResult:
        """Return zero or more records for the given request."""

    async def close(self) -> None:
        """Override to close pools, HTTP clients, or file handles."""
