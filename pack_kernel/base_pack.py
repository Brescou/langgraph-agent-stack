"""
pack_kernel/base_pack.py — Abstract contract for all domain packs.

Every domain pack MUST inherit from BaseDomainPack and implement the three
abstract methods (run, arun, stream_events).  The class-level attributes
pack_id, name, and description are required metadata used by PackRegistry.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Canonical SSE event shape (wire format for all pack stream consumers)
# ---------------------------------------------------------------------------

PACK_STREAM_EVENT_TYPE_KEY = "type"


def _serialize_stream_field(value: Any) -> Any:
    """Make a stream event field JSON-safe for SSE ``data:`` lines."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def pack_stream_event(event_type: str, **fields: Any) -> dict[str, Any]:
    """Build a canonical pack stream event.

    All packs must emit events as ``{"type": "<kind>", ...payload}`` so legacy
    ``POST /run/stream`` and per-pack ``/run/stream`` routes share one schema.

    Args:
        event_type: Event kind (e.g. ``phase_started``, ``token``).
        **fields: Payload fields merged at the top level (not nested under ``data``).

    Returns:
        Canonical event dict with a ``type`` key.
    """
    return {
        PACK_STREAM_EVENT_TYPE_KEY: event_type,
        **{key: _serialize_stream_field(val) for key, val in fields.items()},
    }


def normalize_pack_stream_event(event: dict[str, Any]) -> dict[str, Any]:
    """Normalize a pack stream event to the canonical ``{type, ...}`` shape.

    Accepts legacy ``{"event": "...", "data": {...}}`` payloads for backward
    compatibility during migration. Idempotent when ``type`` is already present.

    Args:
        event: Raw event dict from a pack implementation.

    Returns:
        Canonical event dict.

    Raises:
        ValueError: When the dict has neither ``type`` nor ``event``.
    """
    if PACK_STREAM_EVENT_TYPE_KEY in event:
        return event
    legacy_kind = event.get("event")
    if not isinstance(legacy_kind, str) or not legacy_kind:
        raise ValueError(f"Stream event missing '{PACK_STREAM_EVENT_TYPE_KEY}' key: {event!r}")
    data = event.get("data", {})
    if not isinstance(data, dict):
        return pack_stream_event(legacy_kind, data=data)
    return pack_stream_event(legacy_kind, **data)


# ---------------------------------------------------------------------------
# Default schemas — used when a concrete pack does not declare its own
# ---------------------------------------------------------------------------


class _DefaultPackInput(BaseModel):
    query: str


class _DefaultPackOutput(BaseModel):
    result: dict[str, Any]


class BaseDomainPack(abc.ABC):
    """
    Abstract base class for all domain packs.

    A domain pack encapsulates a complete multi-agent pipeline as a deployable
    unit.  It owns its LangGraph graph, its agents, and its output schema.

    Class-level attributes (MUST be set on every concrete subclass):
        pack_id:     Unique stable identifier used by PackRegistry (e.g. "research_analysis").
        name:        Human-readable display name.
        description: One-sentence description of what this pack does.

    Constructor args (same contract as the former MultiAgentGraph):
        run_id:      Optional stable identifier for the pipeline run.
        llm:         Optional pre-built LangChain chat model (injected by API at startup).
        checkpointer: Optional pre-built LangGraph checkpointer.
        budget_usd:  Per-agent USD cost ceiling for LLM calls.
                     Each agent in the pipeline enforces this limit independently —
                     the total pipeline cost may reach ``budget_usd * number_of_agents``.
                     Set to ``None`` to disable budget enforcement (default).
    """

    pack_id: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str]
    version: ClassVar[str] = "1.0"
    input_schema: ClassVar[type[BaseModel]] = _DefaultPackInput
    output_schema: ClassVar[type[BaseModel]] = _DefaultPackOutput

    def __init__(
        self,
        run_id: str | None = None,
        llm: Any | None = None,
        checkpointer: Any | None = None,
        budget_usd: float | None = None,
    ) -> None:
        if budget_usd is not None and budget_usd < 0:
            raise ValueError(f"budget_usd must be non-negative, got {budget_usd!r}")
        self.run_id = run_id
        self._llm = llm
        self._checkpointer = checkpointer
        self._budget_usd = budget_usd

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def run(self, query: str) -> Any:
        """Execute the pack pipeline synchronously and return a structured result."""

    @abc.abstractmethod
    async def arun(self, query: str) -> Any:
        """Execute the pack pipeline asynchronously and return a structured result."""

    async def stream_events(self, query: str) -> AsyncIterator[dict[str, Any]]:
        """Stream pipeline execution events in the canonical SSE schema.

        Subclasses implement :meth:`_iter_stream_events` and should build
        events with :func:`pack_stream_event`. Legacy ``{event, data}`` dicts
        are normalized here before they reach API consumers.
        """
        async for raw in self._iter_stream_events(query):
            yield normalize_pack_stream_event(raw)

    @abc.abstractmethod
    def _iter_stream_events(self, query: str) -> AsyncIterator[dict[str, Any]]:
        """Yield raw pipeline events (canonical or legacy shape).

        Concrete packs implement this as an ``async def`` generator (with
        ``yield``). The return annotation is ``AsyncIterator`` because callers
        receive the async generator object directly (no ``await`` on the call).
        """

    # ------------------------------------------------------------------
    # Cost tracking
    # ------------------------------------------------------------------

    @property
    def cost_usd(self) -> float:
        """Total USD spent on LLM calls during this run.

        Subclasses should override this to aggregate costs across all
        internal agents. Returns 0.0 by default.
        """
        return 0.0

    # ------------------------------------------------------------------
    # Optional lifecycle hooks
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release any resources held by the pack (thread pools, connections)."""

    def __enter__(self) -> BaseDomainPack:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
