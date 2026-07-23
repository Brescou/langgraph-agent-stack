"""api/state.py — Global mutable state shared across the API layer.

Populated during lifespan startup (api/lifespan.py) and read by middleware,
dependencies, and endpoints.  All mutations go through lifespan; all reads
go through the accessor helpers at the bottom of this module.
"""

from __future__ import annotations

import importlib.metadata
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.language_models import BaseChatModel

from core.security import (
    InMemorySessionRegistry,
    InputValidator,
    SessionRegistryBackend,
    IdempotencyStore,
    InMemoryIdempotencyStore,
)

try:
    APP_VERSION = importlib.metadata.version("langgraph-agent-stack")
except importlib.metadata.PackageNotFoundError:
    APP_VERSION = "0.0.0"

# ---------------------------------------------------------------------------
# Shared resources — set during lifespan startup
# ---------------------------------------------------------------------------

start_time: float = 0.0
executor: ThreadPoolExecutor | None = None
shared_llm: BaseChatModel | None = None
shared_checkpointer: Any | None = None
shared_memory: Any | None = None
active_pack_cls: Any | None = None  # default pack resolved from PackRegistry
shared_connector: Any | None = None  # optional BaseConnector

# ---------------------------------------------------------------------------
# Security primitives
# ---------------------------------------------------------------------------

rate_limiter: Any | None = None
input_validator: InputValidator = InputValidator(max_length=2000)

# Human-review queue for regulated pack outputs (core/review_store.py).
# Set during lifespan startup; None means review tracking is unavailable.
review_store: Any | None = None

# ---------------------------------------------------------------------------
# Lifecycle flags
# ---------------------------------------------------------------------------

shutting_down: threading.Event = threading.Event()
_init_lock: threading.Lock = threading.Lock()

# ---------------------------------------------------------------------------
# In-flight session registry — dedupes concurrent runs per session_id
# ---------------------------------------------------------------------------

session_registry: SessionRegistryBackend | None = None
_session_registry_lock: threading.Lock = threading.Lock()

# ---------------------------------------------------------------------------
# Idempotency store
# ---------------------------------------------------------------------------

idempotency_store: IdempotencyStore | None = None
_idempotency_store_lock: threading.Lock = threading.Lock()


def _get_session_registry() -> SessionRegistryBackend:
    """Return the registry, lazily defaulting to in-memory.

    The lifespan wires the configured backend (memory or Redis); the lazy
    fallback keeps direct callers (unit tests) working without startup.
    """
    global session_registry
    if session_registry is None:
        with _session_registry_lock:
            if session_registry is None:
                session_registry = InMemorySessionRegistry()
    return session_registry


def try_acquire_session(session_id: str) -> bool:
    """Mark *session_id* as having a run in flight.

    Returns:
        True if the session was free and is now marked in flight;
        False if a run is already in progress for this session.
    """
    return _get_session_registry().try_acquire(session_id)


def release_session(session_id: str) -> None:
    """Release the in-flight marker for *session_id* (idempotent)."""
    _get_session_registry().release(session_id)


def _get_idempotency_store() -> IdempotencyStore:
    """Return the store, lazily defaulting to in-memory.

    The lifespan wires the configured backend (memory or Redis); the lazy
    fallback keeps direct callers (unit tests) working without startup.
    """
    global idempotency_store

    if idempotency_store is None:
        with _idempotency_store_lock:
            if idempotency_store is None:
                idempotency_store = InMemoryIdempotencyStore()

    return idempotency_store


def get_shared_idempotency_store() -> IdempotencyStore:
    """Return the shared idempotency store."""
    return _get_idempotency_store()

# ---------------------------------------------------------------------------
# Accessor helpers (lazy-retry on None)
# ---------------------------------------------------------------------------


def get_shared_llm() -> BaseChatModel | None:
    """Return the shared LLM, retrying init if the first attempt failed."""
    global shared_llm
    if shared_llm is None:
        with _init_lock:
            if shared_llm is None:
                from core.config import get_settings
                from core.llm import get_llm

                try:
                    shared_llm = get_llm(get_settings().llm_config)
                except (ImportError, ValueError):
                    pass
    return shared_llm


def get_shared_checkpointer() -> Any | None:
    """Return the shared checkpointer initialized during lifespan startup."""
    return shared_checkpointer
