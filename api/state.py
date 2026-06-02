"""api/state.py — Global mutable state shared across the API layer.

Populated during lifespan startup (api/lifespan.py) and read by middleware,
dependencies, and endpoints.  All mutations go through lifespan; all reads
go through the accessor helpers at the bottom of this module.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.language_models import BaseChatModel

from core.security import InputValidator

APP_VERSION = "0.5.0"

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

# ---------------------------------------------------------------------------
# Lifecycle flags
# ---------------------------------------------------------------------------

shutting_down: threading.Event = threading.Event()
_init_lock: threading.Lock = threading.Lock()

# ---------------------------------------------------------------------------
# Accessor helpers (lazy-retry on None)
# ---------------------------------------------------------------------------


def get_shared_llm() -> BaseChatModel | None:
    """Return the shared LLM, retrying init if the first attempt failed."""
    global shared_llm
    if shared_llm is None:
        with _init_lock:
            if shared_llm is None:
                from api.lifespan import _init_llm_and_checkpointer
                from core.config import get_settings

                _init_llm_and_checkpointer(get_settings())
    return shared_llm


def get_shared_checkpointer() -> Any | None:
    """Return the shared checkpointer, retrying init if the first attempt failed."""
    global shared_checkpointer
    if shared_checkpointer is None:
        with _init_lock:
            if shared_checkpointer is None:
                from api.lifespan import _init_llm_and_checkpointer
                from core.config import get_settings

                _init_llm_and_checkpointer(get_settings())
    return shared_checkpointer
