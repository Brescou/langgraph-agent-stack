"""
core/memory.py — Memory and checkpointing layer for the LangGraph agent stack.

This module centralises all persistence concerns:

* ``create_checkpointer`` — factory that returns the appropriate LangGraph
  ``BaseCheckpointSaver`` based on ``Settings.memory_backend``.
* ``ConversationMemory`` — run-level history stored in a SQLite ``runs``
  table (default / fallback backend).
* ``RedisRunHistory`` — run-level history stored in Redis hashes + sorted
  sets.  Used when ``MEMORY_BACKEND=redis`` and the ``redis`` package is
  installed.
* ``PostgresRunHistory`` — run-level history stored in a ``run_history``
  PostgreSQL table.  Used when ``MEMORY_BACKEND=postgres`` and ``psycopg``
  is installed.
* ``create_run_history`` — factory that returns the appropriate run history
  store based on ``Settings.memory_backend``.

Backend matrix
--------------
+-------------------+-------------------------------+---------------------------+
| ``memory_backend``| Checkpointer                  | Run history store         |
+===================+===============================+===========================+
| ``sqlite``        | ``AsyncSqliteSaver``          | ``ConversationMemory``    |
+-------------------+-------------------------------+---------------------------+
| ``redis``         | ``AsyncRedisSaver``           | ``RedisRunHistory``       |
+-------------------+-------------------------------+---------------------------+
| ``postgres``      | ``AsyncPostgresSaver``        | ``PostgresRunHistory``    |
+-------------------+-------------------------------+---------------------------+
| fallback / error  | ``MemorySaver``               | ``ConversationMemory``    |
+-------------------+-------------------------------+---------------------------+

Usage example::

    from core.memory import create_checkpointer, create_run_history
    from core.config import get_settings

    settings = get_settings()
    checkpointer = create_checkpointer(settings)
    history = create_run_history(settings)

    history.save_run(run_id, query, result, metadata)
    run = history.get_run(run_id)
    recent = history.list_runs(limit=5)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, cast, runtime_checkable
from urllib.parse import urlparse

from langgraph.checkpoint.memory import MemorySaver

from core.config import MemoryBackend, Settings

logger = logging.getLogger(__name__)


def _redact_url(url: str) -> str:
    """Return ``scheme://host:port/...`` with credentials stripped for logging.

    Robust to URLs without credentials and to malformed inputs: when no
    hostname can be parsed the constant ``"[REDACTED]"`` is returned so a
    secret can never leak into log sinks.

    Args:
        url: Connection URL/DSN, possibly containing ``user:password@``.

    Returns:
        The URL without userinfo, query, or fragment — or ``"[REDACTED]"``.
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            return "[REDACTED]"
        netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
        return f"{parsed.scheme}://{netloc}{parsed.path}"
    except ValueError:
        return "[REDACTED]"


def _fallback_or_raise(message: str) -> MemorySaver:
    """Return ``MemorySaver`` in development; raise ``RuntimeError`` in production."""
    from core.config import get_settings

    if get_settings().environment != "development":
        raise RuntimeError(message)
    logger.warning(message + " — falling back to MemorySaver (dev only).")
    return MemorySaver()


# ---------------------------------------------------------------------------
# Checkpointer factory
# ---------------------------------------------------------------------------


async def init_checkpointer(settings: Settings) -> Any:
    """Initialize an async LangGraph checkpointer at application startup.

    Called from the FastAPI lifespan so async context managers (``aiosqlite``,
    ``AsyncRedisSaver``, ``AsyncPostgresSaver``) are entered once and reused
    for ``stream_events()`` / ``arun()`` on the main event loop.  Sync
    ``run()`` calls execute in ``ThreadPoolExecutor`` workers and remain
    compatible with the async savers.

    Args:
        settings: Application settings.

    Returns:
        A configured async ``BaseCheckpointSaver`` instance.
    """
    backend = settings.memory_backend

    if backend == MemoryBackend.SQLITE:
        return await _init_sqlite_checkpointer(settings.sqlite_path)

    if backend == MemoryBackend.REDIS:
        return await _init_redis_checkpointer(settings.redis_url)

    if backend == MemoryBackend.POSTGRES:
        return await _init_postgres_checkpointer(settings.postgres_url)

    logger.warning(
        "Unknown memory_backend %r — falling back to MemorySaver.",
        backend,
    )
    return _fallback_or_raise(
        f"Unknown memory_backend {backend!r} — cannot create checkpointer."
    )


def create_checkpointer(settings: Settings) -> Any:
    """
    Construct and return a LangGraph checkpoint saver based on ``settings``.

    The function attempts to import the backend-specific saver package.  If
    the package is not installed, or if the backend is unrecognised, it falls
    back to the in-process ``MemorySaver`` and logs a warning so the issue is
    observable without crashing the application.

    Args:
        settings: The application ``Settings`` instance.  Read fields:
            ``memory_backend``, ``sqlite_path``, ``redis_url``, ``postgres_url``.

    Returns:
        A configured LangGraph ``BaseCheckpointSaver`` instance.  The exact
        concrete type depends on the resolved backend:

        * ``AsyncSqliteSaver``     — when ``memory_backend == MemoryBackend.SQLITE``
          and ``langgraph-checkpoint-sqlite`` is installed.
        * ``AsyncRedisSaver``      — when ``memory_backend == MemoryBackend.REDIS``
          and ``langgraph-checkpoint-redis`` is installed.
        * ``AsyncPostgresSaver``   — when ``memory_backend == MemoryBackend.POSTGRES``
          and ``langgraph-checkpoint-postgres`` is installed.
        * ``MemorySaver``          — fallback for all other cases.
    """
    backend = settings.memory_backend

    if backend == MemoryBackend.SQLITE:
        return _create_sqlite_checkpointer(settings.sqlite_path)

    if backend == MemoryBackend.REDIS:
        return _create_redis_checkpointer(settings.redis_url)

    if backend == MemoryBackend.POSTGRES:
        return _create_postgres_checkpointer(settings.postgres_url)

    logger.warning(
        "Unknown memory_backend %r — falling back to MemorySaver.",
        backend,
    )
    return _fallback_or_raise(
        f"Unknown memory_backend {backend!r} — cannot create checkpointer."
    )


async def _enter_async_checkpointer_cm(cm: Any) -> Any:
    """Enter an async checkpointer context manager and run ``setup()`` when present."""
    checkpointer = await cm.__aenter__()
    setup_result = checkpointer.setup()
    if asyncio.iscoroutine(setup_result):
        await setup_result
    return checkpointer


def _sync_enter_async_checkpointer_cm(cm: Any) -> Any:
    """Enter an async checkpointer CM from a synchronous caller (no running loop)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_enter_async_checkpointer_cm(cm))
    raise RuntimeError(
        "create_checkpointer() cannot be called from a running event loop; "
        "use init_checkpointer() during lifespan startup instead."
    )


async def _init_sqlite_checkpointer(sqlite_path: str) -> Any:
    """Build an ``AsyncSqliteSaver`` checkpointer (lifespan / async path)."""
    db_path = Path(sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from langgraph.checkpoint.sqlite.aio import (  # type: ignore[import]
            AsyncSqliteSaver,
        )

        cm = AsyncSqliteSaver.from_conn_string(str(db_path))
        checkpointer = await _enter_async_checkpointer_cm(cm)
        _set_async_checkpointer_cm(cm)
        logger.info(
            "Checkpointer: AsyncSqliteSaver initialised",
            extra={"path": str(db_path)},
        )
        return checkpointer

    except ImportError:
        logger.warning(
            "langgraph-checkpoint-sqlite not installed — falling back to "
            "MemorySaver.  Install with: pip install langgraph-checkpoint-sqlite",
            extra={"sqlite_path": sqlite_path},
        )
        return _fallback_or_raise("langgraph-checkpoint-sqlite not installed.")


def _create_sqlite_checkpointer(sqlite_path: str) -> Any:
    """Build a ``SqliteSaver`` checkpointer for sync ``run()`` callers (tests)."""
    db_path = Path(sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import]

        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA cache_size=1000")
        checkpointer = SqliteSaver(conn)
        logger.info(
            "Checkpointer: SqliteSaver initialised",
            extra={"path": str(db_path)},
        )
        return checkpointer

    except ImportError:
        logger.warning(
            "langgraph-checkpoint-sqlite not installed — falling back to "
            "MemorySaver.  Install with: pip install langgraph-checkpoint-sqlite",
            extra={"sqlite_path": sqlite_path},
        )
        return _fallback_or_raise("langgraph-checkpoint-sqlite not installed.")


# Module-level references for cleanup
_active_checkpointer_cm: Any = None
_active_async_checkpointer_cm: Any = None
_checkpointer_lock: threading.Lock = threading.Lock()


def _set_checkpointer_cm(cm: Any) -> None:
    """Thread-safe store of a sync context manager for cleanup at shutdown."""
    global _active_checkpointer_cm
    with _checkpointer_lock:
        _active_checkpointer_cm = cm


def _set_async_checkpointer_cm(cm: Any) -> None:
    """Thread-safe store of an async context manager for cleanup at shutdown."""
    global _active_async_checkpointer_cm
    with _checkpointer_lock:
        _active_async_checkpointer_cm = cm


def cleanup_checkpointer() -> None:
    """Exit any sync checkpointer context manager. Call at shutdown."""
    global _active_checkpointer_cm
    with _checkpointer_lock:
        if _active_checkpointer_cm is not None:
            try:
                _active_checkpointer_cm.__exit__(None, None, None)
                logger.debug("Sync checkpointer context manager exited cleanly.")
            except Exception as exc:
                logger.warning("Checkpointer cleanup failed", extra={"error": str(exc)})
            finally:
                _active_checkpointer_cm = None


async def cleanup_checkpointer_async() -> None:
    """Exit async and sync checkpointer context managers. Call at lifespan shutdown."""
    global _active_async_checkpointer_cm
    with _checkpointer_lock:
        cm = _active_async_checkpointer_cm
        _active_async_checkpointer_cm = None
    if cm is not None:
        try:
            await cm.__aexit__(None, None, None)
            logger.debug("Async checkpointer context manager exited cleanly.")
        except Exception as exc:
            logger.warning(
                "Async checkpointer cleanup failed", extra={"error": str(exc)}
            )
    cleanup_checkpointer()


async def _init_redis_checkpointer(redis_url: str) -> Any:
    """Build an ``AsyncRedisSaver`` checkpointer (lifespan / async path)."""
    try:
        from langgraph.checkpoint.redis.aio import (  # type: ignore[import]
            AsyncRedisSaver,
        )

        cm = AsyncRedisSaver.from_conn_string(redis_url)
        checkpointer = await _enter_async_checkpointer_cm(cm)
        _set_async_checkpointer_cm(cm)
        logger.info(
            "Checkpointer: AsyncRedisSaver initialised",
            extra={"url": _redact_url(redis_url)},
        )
        return checkpointer

    except ImportError:
        logger.warning(
            "langgraph-checkpoint-redis not installed — falling back to "
            "MemorySaver.  Install with: pip install langgraph-checkpoint-redis",
            extra={"redis_url": _redact_url(redis_url)},
        )
        return _fallback_or_raise("langgraph-checkpoint-redis not installed.")


def _create_redis_checkpointer(redis_url: str) -> Any:
    """Build a ``RedisSaver`` checkpointer for sync ``run()`` callers (tests)."""
    try:
        from langgraph.checkpoint.redis import RedisSaver  # type: ignore[import]

        conn = RedisSaver.from_conn_string(redis_url)
        checkpointer = conn.__enter__()
        _set_checkpointer_cm(conn)
        logger.info(
            "Checkpointer: RedisSaver initialised",
            extra={"url": _redact_url(redis_url)},
        )
        return checkpointer

    except ImportError:
        logger.warning(
            "langgraph-checkpoint-redis not installed — falling back to "
            "MemorySaver.  Install with: pip install langgraph-checkpoint-redis",
            extra={"redis_url": _redact_url(redis_url)},
        )
        return _fallback_or_raise("langgraph-checkpoint-redis not installed.")


async def _init_postgres_checkpointer(postgres_url: str | None) -> Any:
    """Build an ``AsyncPostgresSaver`` checkpointer (lifespan / async path)."""
    if not postgres_url:
        logger.warning(
            "MEMORY_BACKEND=postgres but POSTGRES_URL is not set — "
            "falling back to MemorySaver."
        )
        return _fallback_or_raise(
            "MEMORY_BACKEND=postgres but POSTGRES_URL is not set."
        )

    try:
        from langgraph.checkpoint.postgres.aio import (  # type: ignore[import]
            AsyncPostgresSaver,
        )

        cm = AsyncPostgresSaver.from_conn_string(postgres_url)
        checkpointer = await _enter_async_checkpointer_cm(cm)
        _set_async_checkpointer_cm(cm)
        logger.info(
            "Checkpointer: AsyncPostgresSaver initialised (tables created)",
            extra={"url": _redact_url(postgres_url)},
        )
        return checkpointer

    except ImportError:
        logger.warning(
            "langgraph-checkpoint-postgres not installed — falling back to "
            "MemorySaver.  Install with: uv sync --extra postgres",
            extra={"postgres_url": _redact_url(postgres_url)},
        )
        return _fallback_or_raise("langgraph-checkpoint-postgres not installed.")


def _create_postgres_checkpointer(postgres_url: str | None) -> Any:
    """Build a ``PostgresSaver`` checkpointer for sync ``run()`` callers (tests)."""
    if not postgres_url:
        logger.warning(
            "MEMORY_BACKEND=postgres but POSTGRES_URL is not set — "
            "falling back to MemorySaver."
        )
        return _fallback_or_raise(
            "MEMORY_BACKEND=postgres but POSTGRES_URL is not set."
        )

    try:
        from langgraph.checkpoint.postgres import (
            PostgresSaver,  # type: ignore[import]
        )

        conn = PostgresSaver.from_conn_string(postgres_url)
        checkpointer = conn.__enter__()
        _set_checkpointer_cm(conn)
        checkpointer.setup()
        logger.info(
            "Checkpointer: PostgresSaver initialised (tables created)",
            extra={"url": _redact_url(postgres_url)},
        )
        return checkpointer

    except ImportError:
        logger.warning(
            "langgraph-checkpoint-postgres not installed — falling back to "
            "MemorySaver.  Install with: uv sync --extra postgres",
            extra={"postgres_url": _redact_url(postgres_url)},
        )
        return _fallback_or_raise("langgraph-checkpoint-postgres not installed.")


# ---------------------------------------------------------------------------
# RunHistoryStore protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RunHistoryStore(Protocol):
    """Protocol for run history storage backends."""

    def save_run(
        self,
        run_id: str,
        query: str,
        result: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def get_run(self, run_id: str) -> dict[str, Any] | None: ...

    def list_runs(self, limit: int = 10) -> list[dict[str, Any]]: ...

    def list_runs_by_session(
        self, session_id: str, limit: int = 50
    ) -> list[dict[str, Any]]: ...

    def get_pack_version_for_session(
        self, session_id: str, pack_id: str
    ) -> str | None: ...

    def health_check(self) -> tuple[str, str]: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# ConversationMemory — SQLite run-history persistence
# ---------------------------------------------------------------------------

_DDL_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    NOT NULL UNIQUE,
    query         TEXT    NOT NULL,
    result_json   TEXT    NOT NULL DEFAULT '{}',
    metadata_json TEXT    NOT NULL DEFAULT '{}',
    created_at    TEXT    NOT NULL
);
"""

_IDX_RUNS_CREATED = """
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs (created_at DESC);
"""


class ConversationMemory:
    """
    Persistent run-history store backed by SQLite.

    Each call to ``save_run`` writes a single row to the ``runs`` table.
    ``get_run`` and ``list_runs`` provide read access to the stored history.

    The class can be used as a context manager to ensure the database
    connection is closed after a block of work::

        with ConversationMemory("./data/memory.db") as mem:
            mem.save_run(run_id, query, result, metadata)

    It can also be used standalone, with ``close()`` called explicitly when
    the memory object is no longer needed.

    When ``backend`` is ``"redis"`` or ``"postgres"``, the SQLite store is
    still used for run history (lightweight audit trail) while the main
    checkpointer is handled by :func:`create_checkpointer`.  An informational
    log explains persistence options so operators know the audit DB differs
    from the checkpoint store.

    Args:
        db_path: Filesystem path to the SQLite database.  The parent
            directory is created automatically.
        backend: The configured memory backend name.  Only used for
            logging when the run-history store diverges from the checkpoint
            backend.
        redis_url: Optional Redis URL (logged for diagnostics).
        postgres_url: Optional Postgres DSN (logged for diagnostics).

    Attributes:
        db_path: Resolved absolute path to the database file.

    Raises:
        sqlite3.Error: If the database cannot be opened or the schema
            cannot be applied.
    """

    def __init__(
        self,
        db_path: str,
        backend: str = "sqlite",
        redis_url: str | None = None,
        postgres_url: str | None = None,
    ) -> None:
        if backend not in ("sqlite",):
            logger.info(
                "ConversationMemory run-history uses a local SQLite file "
                "('%s') while the checkpoint backend is '%s'.  To persist "
                "run history across pod restarts, ensure the SQLite path is "
                "backed by a PersistentVolume.  Alternatively, set "
                "sqlite_path=':memory:' if run history can be ephemeral.",
                db_path,
                backend,
                extra={"backend": backend, "db_path": db_path},
            )

        if db_path == ":memory:":
            self.db_path: Path = Path(db_path)
        else:
            self.db_path = Path(db_path).resolve()
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit for PRAGMAs
            timeout=30,
        )
        self._conn.row_factory = sqlite3.Row
        # Enable WAL mode for concurrent read/write access
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=1000")
        # Switch to DEFERRED transactions after PRAGMAs
        self._conn.isolation_level = "DEFERRED"
        self._lock = threading.Lock()
        self._apply_schema()

        logger.debug(
            "ConversationMemory initialised",
            extra={"db_path": str(self.db_path), "backend": backend},
        )

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> ConversationMemory:
        """Return self so the instance can be used as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the database connection on exit, regardless of exceptions."""
        self.close()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def _apply_schema(self) -> None:
        """
        Create the ``runs`` table and index if they do not already exist.

        This is idempotent: safe to call on every startup against a
        database that already contains data.
        """
        with self._transaction():
            self._conn.execute(_DDL_RUNS)
            self._conn.execute(_IDX_RUNS_CREATED)

    @contextmanager
    def _transaction(self) -> Generator[None, None, None]:
        """
        Yield a transactional context.

        Commits on clean exit; rolls back on any exception so partial
        writes never reach the database.

        Yields:
            Nothing — the caller operates directly on ``self._conn``.

        Raises:
            sqlite3.Error: Re-raised after rollback on failure.
        """
        with self._lock:
            try:
                yield
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_run(
        self,
        run_id: str,
        query: str,
        result: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Persist the outcome of a single agent run.

        If a row with the same ``run_id`` already exists it is replaced,
        allowing callers to update a run record after a retry without
        duplicating history.

        Args:
            run_id: Unique identifier for the agent run (UUID string).
            query: The original user query or task description.
            result: Arbitrary result payload — will be JSON-serialised.
                Should contain at minimum ``{"summary": "..."}`` or a
                serialised ``ResearchResult`` / ``AnalysisReport`` dict.
            metadata: Optional key-value metadata (agent name, thread_id,
                confidence score, elapsed seconds, etc.).  Defaults to an
                empty dict when omitted.

        Raises:
            ValueError: If ``run_id`` or ``query`` is empty.
            sqlite3.Error: On database write failure.
        """
        if not run_id or not run_id.strip():
            raise ValueError("save_run: run_id must not be empty.")
        if not query or not query.strip():
            raise ValueError("save_run: query must not be empty.")

        result_json = json.dumps(result, ensure_ascii=False, default=str)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False, default=str)
        created_at = datetime.now(UTC).isoformat()

        with self._transaction():
            self._conn.execute(
                """
                INSERT INTO runs (run_id, query, result_json, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    query         = excluded.query,
                    result_json   = excluded.result_json,
                    metadata_json = excluded.metadata_json,
                    created_at    = excluded.created_at
                """,
                (run_id, query.strip(), result_json, metadata_json, created_at),
            )

        logger.debug(
            "ConversationMemory.save_run",
            extra={"run_id": run_id, "query_preview": query[:80]},
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """
        Retrieve a single run record by its ``run_id``.

        Args:
            run_id: The unique run identifier to look up.

        Returns:
            A dict with keys ``id``, ``run_id``, ``query``, ``result``,
            ``metadata``, and ``created_at`` — or ``None`` when no record
            with that ``run_id`` exists.

        Raises:
            sqlite3.Error: On database read failure.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT id, run_id, query, result_json, metadata_json, created_at "
                "FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def list_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Return the most recent run records, newest first.

        Args:
            limit: Maximum number of records to return.  Must be a
                positive integer.  Defaults to 10.

        Returns:
            A list of run dicts (same shape as ``get_run``), ordered by
            ``created_at`` descending.  An empty list is returned when no
            runs have been saved yet.

        Raises:
            ValueError: If ``limit`` is less than 1.
            sqlite3.Error: On database read failure.
        """
        if limit < 1:
            raise ValueError(f"list_runs: limit must be >= 1, got {limit}.")

        with self._lock:
            rows = self._conn.execute(
                "SELECT id, run_id, query, result_json, metadata_json, created_at "
                "FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def list_runs_by_session(
        self, session_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """
        Return run records whose metadata contains ``session_id``, newest first.

        Filters in SQL using SQLite's ``json_extract`` so only matching rows
        are loaded — avoids fetching all rows and filtering in Python.

        Args:
            session_id: The session identifier to filter by.
            limit: Maximum number of records to return.  Must be >= 1.

        Returns:
            A list of run dicts ordered by ``created_at`` descending.

        Raises:
            ValueError: If ``session_id`` is empty or ``limit`` < 1.
            sqlite3.Error: On database read failure.
        """
        if not session_id or not session_id.strip():
            raise ValueError("list_runs_by_session: session_id must not be empty.")
        if limit < 1:
            raise ValueError(f"list_runs_by_session: limit must be >= 1, got {limit}.")

        with self._lock:
            rows = self._conn.execute(
                "SELECT id, run_id, query, result_json, metadata_json, created_at "
                "FROM runs "
                "WHERE json_extract(metadata_json, '$.session_id') = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def get_pack_version_for_session(self, session_id: str, pack_id: str) -> str | None:
        """Return the pack_version last used for this session+pack, or None if no history.

        Args:
            session_id: The session identifier to look up.
            pack_id: The pack identifier to filter by.

        Returns:
            The pack_version string from the most recent matching run, or ``None``
            when no run exists for this session+pack combination.
        """
        with self._lock:
            row = self._conn.execute(
                """
                SELECT metadata_json FROM runs
                WHERE json_extract(metadata_json, '$.session_id') = ?
                  AND json_extract(metadata_json, '$.pack_id') = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (session_id, pack_id),
            ).fetchone()
        if row is None:
            return None
        try:
            meta = json.loads(row[0])
            return meta.get("pack_version")
        except (json.JSONDecodeError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> tuple[str, str]:
        """Verify SQLite connectivity with a simple query.

        Returns:
            Tuple of ``(status, detail)`` where status is ``"ok"`` or
            ``"degraded"``.
        """
        try:
            with self._lock:
                self._conn.execute("SELECT 1").fetchone()
            return ("ok", str(self.db_path))
        except Exception as exc:
            return ("degraded", f"SQLite unreachable: {exc}")

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Close the underlying SQLite connection.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        try:
            self._conn.close()
            logger.debug(
                "ConversationMemory: connection closed",
                extra={"db_path": str(self.db_path)},
            )
        except Exception as exc:
            logger.warning(
                "ConversationMemory.close() failed",
                extra={"db_path": str(self.db_path), "error": str(exc)},
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """
        Convert a ``sqlite3.Row`` from the ``runs`` table into a plain dict.

        JSON columns (``result_json``, ``metadata_json``) are decoded;
        decode errors fall back to an empty dict so callers always receive
        a consistently-typed structure.

        Args:
            row: A row fetched from the ``runs`` table.

        Returns:
            Dict with keys: ``id``, ``run_id``, ``query``, ``result``,
            ``metadata``, ``created_at``.
        """
        try:
            result: dict[str, Any] = json.loads(row["result_json"])
        except (json.JSONDecodeError, TypeError):
            result = {}

        try:
            metadata: dict[str, Any] = json.loads(row["metadata_json"])
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        return {
            "id": row["id"],
            "run_id": row["run_id"],
            "query": row["query"],
            "result": result,
            "metadata": metadata,
            "created_at": row["created_at"],
        }


# ---------------------------------------------------------------------------
# RedisRunHistory — Redis-backed run-history persistence
# ---------------------------------------------------------------------------


class RedisRunHistory:
    """Run history store backed by Redis.

    Uses a Redis hash per run (``run:{run_id}``) and a sorted set
    (``runs:timeline``) for chronological ordering.  Session filtering
    uses a per-session sorted set (``runs:session:{session_id}``).
    """

    _KEY_PREFIX = "runhistory"

    def __init__(self, redis_url: str) -> None:
        try:
            import redis as redis_lib
        except ImportError:
            raise ImportError(
                "redis package is required for RedisRunHistory. "
                "Install with: uv sync --extra redis"
            )
        self._redis = redis_lib.Redis.from_url(redis_url, decode_responses=True)
        self._prefix = self._KEY_PREFIX
        logger.info(
            "RedisRunHistory initialised",
            extra={"url": _redact_url(redis_url)},
        )

    def _run_key(self, run_id: str) -> str:
        return f"{self._prefix}:run:{run_id}"

    def _timeline_key(self) -> str:
        return f"{self._prefix}:timeline"

    def _session_key(self, session_id: str) -> str:
        return f"{self._prefix}:session:{session_id}"

    def _session_pack_key(self, session_id: str, pack_id: str) -> str:
        return f"{self._prefix}:session:{session_id}:pack:{pack_id}"

    def save_run(
        self,
        run_id: str,
        query: str,
        result: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not run_id or not run_id.strip():
            raise ValueError("save_run: run_id must not be empty.")
        if not query or not query.strip():
            raise ValueError("save_run: query must not be empty.")

        created_at = datetime.now(UTC).isoformat()
        meta = metadata or {}
        data = {
            "run_id": run_id,
            "query": query.strip(),
            "result_json": json.dumps(result, ensure_ascii=False, default=str),
            "metadata_json": json.dumps(meta, ensure_ascii=False, default=str),
            "created_at": created_at,
        }
        pipe = self._redis.pipeline()
        pipe.hset(self._run_key(run_id), mapping=data)
        score = datetime.fromisoformat(created_at).timestamp()
        pipe.zadd(self._timeline_key(), {run_id: score})
        session_id = meta.get("session_id")
        pack_id = meta.get("pack_id")
        if session_id:
            pipe.zadd(self._session_key(session_id), {run_id: score})
        if session_id and pack_id:
            pipe.zadd(self._session_pack_key(session_id, str(pack_id)), {run_id: score})
        pipe.execute()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        data = cast(
            dict[str, str],
            self._redis.hgetall(self._run_key(run_id)),
        )
        if not data:
            return None
        return self._decode(data)

    def list_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError(f"list_runs: limit must be >= 1, got {limit}.")
        run_ids = cast(
            list[str],
            self._redis.zrevrange(self._timeline_key(), 0, limit - 1),
        )
        return [r for rid in run_ids if (r := self.get_run(rid)) is not None]

    def list_runs_by_session(
        self, session_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        if not session_id or not session_id.strip():
            raise ValueError("list_runs_by_session: session_id must not be empty.")
        if limit < 1:
            raise ValueError(f"list_runs_by_session: limit must be >= 1, got {limit}.")
        run_ids = cast(
            list[str],
            self._redis.zrevrange(self._session_key(session_id), 0, limit - 1),
        )
        return [r for rid in run_ids if (r := self.get_run(rid)) is not None]

    def get_pack_version_for_session(self, session_id: str, pack_id: str) -> str | None:
        """Return pack_version from the latest run for this session and pack_id."""
        if not session_id or not session_id.strip() or not pack_id:
            return None
        run_ids = cast(
            list[str],
            self._redis.zrevrange(self._session_pack_key(session_id, pack_id), 0, 0),
        )
        if not run_ids:
            for run in self.list_runs_by_session(session_id, limit=50):
                meta = run.get("metadata") or {}
                if meta.get("pack_id") == pack_id:
                    return meta.get("pack_version")
            return None
        run = self.get_run(run_ids[0])
        if run is None:
            return None
        meta = run.get("metadata") or {}
        version = meta.get("pack_version")
        return str(version) if version is not None else None

    def health_check(self) -> tuple[str, str]:
        try:
            self._redis.ping()
            return ("ok", "redis reachable")
        except Exception as exc:
            return ("degraded", f"redis unreachable: {exc}")

    def close(self) -> None:
        try:
            self._redis.close()
        except Exception as exc:
            logger.debug("Redis close failed (ignored): %s", exc)

    @staticmethod
    def _decode(data: dict[str, str]) -> dict[str, Any]:
        try:
            result = json.loads(data.get("result_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            result = {}
        try:
            metadata = json.loads(data.get("metadata_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        return {
            "id": data.get("run_id", ""),
            "run_id": data.get("run_id", ""),
            "query": data.get("query", ""),
            "result": result,
            "metadata": metadata,
            "created_at": data.get("created_at", ""),
        }


# ---------------------------------------------------------------------------
# PostgresRunHistory — PostgreSQL-backed run-history persistence
# ---------------------------------------------------------------------------


class PostgresRunHistory:
    """Run history store backed by PostgreSQL.

    Creates a ``run_history`` table (separate from LangGraph checkpoint
    tables) and uses standard SQL for all queries.
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS run_history (
        id            SERIAL PRIMARY KEY,
        run_id        TEXT NOT NULL UNIQUE,
        query         TEXT NOT NULL,
        result_json   TEXT NOT NULL DEFAULT '{}',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at    TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_run_history_created
        ON run_history (created_at DESC);
    """

    def __init__(self, postgres_url: str) -> None:
        try:
            import psycopg
        except ImportError:
            raise ImportError(
                "psycopg is required for PostgresRunHistory. "
                "Install with: uv sync --extra postgres"
            )
        self._conn = psycopg.connect(postgres_url, autocommit=True)
        self._lock = threading.Lock()
        self._conn.execute(self._DDL)
        logger.info(
            "PostgresRunHistory initialised",
            extra={"url": _redact_url(postgres_url)},
        )

    def save_run(
        self,
        run_id: str,
        query: str,
        result: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not run_id or not run_id.strip():
            raise ValueError("save_run: run_id must not be empty.")
        if not query or not query.strip():
            raise ValueError("save_run: query must not be empty.")

        created_at = datetime.now(UTC).isoformat()
        result_json = json.dumps(result, ensure_ascii=False, default=str)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False, default=str)

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO run_history (run_id, query, result_json, metadata_json, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    query         = EXCLUDED.query,
                    result_json   = EXCLUDED.result_json,
                    metadata_json = EXCLUDED.metadata_json,
                    created_at    = EXCLUDED.created_at
                """,
                (run_id, query.strip(), result_json, metadata_json, created_at),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, run_id, query, result_json, metadata_json, created_at "
                "FROM run_history WHERE run_id = %s",
                (run_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError(f"list_runs: limit must be >= 1, got {limit}.")
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, run_id, query, result_json, metadata_json, created_at "
                "FROM run_history ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def list_runs_by_session(
        self, session_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        if not session_id or not session_id.strip():
            raise ValueError("list_runs_by_session: session_id must not be empty.")
        if limit < 1:
            raise ValueError(f"list_runs_by_session: limit must be >= 1, got {limit}.")
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, run_id, query, result_json, metadata_json, created_at "
                "FROM run_history "
                "WHERE metadata_json::jsonb->>'session_id' = %s "
                "ORDER BY created_at DESC LIMIT %s",
                (session_id, limit),
            )
            rows = cur.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_pack_version_for_session(self, session_id: str, pack_id: str) -> str | None:
        """Return pack_version from the latest run for this session and pack_id."""
        if not session_id or not session_id.strip() or not pack_id:
            return None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT metadata_json FROM run_history
                WHERE metadata_json::jsonb->>'session_id' = %s
                  AND metadata_json::jsonb->>'pack_id' = %s
                ORDER BY created_at DESC LIMIT 1
                """,
                (session_id, pack_id),
            ).fetchone()
        if row is None:
            return None
        try:
            meta = json.loads(row[0])
            version = meta.get("pack_version")
            return str(version) if version is not None else None
        except (json.JSONDecodeError, TypeError):
            return None

    def health_check(self) -> tuple[str, str]:
        try:
            with self._lock:
                self._conn.execute("SELECT 1").fetchone()
            return ("ok", "postgres reachable")
        except Exception as exc:
            return ("degraded", f"postgres unreachable: {exc}")

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception as exc:
            logger.debug("Postgres close failed (ignored): %s", exc)

    @staticmethod
    def _row_to_dict(row: tuple) -> dict[str, Any]:
        id_, run_id, query, result_json, metadata_json, created_at = row
        try:
            result = json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            result = {}
        try:
            metadata = json.loads(metadata_json)
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        return {
            "id": id_,
            "run_id": run_id,
            "query": query,
            "result": result,
            "metadata": metadata,
            "created_at": created_at,
        }


# ---------------------------------------------------------------------------
# Run history factory
# ---------------------------------------------------------------------------


def create_run_history(
    settings: Settings,
) -> ConversationMemory | RedisRunHistory | PostgresRunHistory:
    """Create the appropriate run history store based on settings.

    When the memory backend is Redis or Postgres AND the required
    packages are installed, run history is stored in the same backend.
    Falls back to SQLite otherwise.

    Args:
        settings: Application settings.

    Returns:
        A run history store instance.
    """
    backend = settings.memory_backend

    if backend == MemoryBackend.REDIS and settings.redis_url:
        try:
            return RedisRunHistory(settings.redis_url)
        except (ImportError, Exception) as exc:
            logger.warning(
                "Failed to create RedisRunHistory, falling back to SQLite: %s",
                exc,
            )

    if backend == MemoryBackend.POSTGRES and settings.postgres_url:
        try:
            return PostgresRunHistory(settings.postgres_url)
        except (ImportError, Exception) as exc:
            logger.warning(
                "Failed to create PostgresRunHistory, falling back to SQLite: %s",
                exc,
            )

    return ConversationMemory(
        settings.sqlite_path,
        backend=backend.value,
        redis_url=settings.redis_url,
        postgres_url=settings.postgres_url,
    )
