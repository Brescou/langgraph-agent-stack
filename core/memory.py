"""
core/memory.py — Memory and checkpointing layer for the LangGraph agent stack.

This module centralises all persistence concerns:

* ``create_checkpointer`` — factory that returns the appropriate LangGraph
  ``BaseCheckpointSaver`` based on ``Settings.memory_backend``.
* ``ConversationMemory`` — thin persistence layer for run-level history stored
  in a SQLite ``runs`` table.  Keeps a full audit trail of every agent
  invocation without coupling to the LangGraph checkpoint format.

Backend matrix
--------------
+-------------------+-------------------------------+---------------------------+
| ``memory_backend``| Checkpointer                  | Notes                     |
+===================+===============================+===========================+
| ``sqlite``        | ``SqliteSaver``               | Requires                  |
|                   |                               | ``langgraph-checkpoint-   |
|                   |                               | sqlite`` package.         |
+-------------------+-------------------------------+---------------------------+
| ``redis``         | ``RedisSaver``                | Requires                  |
|                   |                               | ``langgraph-checkpoint-   |
|                   |                               | redis`` package.          |
+-------------------+-------------------------------+---------------------------+
| fallback / error  | ``MemorySaver``               | In-process only; loses    |
|                   |                               | state on restart.         |
+-------------------+-------------------------------+---------------------------+

Usage example::

    from core.memory import create_checkpointer, ConversationMemory
    from core.config import settings

    checkpointer = create_checkpointer(settings)

    memory = ConversationMemory(settings.sqlite_path)
    with memory:
        memory.save_run(run_id, query, result, metadata)
        run = memory.get_run(run_id)
        recent = memory.list_runs(limit=5)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Generator, Optional, Type

from langgraph.checkpoint.memory import MemorySaver

from core.config import MemoryBackend, Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checkpointer factory
# ---------------------------------------------------------------------------


def create_checkpointer(settings: Settings) -> Any:
    """
    Construct and return a LangGraph checkpoint saver based on ``settings``.

    The function attempts to import the backend-specific saver package.  If
    the package is not installed, or if the backend is unrecognised, it falls
    back to the in-process ``MemorySaver`` and logs a warning so the issue is
    observable without crashing the application.

    Args:
        settings: The application ``Settings`` instance.  Read fields:
            ``memory_backend``, ``sqlite_path``, ``redis_url``.

    Returns:
        A configured LangGraph ``BaseCheckpointSaver`` instance.  The exact
        concrete type depends on the resolved backend:

        * ``SqliteSaver`` — when ``memory_backend == MemoryBackend.SQLITE``
          and ``langgraph-checkpoint-sqlite`` is installed.
        * ``RedisSaver``  — when ``memory_backend == MemoryBackend.REDIS``
          and ``langgraph-checkpoint-redis`` is installed.
        * ``MemorySaver`` — fallback for all other cases.
    """
    backend = settings.memory_backend

    if backend == MemoryBackend.SQLITE:
        return _create_sqlite_checkpointer(settings.sqlite_path)

    if backend == MemoryBackend.REDIS:
        return _create_redis_checkpointer(settings.redis_url)

    logger.warning(
        "Unknown memory_backend %r — falling back to MemorySaver.",
        backend,
    )
    return MemorySaver()


def _create_sqlite_checkpointer(sqlite_path: str) -> Any:
    """
    Build a ``SqliteSaver`` checkpointer backed by the given file path.

    The parent directory is created automatically if it does not exist.
    Falls back to ``MemorySaver`` when the ``langgraph-checkpoint-sqlite``
    package is not installed.

    Args:
        sqlite_path: Filesystem path to the SQLite database file.

    Returns:
        A ``SqliteSaver`` instance, or a ``MemorySaver`` on import failure.
    """
    db_path = Path(sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import]

        conn = sqlite3.connect(str(db_path), check_same_thread=False)
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
        return MemorySaver()


def _create_redis_checkpointer(redis_url: str) -> Any:
    """
    Build a ``RedisSaver`` checkpointer connected to ``redis_url``.

    Falls back to ``MemorySaver`` when the ``langgraph-checkpoint-redis``
    package is not installed.

    Args:
        redis_url: Redis connection URL (e.g. ``redis://localhost:6379/0``).

    Returns:
        A ``RedisSaver`` instance, or a ``MemorySaver`` on import failure.
    """
    try:
        from langgraph.checkpoint.redis import RedisSaver  # type: ignore[import]

        checkpointer = RedisSaver.from_conn_string(redis_url)
        logger.info(
            "Checkpointer: RedisSaver initialised",
            extra={"url": redis_url},
        )
        return checkpointer

    except ImportError:
        logger.warning(
            "langgraph-checkpoint-redis not installed — falling back to "
            "MemorySaver.  Install with: pip install langgraph-checkpoint-redis",
            extra={"redis_url": redis_url},
        )
        return MemorySaver()


# ---------------------------------------------------------------------------
# ConversationMemory — run-history persistence
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
    Persistent run-history store backed by a local SQLite database.

    Each call to ``save_run`` writes a single row to the ``runs`` table.
    ``get_run`` and ``list_runs`` provide read access to the stored history.

    The class can be used as a context manager to ensure the database
    connection is closed after a block of work:

    ::

        with ConversationMemory("./data/memory.db") as mem:
            mem.save_run(run_id, query, result, metadata)

    It can also be used standalone, with ``close()`` called explicitly when
    the memory object is no longer needed.

    Args:
        db_path: Filesystem path to the SQLite database.  The parent
            directory is created automatically.

    Attributes:
        db_path: Resolved absolute path to the database file.

    Raises:
        sqlite3.Error: If the database cannot be opened or the schema
            cannot be applied.
    """

    def __init__(self, db_path: str) -> None:
        # ":memory:" is SQLite's special token for a pure in-memory database.
        # Resolving it to an absolute path would turn it into a real filename
        # (e.g. /project/:memory:), so we preserve the token as-is.
        if db_path == ":memory:":
            self.db_path: Path = Path(db_path)
        else:
            self.db_path = Path(db_path).resolve()
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage transactions explicitly
        )
        self._conn.row_factory = sqlite3.Row
        self._apply_schema()

        logger.debug(
            "ConversationMemory initialised",
            extra={"db_path": str(self.db_path)},
        )

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "ConversationMemory":
        """Return self so the instance can be used as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
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
        self._conn.execute("BEGIN")
        try:
            yield
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_run(
        self,
        run_id: str,
        query: str,
        result: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
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
        metadata_json = json.dumps(
            metadata or {}, ensure_ascii=False, default=str
        )
        created_at = datetime.now(timezone.utc).isoformat()

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

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
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

        rows = self._conn.execute(
            "SELECT id, run_id, query, result_json, metadata_json, created_at "
            "FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

        return [self._row_to_dict(row) for row in rows]

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
        except Exception:
            pass  # Already closed or never opened — silently swallow

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
