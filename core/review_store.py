"""
core/review_store.py — Human-review queue for regulated pack outputs.

Regulated vertical packs (talent_screening, contract_reviewer, …) return
``human_review_required=True`` on every output (see
``domain_packs/common/compliance.py``). This module provides the missing
compliance loop: a small persistent queue of pending reviews that a human
reviewer can list, inspect, and decide (approve / reject) via the
``/reviews`` API endpoints.

Design notes:
    * ``ReviewRecord.output_summary`` stores only a **truncated** summary of
      the pack output, never the full payload — regulated outputs may contain
      sensitive personal data already persisted by the run-history backend,
      and the review queue must not duplicate it.
    * ``decide()`` is conflict-safe: deciding an already-decided review
      raises :class:`ReviewAlreadyDecidedError` (mapped to HTTP 409).
    * Two backends mirror the rate-limiter pattern (``core/security.py``):
      ``InMemoryReviewStore`` (default, per-process) and ``SqliteReviewStore``
      (survives restarts; same WAL pragmas as ``core/memory.py``).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

#: Review lifecycle states.
ReviewStatus = Literal["pending", "approved", "rejected"]

#: Decisions a reviewer may record (a review can never go back to pending).
ReviewDecision = Literal["approved", "rejected"]

#: Maximum characters kept in ``output_summary`` (full payloads are never stored).
OUTPUT_SUMMARY_MAX_CHARS = 500

_VALID_DECISIONS: frozenset[str] = frozenset({"approved", "rejected"})


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def summarize_output(payload: Any) -> str:
    """Build a truncated, JSON-ish summary of a pack output payload.

    Only the first :data:`OUTPUT_SUMMARY_MAX_CHARS` characters are kept so the
    review queue never duplicates full (potentially sensitive) pack outputs.
    """
    if payload is None:
        return ""
    try:
        text = json.dumps(payload, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(payload)
    return text[:OUTPUT_SUMMARY_MAX_CHARS]


class ReviewRecord(BaseModel):
    """A single human-review queue entry for one regulated pack run."""

    run_id: str = Field(description="Run identifier the review refers to.")
    pack_id: str = Field(description="Regulated pack that produced the output.")
    session_id: str | None = Field(
        default=None, description="Optional session the run belonged to."
    )
    status: ReviewStatus = Field(
        default="pending", description="Review lifecycle state."
    )
    output_summary: str = Field(
        default="",
        description=("Truncated summary of the pack output (never the full payload)."),
    )
    created_at: str = Field(description="ISO-8601 UTC timestamp of creation.")
    decided_at: str | None = Field(
        default=None, description="ISO-8601 UTC timestamp of the decision."
    )
    reviewer: str | None = Field(
        default=None, description="Identifier of the human who decided."
    )
    notes: str | None = Field(
        default=None, description="Optional free-text reviewer notes."
    )
    reason: str | None = Field(
        default=None,
        description=(
            "Why the run was queued when the pack does not mandate review, "
            "e.g. a guardrail escalation."
        ),
    )


class ReviewNotFoundError(KeyError):
    """Raised when no review exists for the given run_id."""


class ReviewAlreadyDecidedError(RuntimeError):
    """Raised when deciding a review that has already been decided (conflict)."""


@runtime_checkable
class ReviewStoreBackend(Protocol):
    """Protocol for human-review queue backends (in-memory or SQLite)."""

    def create(
        self,
        *,
        run_id: str,
        pack_id: str,
        session_id: str | None = None,
        output_summary: str = "",
        reason: str | None = None,
    ) -> ReviewRecord:
        """Create a pending review for *run_id*. Raises ValueError on duplicate."""
        ...

    def get(self, run_id: str) -> ReviewRecord | None:
        """Return the review for *run_id*, or None when unknown."""
        ...

    def list_reviews(
        self,
        *,
        status: ReviewStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ReviewRecord]:
        """Return reviews newest-first, optionally filtered by status."""
        ...

    def decide(
        self,
        run_id: str,
        *,
        status: ReviewDecision,
        reviewer: str,
        notes: str | None = None,
    ) -> ReviewRecord:
        """Record a decision. Raises ReviewNotFoundError / ReviewAlreadyDecidedError."""
        ...

    def close(self) -> None:
        """Release backend resources (no-op for the in-memory store)."""
        ...


def _validate_decision_args(status: str, reviewer: str) -> None:
    if status not in _VALID_DECISIONS:
        raise ValueError(
            f"decide: status must be 'approved' or 'rejected', got {status!r}."
        )
    if not reviewer or not reviewer.strip():
        raise ValueError("decide: reviewer must not be empty.")


def _validate_list_args(limit: int, offset: int) -> None:
    if limit < 1:
        raise ValueError(f"list_reviews: limit must be >= 1, got {limit}.")
    if offset < 0:
        raise ValueError(f"list_reviews: offset must be >= 0, got {offset}.")


class InMemoryReviewStore:
    """Thread-safe in-memory review queue (default — dev / single process)."""

    def __init__(self) -> None:
        self._records: dict[str, ReviewRecord] = {}
        self._lock = threading.Lock()

    def create(
        self,
        *,
        run_id: str,
        pack_id: str,
        session_id: str | None = None,
        output_summary: str = "",
        reason: str | None = None,
    ) -> ReviewRecord:
        if not run_id or not run_id.strip():
            raise ValueError("create: run_id must not be empty.")
        if not pack_id or not pack_id.strip():
            raise ValueError("create: pack_id must not be empty.")
        record = ReviewRecord(
            run_id=run_id,
            pack_id=pack_id,
            session_id=session_id,
            output_summary=output_summary[:OUTPUT_SUMMARY_MAX_CHARS],
            created_at=_utc_now_iso(),
            reason=reason,
        )
        with self._lock:
            if run_id in self._records:
                raise ValueError(f"create: review for run_id {run_id!r} exists.")
            self._records[run_id] = record
        return record

    def get(self, run_id: str) -> ReviewRecord | None:
        with self._lock:
            record = self._records.get(run_id)
        return record.model_copy() if record is not None else None

    def list_reviews(
        self,
        *,
        status: ReviewStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ReviewRecord]:
        _validate_list_args(limit, offset)
        with self._lock:
            # dicts preserve insertion order → reversed = newest first.
            records = [
                r.model_copy()
                for r in reversed(self._records.values())
                if status is None or r.status == status
            ]
        return records[offset : offset + limit]

    def decide(
        self,
        run_id: str,
        *,
        status: ReviewDecision,
        reviewer: str,
        notes: str | None = None,
    ) -> ReviewRecord:
        _validate_decision_args(status, reviewer)
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                raise ReviewNotFoundError(f"No review found for run_id {run_id!r}.")
            if record.status != "pending":
                raise ReviewAlreadyDecidedError(
                    f"Review for run_id {run_id!r} was already decided "
                    f"({record.status})."
                )
            updated = record.model_copy(
                update={
                    "status": status,
                    "decided_at": _utc_now_iso(),
                    "reviewer": reviewer.strip(),
                    "notes": notes,
                }
            )
            self._records[run_id] = updated
        return updated.model_copy()

    def close(self) -> None:
        """No resources to release for the in-memory store."""


class SqliteReviewStore:
    """SQLite-backed review queue — survives process restarts.

    Uses the same WAL pragmas as ``core/memory.py`` so concurrent readers
    (review endpoints) never block the writer (run completion hook).
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL UNIQUE,
        pack_id TEXT NOT NULL,
        session_id TEXT,
        status TEXT NOT NULL DEFAULT 'pending'
            CHECK (status IN ('pending', 'approved', 'rejected')),
        output_summary TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        decided_at TEXT,
        reviewer TEXT,
        notes TEXT,
        reason TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_reviews_status_id ON reviews (status, id DESC);
    """

    _COLUMNS = (
        "run_id, pack_id, session_id, status, output_summary, "
        "created_at, decided_at, reviewer, notes, reason"
    )

    def __init__(self, db_path: str = "./data/review_store.db") -> None:
        if db_path == ":memory:":
            self.db_path = Path(":memory:")
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
        # WAL mode — same pragmas as core/memory.py for concurrent access.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA cache_size=1000")
        self._conn.isolation_level = "DEFERRED"
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(self._SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created."""
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(reviews)").fetchall()
        }
        if "reason" not in existing:
            self._conn.execute("ALTER TABLE reviews ADD COLUMN reason TEXT")

    def create(
        self,
        *,
        run_id: str,
        pack_id: str,
        session_id: str | None = None,
        output_summary: str = "",
        reason: str | None = None,
    ) -> ReviewRecord:
        if not run_id or not run_id.strip():
            raise ValueError("create: run_id must not be empty.")
        if not pack_id or not pack_id.strip():
            raise ValueError("create: pack_id must not be empty.")
        created_at = _utc_now_iso()
        summary = output_summary[:OUTPUT_SUMMARY_MAX_CHARS]
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO reviews "
                    "(run_id, pack_id, session_id, status, output_summary, "
                    "created_at, reason) "
                    "VALUES (?, ?, ?, 'pending', ?, ?, ?)",
                    (run_id, pack_id, session_id, summary, created_at, reason),
                )
                self._conn.commit()
            except sqlite3.IntegrityError as exc:
                self._conn.rollback()
                raise ValueError(
                    f"create: review for run_id {run_id!r} exists."
                ) from exc
            except sqlite3.Error:
                self._conn.rollback()
                raise
        return ReviewRecord(
            run_id=run_id,
            pack_id=pack_id,
            session_id=session_id,
            output_summary=summary,
            created_at=created_at,
            reason=reason,
        )

    def get(self, run_id: str) -> ReviewRecord | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {self._COLUMNS} FROM reviews WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def list_reviews(
        self,
        *,
        status: ReviewStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ReviewRecord]:
        _validate_list_args(limit, offset)
        base = f"SELECT {self._COLUMNS} FROM reviews"
        with self._lock:
            if status is None:
                rows = self._conn.execute(
                    f"{base} ORDER BY id DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    f"{base} WHERE status = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                    (status, limit, offset),
                ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def decide(
        self,
        run_id: str,
        *,
        status: ReviewDecision,
        reviewer: str,
        notes: str | None = None,
    ) -> ReviewRecord:
        _validate_decision_args(status, reviewer)
        decided_at = _utc_now_iso()
        with self._lock:
            try:
                cursor = self._conn.execute(
                    "UPDATE reviews SET status = ?, decided_at = ?, reviewer = ?, "
                    "notes = ? WHERE run_id = ? AND status = 'pending'",
                    (status, decided_at, reviewer.strip(), notes, run_id),
                )
                self._conn.commit()
            except sqlite3.Error:
                self._conn.rollback()
                raise
            if cursor.rowcount == 0:
                row = self._conn.execute(
                    "SELECT status FROM reviews WHERE run_id = ?", (run_id,)
                ).fetchone()
                if row is None:
                    raise ReviewNotFoundError(f"No review found for run_id {run_id!r}.")
                raise ReviewAlreadyDecidedError(
                    f"Review for run_id {run_id!r} was already decided "
                    f"({row['status']})."
                )
        record = self.get(run_id)
        if record is None:  # pragma: no cover — row was just updated
            raise ReviewNotFoundError(f"No review found for run_id {run_id!r}.")
        return record

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ReviewRecord:
        return ReviewRecord(
            run_id=row["run_id"],
            pack_id=row["pack_id"],
            session_id=row["session_id"],
            status=row["status"],
            output_summary=row["output_summary"],
            created_at=row["created_at"],
            decided_at=row["decided_at"],
            reviewer=row["reviewer"],
            notes=row["notes"],
            reason=row["reason"],
        )


def create_review_store(
    backend: Literal["memory", "sqlite"] = "memory",
    sqlite_path: str = "./data/review_store.db",
) -> ReviewStoreBackend:
    """Factory: build a review store matching the configured backend.

    Args:
        backend: ``"memory"`` (default, per-process) or ``"sqlite"``
            (persists decisions across restarts).
        sqlite_path: Database file path used when ``backend="sqlite"``.

    Returns:
        A review store instance satisfying ``ReviewStoreBackend``.
    """
    if backend == "sqlite":
        return SqliteReviewStore(db_path=sqlite_path)
    return InMemoryReviewStore()
