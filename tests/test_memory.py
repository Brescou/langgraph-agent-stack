"""
tests/test_memory.py — Unit tests for core/memory.py ConversationMemory.

Each test receives an isolated ConversationMemory instance backed by a
temporary SQLite file that is deleted after the test completes.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any

import pytest

from core.memory import ConversationMemory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def memory() -> ConversationMemory:
    """
    Return a fresh ConversationMemory instance backed by a temporary SQLite file.

    A new temporary file is created for every test function so tests are fully
    isolated from one another.  The file is removed after the test completes.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    mem = ConversationMemory(db_path)
    yield mem
    mem.close()
    Path(db_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_run_id() -> str:
    """Return a new UUID string suitable as a run_id."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# save_run / get_run
# ---------------------------------------------------------------------------


def test_save_and_get_run(memory: ConversationMemory) -> None:
    """save_run followed by get_run must return a record with the correct fields."""
    run_id = _make_run_id()
    query = "What is quantum computing?"
    result: dict[str, Any] = {"summary": "Quantum computers use qubits."}
    metadata: dict[str, Any] = {"agent": "ResearchAgent", "confidence": 0.85}

    memory.save_run(run_id, query, result, metadata)

    record = memory.get_run(run_id)

    assert record is not None
    assert record["run_id"] == run_id
    assert record["query"] == query
    assert record["result"] == result
    assert record["metadata"] == metadata
    assert "created_at" in record
    assert "id" in record


def test_save_run_returns_correct_query(memory: ConversationMemory) -> None:
    """The stored query must exactly match the input after stripping."""
    run_id = _make_run_id()
    query = "  CAP theorem in distributed systems  "

    memory.save_run(run_id, query, {})

    record = memory.get_run(run_id)
    assert record is not None
    assert record["query"] == query.strip()


def test_save_run_overwrites_existing(memory: ConversationMemory) -> None:
    """Saving with the same run_id must overwrite the previous record."""
    run_id = _make_run_id()

    memory.save_run(run_id, "first query", {"summary": "first"})
    memory.save_run(run_id, "updated query", {"summary": "updated"})

    record = memory.get_run(run_id)
    assert record is not None
    assert record["query"] == "updated query"
    assert record["result"]["summary"] == "updated"


def test_save_run_without_metadata(memory: ConversationMemory) -> None:
    """Omitting metadata must store an empty dict without raising."""
    run_id = _make_run_id()

    memory.save_run(run_id, "test query", {"key": "value"})

    record = memory.get_run(run_id)
    assert record is not None
    assert record["metadata"] == {}


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


def test_list_runs_returns_results(memory: ConversationMemory) -> None:
    """list_runs must return saved runs as a non-empty list."""
    for i in range(3):
        memory.save_run(_make_run_id(), f"query {i}", {"summary": f"result {i}"})

    runs = memory.list_runs()

    assert isinstance(runs, list)
    assert len(runs) == 3


def test_list_runs_limit(memory: ConversationMemory) -> None:
    """list_runs must respect the limit parameter and return at most limit records."""
    for i in range(5):
        memory.save_run(_make_run_id(), f"query {i}", {})

    runs = memory.list_runs(limit=2)

    assert len(runs) == 2


def test_list_runs_empty_database(memory: ConversationMemory) -> None:
    """list_runs on an empty database must return an empty list without raising."""
    runs = memory.list_runs()

    assert runs == []


def test_list_runs_default_limit(memory: ConversationMemory) -> None:
    """list_runs default limit of 10 must not return more than 10 records."""
    for i in range(15):
        memory.save_run(_make_run_id(), f"query {i}", {})

    runs = memory.list_runs()

    assert len(runs) <= 10


def test_list_runs_invalid_limit(memory: ConversationMemory) -> None:
    """list_runs with limit < 1 must raise ValueError."""
    with pytest.raises(ValueError, match="limit must be >= 1"):
        memory.list_runs(limit=0)


# ---------------------------------------------------------------------------
# get_nonexistent_run
# ---------------------------------------------------------------------------


def test_get_nonexistent_run(memory: ConversationMemory) -> None:
    """get_run for a run_id that does not exist must return None."""
    result = memory.get_run("nonexistent-run-id-xyz")

    assert result is None


def test_get_nonexistent_run_uuid(memory: ConversationMemory) -> None:
    """get_run with a plausible UUID that was never saved must return None."""
    result = memory.get_run(_make_run_id())

    assert result is None


# ---------------------------------------------------------------------------
# Context manager protocol
# ---------------------------------------------------------------------------


def test_context_manager_usage() -> None:
    """ConversationMemory must work as a context manager and close cleanly."""
    run_id = _make_run_id()

    with ConversationMemory(":memory:") as mem:
        mem.save_run(run_id, "context manager test", {"ok": True})
        record = mem.get_run(run_id)
        assert record is not None
        assert record["run_id"] == run_id
    # After __exit__ the connection is closed — no exception should be raised


# ---------------------------------------------------------------------------
# save_run validation
# ---------------------------------------------------------------------------


def test_save_run_empty_run_id_raises(memory: ConversationMemory) -> None:
    """save_run with an empty run_id must raise ValueError."""
    with pytest.raises(ValueError, match="run_id must not be empty"):
        memory.save_run("", "some query", {})


def test_save_run_empty_query_raises(memory: ConversationMemory) -> None:
    """save_run with an empty query must raise ValueError."""
    with pytest.raises(ValueError, match="query must not be empty"):
        memory.save_run(_make_run_id(), "   ", {})
