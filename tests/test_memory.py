"""
tests/test_memory.py — Unit tests for core/memory.py ConversationMemory.

Each test receives an isolated ConversationMemory instance backed by a
temporary SQLite file that is deleted after the test completes.
"""

from __future__ import annotations

import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.config import MemoryBackend
from core.memory import (
    ConversationMemory,
    PostgresRunHistory,
    RedisRunHistory,
    create_run_history,
)

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


# ---------------------------------------------------------------------------
# list_runs_by_session
# ---------------------------------------------------------------------------


def test_list_runs_by_session(memory: ConversationMemory) -> None:
    """list_runs_by_session should only return runs matching the session_id."""
    memory.save_run("run-1", "query 1", {"result": "r1"}, {"session_id": "session-A"})
    memory.save_run("run-2", "query 2", {"result": "r2"}, {"session_id": "session-B"})
    memory.save_run("run-3", "query 3", {"result": "r3"}, {"session_id": "session-A"})

    runs = memory.list_runs_by_session("session-A")
    assert len(runs) == 2
    for run in runs:
        assert run["metadata"]["session_id"] == "session-A"


def test_list_runs_by_session_empty(memory: ConversationMemory) -> None:
    """list_runs_by_session for nonexistent session returns empty list."""
    runs = memory.list_runs_by_session("no-such-session")
    assert runs == []


# ---------------------------------------------------------------------------
# get_pack_version_for_session
# ---------------------------------------------------------------------------


def test_get_pack_version_for_session_returns_none_when_no_history():
    mem = ConversationMemory(":memory:")
    result = mem.get_pack_version_for_session("session-1", "research_analysis")
    assert result is None
    mem.close()


def test_get_pack_version_for_session_returns_version():
    mem = ConversationMemory(":memory:")
    mem.save_run(
        run_id="run-1",
        query="test query",
        result={},
        metadata={
            "session_id": "session-1",
            "pack_id": "research_analysis",
            "pack_version": "1.0",
        },
    )
    result = mem.get_pack_version_for_session("session-1", "research_analysis")
    assert result == "1.0"
    mem.close()


def test_get_pack_version_for_session_returns_most_recent():
    """When multiple runs exist, the most recent pack_version is returned."""
    mem = ConversationMemory(":memory:")
    import time

    mem.save_run(
        run_id="run-1",
        query="q1",
        result={},
        metadata={"session_id": "s1", "pack_id": "ra", "pack_version": "1.0"},
    )
    time.sleep(0.01)
    mem.save_run(
        run_id="run-2",
        query="q2",
        result={},
        metadata={"session_id": "s1", "pack_id": "ra", "pack_version": "2.0"},
    )
    result = mem.get_pack_version_for_session("s1", "ra")
    assert result == "2.0"
    mem.close()


def test_get_pack_version_for_session_filters_by_pack_id():
    """Different pack_ids for same session are isolated."""
    mem = ConversationMemory(":memory:")
    mem.save_run(
        run_id="run-1",
        query="q",
        result={},
        metadata={"session_id": "s1", "pack_id": "pack_a", "pack_version": "1.0"},
    )
    mem.save_run(
        run_id="run-2",
        query="q",
        result={},
        metadata={"session_id": "s1", "pack_id": "pack_b", "pack_version": "2.0"},
    )
    assert mem.get_pack_version_for_session("s1", "pack_a") == "1.0"
    assert mem.get_pack_version_for_session("s1", "pack_b") == "2.0"
    mem.close()


def test_get_pack_version_for_session_cross_session_isolation():
    """Different sessions do not bleed into each other."""
    mem = ConversationMemory(":memory:")
    mem.save_run(
        run_id="run-1",
        query="q",
        result={},
        metadata={"session_id": "session-A", "pack_id": "ra", "pack_version": "1.0"},
    )
    result = mem.get_pack_version_for_session("session-B", "ra")
    assert result is None
    mem.close()


# ---------------------------------------------------------------------------
# close() idempotent
# ---------------------------------------------------------------------------


def test_close_idempotent() -> None:
    """Calling close() multiple times should not raise."""
    mem = ConversationMemory(":memory:")
    mem.close()
    mem.close()  # Should not raise


# ---------------------------------------------------------------------------
# Corrupted JSON handling
# ---------------------------------------------------------------------------


def test_get_run_with_corrupted_json(memory: ConversationMemory) -> None:
    """A run with corrupted result JSON should return an empty dict for result."""
    run_id = _make_run_id()
    memory._conn.execute(
        "INSERT INTO runs (run_id, query, result_json, metadata_json, created_at) "
        "VALUES (?, ?, ?, ?, datetime('now'))",
        (run_id, "test query", "{invalid json", "{}"),
    )
    memory._conn.commit()

    record = memory.get_run(run_id)
    assert record is not None
    assert record["result"] == {}


def test_create_checkpointer_redis_with_mock():
    """create_checkpointer returns RedisSaver when redis package is available."""
    from unittest.mock import MagicMock, patch

    mock_redis_saver = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_redis_saver)
    mock_redis_module = MagicMock()
    mock_redis_module.RedisSaver.from_conn_string.return_value = mock_ctx

    with patch.dict(
        "sys.modules",
        {"langgraph.checkpoint.redis": mock_redis_module},
    ):
        from core.memory import _create_redis_checkpointer

        result = _create_redis_checkpointer("redis://localhost:6379/0")

    assert result is mock_redis_saver
    mock_redis_module.RedisSaver.from_conn_string.assert_called_once_with(
        "redis://localhost:6379/0"
    )


def test_create_checkpointer_postgres_missing_url():
    """create_checkpointer falls back when postgres_url is None in development."""
    from unittest.mock import MagicMock, patch

    mock_settings = MagicMock()
    mock_settings.environment = "development"

    with patch("core.config.get_settings", return_value=mock_settings):
        from core.memory import _create_postgres_checkpointer

        result = _create_postgres_checkpointer(None)

    from langgraph.checkpoint.memory import MemorySaver

    assert isinstance(result, MemorySaver)


def test_create_checkpointer_postgres_sync_with_setup():
    """PostgresSaver (sync) is used and setup() is called on creation."""
    from unittest.mock import MagicMock, patch

    mock_saver = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_saver)
    mock_pg_module = MagicMock()
    mock_pg_module.PostgresSaver.from_conn_string.return_value = mock_ctx

    with patch.dict(
        "sys.modules",
        {"langgraph.checkpoint.postgres": mock_pg_module},
    ):
        from core.memory import _create_postgres_checkpointer

        result = _create_postgres_checkpointer(
            "postgresql+psycopg://u:p@localhost:5432/db"
        )

    assert result is mock_saver
    mock_pg_module.PostgresSaver.from_conn_string.assert_called_once_with(
        "postgresql+psycopg://u:p@localhost:5432/db"
    )
    mock_saver.setup.assert_called_once()


# ---------------------------------------------------------------------------
# RedisRunHistory tests
# ---------------------------------------------------------------------------


class TestRedisRunHistory:
    """Unit tests for RedisRunHistory with fully mocked redis."""

    @staticmethod
    def _make_store() -> tuple[RedisRunHistory, MagicMock]:
        """Create a RedisRunHistory bypassing __init__, with a mock redis client."""
        store = RedisRunHistory.__new__(RedisRunHistory)
        mock_redis_instance = MagicMock()
        store._redis = mock_redis_instance
        store._prefix = "runhistory"
        return store, mock_redis_instance

    def test_save_and_get_run(self) -> None:
        """save_run stores via pipeline; get_run decodes the hash."""
        store, mock_redis = self._make_store()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe

        store.save_run(
            "run-1", "test query", {"summary": "result"}, {"session_id": "s1"}
        )

        mock_pipe.hset.assert_called_once()
        assert mock_pipe.hset.call_args[0][0] == "runhistory:run:run-1"
        assert mock_pipe.hset.call_args[1]["mapping"]["run_id"] == "run-1"
        assert mock_pipe.zadd.call_count == 2  # timeline + session
        mock_pipe.execute.assert_called_once()

        mock_redis.hgetall.return_value = {
            "run_id": "run-1",
            "query": "test query",
            "result_json": '{"summary": "result"}',
            "metadata_json": '{"session_id": "s1"}',
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        result = store.get_run("run-1")
        assert result is not None
        assert result["run_id"] == "run-1"
        assert result["result"]["summary"] == "result"
        assert result["metadata"]["session_id"] == "s1"

    def test_save_run_empty_run_id_raises(self) -> None:
        store, _ = self._make_store()
        with pytest.raises(ValueError, match="run_id must not be empty"):
            store.save_run("", "some query", {})

    def test_save_run_empty_query_raises(self) -> None:
        store, _ = self._make_store()
        with pytest.raises(ValueError, match="query must not be empty"):
            store.save_run("run-1", "   ", {})

    def test_list_runs(self) -> None:
        """list_runs fetches IDs from timeline sorted set, then each hash."""
        store, mock_redis = self._make_store()
        mock_redis.zrevrange.return_value = ["run-2", "run-1"]

        def hgetall_effect(key: str) -> dict[str, str]:
            if "run-2" in key:
                return {
                    "run_id": "run-2",
                    "query": "q2",
                    "result_json": "{}",
                    "metadata_json": "{}",
                    "created_at": "2026-01-01T00:00:02",
                }
            if "run-1" in key:
                return {
                    "run_id": "run-1",
                    "query": "q1",
                    "result_json": "{}",
                    "metadata_json": "{}",
                    "created_at": "2026-01-01T00:00:01",
                }
            return {}

        mock_redis.hgetall.side_effect = hgetall_effect

        runs = store.list_runs(limit=10)
        assert len(runs) == 2
        assert runs[0]["run_id"] == "run-2"
        assert runs[1]["run_id"] == "run-1"
        mock_redis.zrevrange.assert_called_once_with("runhistory:timeline", 0, 9)

    def test_list_runs_invalid_limit_raises(self) -> None:
        store, _ = self._make_store()
        with pytest.raises(ValueError, match="limit must be >= 1"):
            store.list_runs(limit=0)

    def test_list_runs_by_session(self) -> None:
        """list_runs_by_session uses the per-session sorted set."""
        store, mock_redis = self._make_store()
        mock_redis.zrevrange.return_value = ["run-3", "run-1"]

        def hgetall_effect(key: str) -> dict[str, str]:
            if "run-3" in key:
                return {
                    "run_id": "run-3",
                    "query": "q3",
                    "result_json": "{}",
                    "metadata_json": '{"session_id": "s1"}',
                    "created_at": "2026-01-01T00:00:03",
                }
            if "run-1" in key:
                return {
                    "run_id": "run-1",
                    "query": "q1",
                    "result_json": "{}",
                    "metadata_json": '{"session_id": "s1"}',
                    "created_at": "2026-01-01T00:00:01",
                }
            return {}

        mock_redis.hgetall.side_effect = hgetall_effect

        runs = store.list_runs_by_session("s1")
        assert len(runs) == 2
        assert runs[0]["run_id"] == "run-3"
        assert runs[1]["run_id"] == "run-1"
        mock_redis.zrevrange.assert_called_once_with("runhistory:session:s1", 0, 49)

    def test_list_runs_by_session_empty_session_raises(self) -> None:
        store, _ = self._make_store()
        with pytest.raises(ValueError, match="session_id must not be empty"):
            store.list_runs_by_session("")

    def test_get_run_not_found(self) -> None:
        """hgetall returning empty dict means run does not exist."""
        store, mock_redis = self._make_store()
        mock_redis.hgetall.return_value = {}
        result = store.get_run("nonexistent")
        assert result is None

    def test_health_check_ok(self) -> None:
        store, mock_redis = self._make_store()
        mock_redis.ping.return_value = True
        status, detail = store.health_check()
        assert status == "ok"
        assert "reachable" in detail

    def test_health_check_degraded(self) -> None:
        store, mock_redis = self._make_store()
        mock_redis.ping.side_effect = ConnectionError("connection refused")
        status, detail = store.health_check()
        assert status == "degraded"
        assert "unreachable" in detail

    def test_close(self) -> None:
        store, mock_redis = self._make_store()
        store.close()
        mock_redis.close.assert_called_once()

    def test_decode_corrupted_json(self) -> None:
        """_decode falls back to empty dicts when JSON is malformed."""
        data = {
            "run_id": "r1",
            "query": "q",
            "result_json": "{invalid",
            "metadata_json": "not-json",
            "created_at": "2026-01-01",
        }
        result = RedisRunHistory._decode(data)
        assert result["result"] == {}
        assert result["metadata"] == {}
        assert result["run_id"] == "r1"

    def test_list_runs_by_session_invalid_limit_raises(self) -> None:
        store, _ = self._make_store()
        with pytest.raises(ValueError, match="limit must be >= 1"):
            store.list_runs_by_session("s1", limit=0)

    def test_close_exception_swallowed(self) -> None:
        """close() silently swallows exceptions from the redis client."""
        store, mock_redis = self._make_store()
        mock_redis.close.side_effect = ConnectionError("already closed")
        store.close()  # must not raise

    def test_import_error(self) -> None:
        """Instantiation raises ImportError when redis is not installed."""
        with patch.dict(sys.modules, {"redis": None}):
            with pytest.raises(ImportError, match="redis package is required"):
                RedisRunHistory("redis://localhost:6379/0")


# ---------------------------------------------------------------------------
# PostgresRunHistory tests
# ---------------------------------------------------------------------------


class TestPostgresRunHistory:
    """Unit tests for PostgresRunHistory with fully mocked psycopg."""

    @staticmethod
    def _make_store() -> tuple[PostgresRunHistory, MagicMock]:
        """Create a PostgresRunHistory bypassing __init__, with a mock connection."""
        store = PostgresRunHistory.__new__(PostgresRunHistory)
        mock_conn = MagicMock()
        store._conn = mock_conn
        store._lock = threading.Lock()
        return store, mock_conn

    def test_save_and_get_run(self) -> None:
        """save_run inserts via execute; get_run fetches a tuple row."""
        store, mock_conn = self._make_store()

        store.save_run(
            "run-1", "test query", {"summary": "result"}, {"session_id": "s1"}
        )

        mock_conn.execute.assert_called_once()
        sql = mock_conn.execute.call_args[0][0]
        assert "INSERT INTO run_history" in sql

        mock_conn.execute.reset_mock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            1,
            "run-1",
            "test query",
            '{"summary": "result"}',
            '{"session_id": "s1"}',
            "2026-01-01T00:00:00+00:00",
        )
        mock_conn.execute.return_value = mock_cursor

        result = store.get_run("run-1")
        assert result is not None
        assert result["run_id"] == "run-1"
        assert result["result"]["summary"] == "result"
        assert result["metadata"]["session_id"] == "s1"

    def test_save_run_empty_run_id_raises(self) -> None:
        store, _ = self._make_store()
        with pytest.raises(ValueError, match="run_id must not be empty"):
            store.save_run("", "some query", {})

    def test_save_run_empty_query_raises(self) -> None:
        store, _ = self._make_store()
        with pytest.raises(ValueError, match="query must not be empty"):
            store.save_run("run-1", "   ", {})

    def test_list_runs(self) -> None:
        """list_runs returns rows ordered by created_at DESC."""
        store, mock_conn = self._make_store()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (2, "run-2", "q2", "{}", "{}", "2026-01-01T00:00:02"),
            (1, "run-1", "q1", "{}", "{}", "2026-01-01T00:00:01"),
        ]
        mock_conn.execute.return_value = mock_cursor

        runs = store.list_runs(limit=10)
        assert len(runs) == 2
        assert runs[0]["run_id"] == "run-2"
        assert runs[1]["run_id"] == "run-1"

    def test_list_runs_invalid_limit_raises(self) -> None:
        store, _ = self._make_store()
        with pytest.raises(ValueError, match="limit must be >= 1"):
            store.list_runs(limit=0)

    def test_list_runs_by_session(self) -> None:
        """list_runs_by_session filters via SQL on session_id."""
        store, mock_conn = self._make_store()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (3, "run-3", "q3", "{}", '{"session_id": "s1"}', "2026-01-01T00:00:03"),
            (1, "run-1", "q1", "{}", '{"session_id": "s1"}', "2026-01-01T00:00:01"),
        ]
        mock_conn.execute.return_value = mock_cursor

        runs = store.list_runs_by_session("s1")
        assert len(runs) == 2
        assert all(r["metadata"]["session_id"] == "s1" for r in runs)

    def test_list_runs_by_session_empty_session_raises(self) -> None:
        store, _ = self._make_store()
        with pytest.raises(ValueError, match="session_id must not be empty"):
            store.list_runs_by_session("")

    def test_get_run_not_found(self) -> None:
        """fetchone returning None means run does not exist."""
        store, mock_conn = self._make_store()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.execute.return_value = mock_cursor

        result = store.get_run("nonexistent")
        assert result is None

    def test_health_check_ok(self) -> None:
        store, mock_conn = self._make_store()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.execute.return_value = mock_cursor

        status, detail = store.health_check()
        assert status == "ok"
        assert "reachable" in detail

    def test_health_check_degraded(self) -> None:
        store, mock_conn = self._make_store()
        mock_conn.execute.side_effect = Exception("connection refused")

        status, detail = store.health_check()
        assert status == "degraded"
        assert "unreachable" in detail

    def test_close(self) -> None:
        store, mock_conn = self._make_store()
        store.close()
        mock_conn.close.assert_called_once()

    def test_row_to_dict_corrupted_json(self) -> None:
        """_row_to_dict falls back to empty dicts when JSON is malformed."""
        row = (1, "run-1", "query", "{bad", "not-json", "2026-01-01")
        result = PostgresRunHistory._row_to_dict(row)
        assert result["result"] == {}
        assert result["metadata"] == {}
        assert result["run_id"] == "run-1"

    def test_list_runs_by_session_invalid_limit_raises(self) -> None:
        store, _ = self._make_store()
        with pytest.raises(ValueError, match="limit must be >= 1"):
            store.list_runs_by_session("s1", limit=0)

    def test_close_exception_swallowed(self) -> None:
        """close() silently swallows exceptions from the psycopg connection."""
        store, mock_conn = self._make_store()
        mock_conn.close.side_effect = Exception("already closed")
        store.close()  # must not raise

    def test_import_error(self) -> None:
        """Instantiation raises ImportError when psycopg is not installed."""
        with patch.dict(sys.modules, {"psycopg": None}):
            with pytest.raises(ImportError, match="psycopg is required"):
                PostgresRunHistory("postgresql://localhost:5432/db")


# ---------------------------------------------------------------------------
# create_run_history factory tests
# ---------------------------------------------------------------------------


class TestCreateRunHistory:
    """Unit tests for the create_run_history factory function."""

    def test_sqlite_backend_returns_conversation_memory(self, tmp_path: Any) -> None:
        settings = MagicMock()
        settings.memory_backend = MemoryBackend.SQLITE
        settings.sqlite_path = str(tmp_path / "test.db")
        settings.redis_url = None
        settings.postgres_url = None

        result = create_run_history(settings)
        assert isinstance(result, ConversationMemory)
        result.close()

    def test_redis_backend_returns_redis_history(self) -> None:
        mock_redis_mod = MagicMock()

        settings = MagicMock()
        settings.memory_backend = MemoryBackend.REDIS
        settings.redis_url = "redis://localhost:6379/0"

        with patch.dict(sys.modules, {"redis": mock_redis_mod}):
            result = create_run_history(settings)

        assert isinstance(result, RedisRunHistory)
        result.close()

    def test_postgres_backend_returns_postgres_history(self) -> None:
        mock_psycopg_mod = MagicMock()

        settings = MagicMock()
        settings.memory_backend = MemoryBackend.POSTGRES
        settings.postgres_url = "postgresql://localhost:5432/db"

        with patch.dict(sys.modules, {"psycopg": mock_psycopg_mod}):
            result = create_run_history(settings)

        assert isinstance(result, PostgresRunHistory)
        result.close()

    def test_redis_backend_fallback_on_import_error(self, tmp_path: Any) -> None:
        settings = MagicMock()
        settings.memory_backend = MemoryBackend.REDIS
        settings.redis_url = "redis://localhost:6379/0"
        settings.sqlite_path = str(tmp_path / "fallback.db")
        settings.postgres_url = None

        with patch.dict(sys.modules, {"redis": None}):
            result = create_run_history(settings)

        assert isinstance(result, ConversationMemory)
        result.close()

    def test_postgres_backend_fallback_on_import_error(self, tmp_path: Any) -> None:
        settings = MagicMock()
        settings.memory_backend = MemoryBackend.POSTGRES
        settings.redis_url = None
        settings.postgres_url = "postgresql://localhost:5432/db"
        settings.sqlite_path = str(tmp_path / "fallback.db")

        with patch.dict(sys.modules, {"psycopg": None}):
            result = create_run_history(settings)

        assert isinstance(result, ConversationMemory)
        result.close()
