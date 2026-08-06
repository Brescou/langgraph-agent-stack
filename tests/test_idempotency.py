"""
tests/test_idempotency.py — Idempotency behavior for pack runs.

Covers:
- completed replay returns identical result without second execution;
- same key with different body returns 409;
- concurrent identical requests execute once;
- failed execution releases key;
- requests without Idempotency-Key are unchanged.

All LLM execution is mocked.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import api.state as state
from api.pack_execution import execute_typed_pack_run
from core.security import (
    IdempotencyConflictError,
    IdempotencyStatus,
    InMemoryIdempotencyStore,
    RedisIdempotencyStore,
    create_idempotency_store,
)

# ---------------------------------------------------------------------------
# Fake pack input
# ---------------------------------------------------------------------------


@dataclass
class FakeBody:
    query: str

    def model_dump_json(self) -> str:
        return f'{{"query":"{self.query}"}}'


@dataclass
class FakeSessionBody:
    query: str
    session_id: str

    def model_dump_json(self) -> str:
        return f'{{"query":"{self.query}","session_id":"{self.session_id}"}}'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_global_state():
    state.shutting_down.clear()
    state.idempotency_store = InMemoryIdempotencyStore()

    if hasattr(state, "shared_memory") and state.shared_memory:
        try:
            state.shared_memory._conn.execute("SELECT 1")
        except (sqlite3.ProgrammingError, AttributeError):
            try:
                state.shared_memory._conn = sqlite3.connect(
                    state.shared_memory.db_path, check_same_thread=False
                )
            except Exception:
                pass

    yield

    state.shutting_down.clear()
    state.idempotency_store = InMemoryIdempotencyStore()


@pytest.fixture
def fake_pack_registry():
    class FakeResult:
        def to_dict(self):
            return {"answer": "hello"}

        def model_dump(self):
            return {"answer": "hello"}

    class FakePipeline:
        cost_usd = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run_from_input(self, body):
            return FakeResult()

        def run(self, query):
            return FakeResult()

    class FakePack:
        output_schema = MagicMock()

        def __new__(cls, *args, **kwargs):
            return FakePipeline()

    fake_version = MagicMock()
    fake_version.version = "1.0"
    fake_version.pack_cls = FakePack

    with (
        patch(
            "api.pack_execution.PackRegistry.get",
            return_value=FakePack,
        ),
        patch(
            "api.pack_execution.PackRegistry._get_versions",
            return_value=[fake_version],
        ),
    ):
        yield FakePack


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeResult:
    def to_dict(self):
        return {"answer": "hello"}

    def model_dump(self):
        return {"answer": "hello"}


async def fake_executor(fn, *args):
    return fn(*args)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completed_replay_returns_same_result_without_second_execution(
    fake_pack_registry,
):
    calls = 0

    def fake_invoke(pack_cls, pipeline, body):
        nonlocal calls
        calls += 1
        return FakeResult()

    with (
        patch(
            "api.pack_execution.run_in_executor",
            side_effect=fake_executor,
        ),
        patch(
            "api.pack_execution.invoke_pack_run",
            side_effect=fake_invoke,
        ),
    ):
        first = await execute_typed_pack_run(
            "test-pack",
            FakeBody("hello"),
            idempotency_key="abc",
        )

        second = await execute_typed_pack_run(
            "test-pack",
            FakeBody("hello"),
            idempotency_key="abc",
        )

    assert first == second
    assert calls == 1


@pytest.mark.asyncio
async def test_same_key_different_body_returns_409(
    fake_pack_registry,
):
    def fake_invoke(pack_cls, pipeline, body):
        return FakeResult()

    with (
        patch(
            "api.pack_execution.run_in_executor",
            side_effect=fake_executor,
        ),
        patch(
            "api.pack_execution.invoke_pack_run",
            side_effect=fake_invoke,
        ),
    ):
        await execute_typed_pack_run(
            "test-pack",
            FakeBody("hello"),
            idempotency_key="abc",
        )

        with pytest.raises(HTTPException) as exc:
            await execute_typed_pack_run(
                "test-pack",
                FakeBody("different"),
                idempotency_key="abc",
            )

    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_concurrent_same_key_executes_once(
    fake_pack_registry,
):
    calls = 0

    def fake_invoke(pack_cls, pipeline, body):
        nonlocal calls

        calls += 1

        import time

        time.sleep(0.05)

        return FakeResult()

    with (
        patch(
            "api.pack_execution.run_in_executor",
            side_effect=fake_executor,
        ),
        patch(
            "api.pack_execution.invoke_pack_run",
            side_effect=fake_invoke,
        ),
    ):
        results = await asyncio.gather(
            execute_typed_pack_run(
                "test-pack",
                FakeBody("hello"),
                idempotency_key="abc",
            ),
            execute_typed_pack_run(
                "test-pack",
                FakeBody("hello"),
                idempotency_key="abc",
            ),
        )

    assert calls == 1
    assert results[0] == results[1]


@pytest.mark.asyncio
async def test_failed_first_run_releases_key_and_retry_executes(
    fake_pack_registry,
):
    calls = 0

    def fake_invoke(pack_cls, pipeline, body):
        nonlocal calls

        calls += 1

        if calls == 1:
            raise RuntimeError("boom")

        return FakeResult()

    with (
        patch(
            "api.pack_execution.run_in_executor",
            side_effect=fake_executor,
        ),
        patch(
            "api.pack_execution.invoke_pack_run",
            side_effect=fake_invoke,
        ),
    ):
        with pytest.raises(RuntimeError):
            await execute_typed_pack_run(
                "test-pack",
                FakeBody("hello"),
                idempotency_key="abc",
            )

        result = await execute_typed_pack_run(
            "test-pack",
            FakeBody("hello"),
            idempotency_key="abc",
        )

    assert result is not None
    assert calls == 2


@pytest.mark.asyncio
async def test_without_idempotency_key_does_not_store(
    fake_pack_registry,
):
    calls = 0

    def fake_invoke(pack_cls, pipeline, body):
        nonlocal calls

        calls += 1
        return FakeResult()

    with (
        patch(
            "api.pack_execution.run_in_executor",
            side_effect=fake_executor,
        ),
        patch(
            "api.pack_execution.invoke_pack_run",
            side_effect=fake_invoke,
        ),
    ):
        first = await execute_typed_pack_run(
            "test-pack",
            FakeBody("hello"),
        )

        second = await execute_typed_pack_run(
            "test-pack",
            FakeBody("hello"),
        )

    assert first != second
    assert calls == 2


def test_in_memory_reservation_can_be_reused_after_ttl() -> None:
    store = InMemoryIdempotencyStore(ttl_seconds=1)

    with patch("core.security.monotonic", side_effect=[100.0, 100.5, 101.1]):
        assert store.reserve("expired", "body-hash") is True
        store.store_result("expired", {"answer": "first"})
        assert store.reserve("expired", "body-hash") is False
        assert store.reserve("expired", "body-hash") is True


class FakeRedis:
    """Small Redis command/script double for cross-store idempotency tests."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, str]] = {}
        self.expiry: dict[str, float] = {}

    def register_script(self, source: str):
        if "expires_at" in source:
            return self._reserve
        return self._store_result

    def _purge(self, key: str, now: float | None = None) -> None:
        if key in self.expiry and self.expiry[key] <= (
            time.time() if now is None else now
        ):
            self.records.pop(key, None)
            self.expiry.pop(key, None)

    def _reserve(self, *, keys: list[str], args: list[object]) -> int:
        key = keys[0]
        body_hash = str(args[0])
        ttl = int(args[1])
        now = float(args[2])
        self._purge(key, now)
        record = self.records.get(key)
        if record is None:
            self.records[key] = {
                "body_hash": body_hash,
                "status": "running",
                "expires_at": str(now + ttl),
            }
            self.expiry[key] = now + ttl
            return 1
        if record["body_hash"] != body_hash:
            return -1
        return 0

    def _store_result(self, *, keys: list[str], args: list[object]) -> int:
        key = keys[0]
        self._purge(key)
        if key not in self.records:
            return 0
        self.records[key]["status"] = "completed"
        self.records[key]["response"] = str(args[0])
        return 1

    def hgetall(self, key: str) -> dict[str, str]:
        self._purge(key)
        return dict(self.records.get(key, {}))

    def delete(self, key: str) -> None:
        self.records.pop(key, None)
        self.expiry.pop(key, None)


def test_redis_store_shares_reservations_and_replay(monkeypatch) -> None:
    import redis

    fake_redis = FakeRedis()
    monkeypatch.setattr(redis.Redis, "from_url", lambda *args, **kwargs: fake_redis)

    first = create_idempotency_store(
        backend="redis",
        redis_url="redis://localhost:6379/0",
        ttl_seconds=60,
    )
    second = create_idempotency_store(
        backend="redis",
        redis_url="redis://localhost:6379/0",
        ttl_seconds=60,
    )

    assert isinstance(first, RedisIdempotencyStore)
    assert isinstance(second, RedisIdempotencyStore)
    assert first.reserve("shared", "body-hash") is True
    assert second.reserve("shared", "body-hash") is False

    from api.pack_execution import PackRunResult

    first.store_result(
        "shared",
        PackRunResult(serialized={"answer": "ok"}, used_version="1", run_id="run-1"),
    )
    record = second.get("shared")

    assert record is not None
    assert record.status is IdempotencyStatus.COMPLETED
    assert record.response == PackRunResult(
        serialized={"answer": "ok"}, used_version="1", run_id="run-1"
    )

    with pytest.raises(IdempotencyConflictError):
        second.reserve("shared", "different-body")


@pytest.mark.asyncio
async def test_403_compliance_gate_releases_reservation(
    fake_pack_registry,
):
    store = state.get_shared_idempotency_store()

    with patch(
        "domain_packs.common.compliance.assert_regulated_pack_runtime_enabled",
        side_effect=ValueError("disabled"),
    ):
        with pytest.raises(HTTPException) as exc:
            await execute_typed_pack_run(
                "test-pack",
                FakeBody("hello"),
                idempotency_key="abc",
            )

    assert exc.value.status_code == 403
    assert store.get("abc") is None


@pytest.mark.asyncio
async def test_409_session_lock_releases_reservation(
    fake_pack_registry,
):
    store = state.get_shared_idempotency_store()

    with (
        patch(
            "api.pack_execution.state.try_acquire_session",
            return_value=None,
        ),
        patch(
            "api.pack_execution.state.try_acquire_session",
            return_value=False,
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            await execute_typed_pack_run(
                "test-pack",
                FakeSessionBody(
                    "hello",
                    session_id="session-1",
                ),
                idempotency_key="abc",
            )

    assert exc.value.status_code == 409
    assert store.get("abc") is None
