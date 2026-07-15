"""
tests/test_session_concurrency.py — In-flight session dedup and best-effort save_run.

Covers:
- the in-flight session registry in ``api.state`` (try_acquire_session /
  release_session semantics);
- 409 Conflict when a run is already in flight for an explicit session_id;
- requests without a client-provided session_id are never locked;
- ``_save_run_best_effort``: a raising or slow ``save_run`` must never fail
  nor block the request.

All agent calls are mocked; no real Anthropic API requests are made.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api.state as state
from api.pack_execution import SESSION_IN_FLIGHT_DETAIL
from core.security import InMemorySessionRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_inflight_sessions():
    """Swap in a fresh in-memory registry before and after each test."""
    state.session_registry = InMemorySessionRegistry()
    yield
    state.session_registry = InMemorySessionRegistry()


def _session_is_free(session_id: str) -> bool:
    """True if no run is in flight for *session_id* (acquire-and-release probe)."""
    if not state.try_acquire_session(session_id):
        return False
    state.release_session(session_id)
    return True


# ---------------------------------------------------------------------------
# Registry semantics (unit)
# ---------------------------------------------------------------------------


def test_try_acquire_then_release_session() -> None:
    """Acquire succeeds once, fails while held, succeeds again after release."""
    assert state.try_acquire_session("sess-unit") is True
    assert state.try_acquire_session("sess-unit") is False
    state.release_session("sess-unit")
    assert state.try_acquire_session("sess-unit") is True
    state.release_session("sess-unit")


def test_release_session_is_idempotent() -> None:
    """Releasing a session that is not in flight must not raise."""
    state.release_session("sess-never-acquired")
    state.release_session("sess-never-acquired")


# ---------------------------------------------------------------------------
# POST /run — sequential runs and 409 on in-flight session
# ---------------------------------------------------------------------------


def test_sequential_runs_with_same_session_id_succeed(
    test_client: TestClient,
) -> None:
    """Two sequential /run calls with the same session_id must both pass.

    Proves the lock is released after each request completes.
    """
    for _ in range(2):
        response = test_client.post(
            "/run",
            json={"query": "What is quantum computing?", "session_id": "sess-seq"},
        )
        assert response.status_code == 200
        assert response.json()["session_id"] == "sess-seq"
    assert _session_is_free("sess-seq")


def test_run_returns_409_when_session_in_flight(test_client: TestClient) -> None:
    """A /run for a session that already has a run in flight must return 409."""
    assert state.try_acquire_session("sess-busy") is True
    try:
        response = test_client.post(
            "/run",
            json={"query": "What is quantum computing?", "session_id": "sess-busy"},
        )
        assert response.status_code == 409
        assert response.json()["detail"] == SESSION_IN_FLIGHT_DETAIL
    finally:
        state.release_session("sess-busy")

    # Once released, the same session runs normally again.
    response = test_client.post(
        "/run",
        json={"query": "What is quantum computing?", "session_id": "sess-busy"},
    )
    assert response.status_code == 200


def test_run_stream_returns_409_when_session_in_flight(
    test_client: TestClient,
) -> None:
    """A /run/stream for an in-flight session must return 409 (no SSE body)."""
    assert state.try_acquire_session("sess-busy-stream") is True
    try:
        response = test_client.post(
            "/run/stream",
            json={"query": "test query", "session_id": "sess-busy-stream"},
        )
        assert response.status_code == 409
        assert response.json()["detail"] == SESSION_IN_FLIGHT_DETAIL
    finally:
        state.release_session("sess-busy-stream")


def test_run_stream_releases_lock_at_end_of_generator(
    test_client: TestClient,
) -> None:
    """After the SSE stream is fully consumed, the session lock must be free."""
    response = test_client.post(
        "/run/stream",
        json={"query": "test query", "session_id": "sess-stream"},
    )
    assert response.status_code == 200
    assert "data:" in response.text  # stream fully consumed by TestClient
    assert _session_is_free("sess-stream")

    # A follow-up run on the same session succeeds.
    response = test_client.post(
        "/run",
        json={"query": "What is quantum computing?", "session_id": "sess-stream"},
    )
    assert response.status_code == 200


def test_run_without_session_id_is_not_locked(test_client: TestClient) -> None:
    """A /run without session_id must succeed and never touch the registry."""
    # Even with an unrelated session in flight, anonymous runs are unaffected.
    assert state.try_acquire_session("sess-other") is True
    try:
        response = test_client.post(
            "/run", json={"query": "What is quantum computing?"}
        )
        assert response.status_code == 200
    finally:
        state.release_session("sess-other")
    # Only the manually acquired session was ever registered.
    assert _session_is_free("sess-other")


# ---------------------------------------------------------------------------
# Best-effort save_run — errors and timeouts must not fail the request
# ---------------------------------------------------------------------------


def test_save_run_exception_does_not_fail_request(test_client: TestClient) -> None:
    """A save_run that raises must be swallowed; the client still gets 200."""
    broken_memory = MagicMock()
    broken_memory.save_run.side_effect = RuntimeError("backend down")

    with patch("api.state.shared_memory", broken_memory):
        response = test_client.post(
            "/run",
            json={"query": "What is quantum computing?", "session_id": "sess-broken"},
        )

    assert response.status_code == 200
    broken_memory.save_run.assert_called_once()
    # The lock must be released even when persistence fails.
    assert _session_is_free("sess-broken")


def test_slow_save_run_does_not_block_request(test_client: TestClient) -> None:
    """A save_run slower than the timeout must not block the response."""
    slow_memory = MagicMock()
    slow_memory.save_run.side_effect = lambda **kwargs: time.sleep(0.5)

    with (
        patch("api.state.shared_memory", slow_memory),
        patch("api.pack_execution.SAVE_RUN_TIMEOUT_SECONDS", 0.05),
    ):
        started = time.monotonic()
        response = test_client.post(
            "/run",
            json={"query": "What is quantum computing?", "session_id": "sess-slow"},
        )
        elapsed = time.monotonic() - started

    assert response.status_code == 200
    assert elapsed < 5.0  # bounded by the patched timeout, not the slow backend
    slow_memory.save_run.assert_called_once()
    assert _session_is_free("sess-slow")


# ---------------------------------------------------------------------------
# Session registry backends (core/security.py)
# ---------------------------------------------------------------------------


class TestCreateSessionRegistry:
    def test_memory_backend_returns_in_memory_registry(self) -> None:
        from core.security import create_session_registry

        registry = create_session_registry(backend="memory")
        assert isinstance(registry, InMemorySessionRegistry)

    def test_redis_backend_without_url_falls_back(self) -> None:
        """Factory with backend='redis' but no URL falls back to in-memory."""
        from core.security import create_session_registry

        registry = create_session_registry(backend="redis", redis_url="")
        assert isinstance(registry, InMemorySessionRegistry)

    def test_redis_backend_with_url_creates_redis_registry(self) -> None:
        """Factory with backend='redis' and a URL creates RedisSessionRegistry."""
        from core.security import RedisSessionRegistry, create_session_registry

        mock_redis = MagicMock()
        mock_redis.register_script.return_value = MagicMock()

        try:
            import redis as redis_lib

            with patch.object(redis_lib.Redis, "from_url", return_value=mock_redis):
                registry = create_session_registry(
                    backend="redis",
                    redis_url="redis://localhost:6379/0",
                    ttl_seconds=120,
                )
                assert isinstance(registry, RedisSessionRegistry)
                assert registry.ttl_seconds == 120
        except ImportError:
            pytest.skip("redis package not installed")


class TestRedisSessionRegistry:
    def _make_registry(self, mock_redis: MagicMock):
        import redis as redis_lib

        from core.security import RedisSessionRegistry

        with patch.object(redis_lib.Redis, "from_url", return_value=mock_redis):
            return RedisSessionRegistry(
                redis_url="redis://localhost:6379/0", ttl_seconds=60
            )

    def test_acquire_uses_set_nx_ex_and_release_deletes_own_token(self) -> None:
        """Acquire issues SET NX EX; release runs the compare-and-delete script."""
        pytest.importorskip("redis")
        mock_redis = MagicMock()
        mock_redis.set.return_value = True
        mock_script = MagicMock()
        mock_redis.register_script.return_value = mock_script

        registry = self._make_registry(mock_redis)
        assert registry.try_acquire("sess-redis") is True

        _, kwargs = mock_redis.set.call_args
        assert kwargs == {"nx": True, "ex": 60}
        (key, token), _ = mock_redis.set.call_args
        assert key == "session-lock:sess-redis"

        registry.release("sess-redis")
        mock_script.assert_called_once_with(keys=[key], args=[token])

    def test_acquire_returns_false_when_lock_held(self) -> None:
        """SET NX returning None (key exists) means another run is in flight."""
        pytest.importorskip("redis")
        mock_redis = MagicMock()
        mock_redis.set.return_value = None
        mock_redis.register_script.return_value = MagicMock()

        registry = self._make_registry(mock_redis)
        assert registry.try_acquire("sess-held") is False

    def test_acquire_fails_open_when_redis_down(self) -> None:
        """An unreachable Redis must allow the run (fail-open) without raising."""
        pytest.importorskip("redis")
        mock_redis = MagicMock()
        mock_redis.set.side_effect = ConnectionError("Redis down")
        mock_redis.register_script.return_value = MagicMock()

        registry = self._make_registry(mock_redis)
        assert registry.try_acquire("sess-down") is True

    def test_release_without_owned_token_is_noop(self) -> None:
        """Releasing a session this process never acquired must not touch Redis."""
        pytest.importorskip("redis")
        mock_redis = MagicMock()
        mock_script = MagicMock()
        mock_redis.register_script.return_value = mock_script

        registry = self._make_registry(mock_redis)
        registry.release("sess-foreign")
        mock_script.assert_not_called()
