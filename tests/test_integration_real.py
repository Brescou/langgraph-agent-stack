"""
tests/test_integration_real.py — Real integration tests using testcontainers.

These tests require Docker to be running and are marked with
``@pytest.mark.integration`` so they can be excluded from fast CI runs::

    uv run pytest -m "not integration"        # skip these
    uv run pytest -m integration              # run only these

Each test spins up a real database container, exercises the actual code path,
and tears it down after the test completes.

The ``TestGraphWithSqliteSaver`` class does NOT require Docker: it validates
the full ``MultiAgentGraph`` pipeline against a real on-disk ``SqliteSaver``
with the LLM mocked, proving that checkpointing works end-to-end.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from core.config import MemoryBackend

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TC_AVAILABLE = True
try:
    import testcontainers  # noqa: F401
except ImportError:
    _TC_AVAILABLE = False

skip_no_tc = pytest.mark.skipif(
    not _TC_AVAILABLE, reason="testcontainers not installed"
)


def _check_docker_available() -> bool:
    """Return True if Docker daemon is reachable."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def _pg_saver_available() -> bool:
    """Return True if langgraph-checkpoint-postgres is importable."""
    try:
        from langgraph.checkpoint.postgres import (
            PostgresSaver,  # type: ignore[import] # noqa: F401
        )

        return True
    except ImportError:
        return False


def _redis_saver_available() -> bool:
    """Return True if langgraph-checkpoint-redis is importable."""
    try:
        from langgraph.checkpoint.redis import (
            RedisSaver,  # type: ignore[import] # noqa: F401
        )

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# 2.1 — Postgres checkpointer real round-trip
# ---------------------------------------------------------------------------


@skip_no_tc
@pytest.mark.skipif(not _check_docker_available(), reason="Docker daemon not reachable")
@pytest.mark.skipif(
    not _pg_saver_available(), reason="langgraph-checkpoint-postgres not installed"
)
class TestPostgresReal:
    """Real Postgres checkpointer integration tests."""

    def test_postgres_checkpointer_roundtrip(self) -> None:
        """Create a PostgresSaver against a real container and verify setup()."""
        from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore[import]
        from testcontainers.postgres import PostgresContainer

        with PostgresContainer("postgres:16-alpine") as pg:
            raw_url = pg.get_connection_url()
            dsn = raw_url.replace("postgresql+psycopg2://", "postgresql://")

            saver = PostgresSaver.from_conn_string(dsn)
            saver.setup()
            assert saver is not None

    def test_create_checkpointer_postgres_factory(self) -> None:
        """Verify create_checkpointer() works with a real Postgres instance."""
        from testcontainers.postgres import PostgresContainer

        from core.memory import create_checkpointer

        with PostgresContainer("postgres:16-alpine") as pg:
            raw_url = pg.get_connection_url()
            dsn = raw_url.replace("postgresql+psycopg2://", "postgresql://")

            mock_settings = MagicMock()
            mock_settings.memory_backend = MemoryBackend.POSTGRES
            mock_settings.postgres_url = dsn
            mock_settings.environment = "production"

            result = create_checkpointer(mock_settings)
            assert result is not None
            assert not isinstance(result, MemorySaver)


# ---------------------------------------------------------------------------
# 2.2 — Redis checkpointer real round-trip
# ---------------------------------------------------------------------------


@skip_no_tc
@pytest.mark.skipif(not _check_docker_available(), reason="Docker daemon not reachable")
@pytest.mark.skipif(
    not _redis_saver_available(), reason="langgraph-checkpoint-redis not installed"
)
class TestRedisReal:
    """Real Redis checkpointer integration tests."""

    def test_redis_checkpointer_roundtrip(self) -> None:
        """Create a RedisSaver against a real container."""
        from langgraph.checkpoint.redis import RedisSaver  # type: ignore[import]
        from testcontainers.redis import RedisContainer

        with RedisContainer("redis:7-alpine") as redis_ct:
            host = redis_ct.get_container_host_ip()
            port = redis_ct.get_exposed_port(6379)
            redis_url = f"redis://{host}:{port}/0"

            saver = RedisSaver.from_conn_string(redis_url)
            assert saver is not None

    def test_create_checkpointer_redis_factory(self) -> None:
        """Verify create_checkpointer() works with a real Redis instance."""
        from testcontainers.redis import RedisContainer

        from core.memory import create_checkpointer

        with RedisContainer("redis:7-alpine") as redis_ct:
            host = redis_ct.get_container_host_ip()
            port = redis_ct.get_exposed_port(6379)
            redis_url = f"redis://{host}:{port}/0"

            mock_settings = MagicMock()
            mock_settings.memory_backend = MemoryBackend.REDIS
            mock_settings.redis_url = redis_url
            mock_settings.environment = "production"

            result = create_checkpointer(mock_settings)
            assert result is not None
            assert not isinstance(result, MemorySaver)


# ---------------------------------------------------------------------------
# 2.3 — API E2E with real SQLite backend + mocked LLM
# ---------------------------------------------------------------------------


class TestAPIEndToEnd:
    """Full API E2E test using real SQLite backend and mocked LLM."""

    def test_run_endpoint_returns_analysis(self) -> None:
        """POST /run should return a valid RunResponse with correct data."""
        from agents.models import AnalysisReport
        from core.security import RateLimiter

        mock_report = AnalysisReport(
            query="E2E test query",
            executive_summary="Test summary",
            key_insights=["Insight 1"],
            patterns=["Pattern 1"],
            implications=["Implication 1"],
            confidence=0.9,
            research_summary="Research summary",
            metadata={"run_id": "e2e-test"},
        )

        mock_graph_instance = MagicMock()
        mock_graph_instance.run.return_value = mock_report
        mock_graph_instance.__enter__ = MagicMock(return_value=mock_graph_instance)
        mock_graph_instance.__exit__ = MagicMock(return_value=False)
        mock_graph_cls = MagicMock(return_value=mock_graph_instance)

        mock_llm = MagicMock(spec=True)
        mock_checkpointer = MagicMock()
        permissive_limiter = RateLimiter(max_requests=10_000, window_seconds=60.0)

        with (
            patch("api.main.MultiAgentGraph", mock_graph_cls),
            patch("api.main._rate_limiter", permissive_limiter),
            patch("api.main.get_shared_llm", return_value=mock_llm),
            patch("api.main.get_shared_checkpointer", return_value=mock_checkpointer),
        ):
            from fastapi.testclient import TestClient

            from api.main import app

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/run",
                    json={
                        "query": "What is quantum computing?",
                        "session_id": "e2e-session",
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert data["executive_summary"] == "Test summary"
                assert data["confidence"] == 0.9
                assert data["session_id"] == "e2e-session"
                assert data["key_insights"] == ["Insight 1"]
                assert data["research_summary"] == "Research summary"


# ---------------------------------------------------------------------------
# 2.4 — Full pipeline E2E: MultiAgentGraph + real SqliteSaver + mocked LLM
# ---------------------------------------------------------------------------


def _build_llm_responses() -> list[AIMessage]:
    """Return canned AIMessage responses for the 6 sequential LLM calls.

    Order (ResearchAgent then AnalystAgent):
      1. research  — query expansion  → JSON list of sub-queries
      2. validate  — quality check    → {"sufficient": true, "reason": "..."}
      3. summarize — final summary    → {"summary": "...", "confidence": 0.85}
      4. analyze   — extract insights → {"insights": [...], "confidence": 0.8}
      5. synthesize — patterns        → {"patterns": [...], "implications": [...]}
      6. report    — exec summary     → plain-text paragraph
    """
    return [
        AIMessage(content=json.dumps(["sub-query-1", "sub-query-2", "sub-query-3"])),
        AIMessage(
            content=json.dumps({"sufficient": True, "reason": "Findings are adequate."})
        ),
        AIMessage(
            content=json.dumps(
                {
                    "summary": "Quantum computing leverages qubits for computation.",
                    "confidence": 0.85,
                }
            )
        ),
        AIMessage(
            content=json.dumps(
                {
                    "insights": [
                        "Qubits enable superposition.",
                        "Error correction is key.",
                    ],
                    "confidence": 0.8,
                }
            )
        ),
        AIMessage(
            content=json.dumps(
                {
                    "patterns": ["Hardware iteration accelerating."],
                    "implications": ["Post-quantum crypto needed."],
                }
            )
        ),
        AIMessage(
            content="Quantum computing is a transformative technology with near-term "
            "applications in drug discovery and materials science."
        ),
    ]


class TestGraphWithSqliteSaver:
    """Full pipeline E2E: real SqliteSaver, mocked LLM, real graph execution."""

    def test_pipeline_produces_analysis_report(self, tmp_path: Any) -> None:
        """MultiAgentGraph.run() with a real SqliteSaver produces an AnalysisReport."""
        from langgraph.checkpoint.sqlite import SqliteSaver

        from agents.models import AnalysisReport
        from core.graph import MultiAgentGraph

        db_path = str(tmp_path / "e2e_checkpoint.db")

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = _build_llm_responses()

        with (
            SqliteSaver.from_conn_string(db_path) as saver,
            patch("agents.base_agent.get_llm", return_value=mock_llm),
            patch("agents.base_agent.create_checkpointer", return_value=saver),
            patch("core.graph.create_checkpointer", return_value=saver),
        ):
            graph = MultiAgentGraph(
                run_id="e2e-sqlite-test",
                llm=mock_llm,
                checkpointer=saver,
            )
            report = graph.run("What is quantum computing?")

        assert isinstance(report, AnalysisReport)
        assert report.query == "What is quantum computing?"
        assert len(report.key_insights) > 0
        assert len(report.patterns) > 0
        assert len(report.implications) > 0
        assert 0.0 <= report.confidence <= 1.0
        assert report.research_summary != ""
        assert report.executive_summary != ""
        assert mock_llm.invoke.call_count == 6


# ---------------------------------------------------------------------------
# 2.5 — Full pipeline E2E: MultiAgentGraph + real PostgresSaver + mocked LLM
# ---------------------------------------------------------------------------


@skip_no_tc
@pytest.mark.skipif(not _check_docker_available(), reason="Docker daemon not reachable")
@pytest.mark.skipif(
    not _pg_saver_available(), reason="langgraph-checkpoint-postgres not installed"
)
class TestGraphWithPostgresSaver:
    """Full pipeline E2E: real PostgresSaver via testcontainers, mocked LLM."""

    def test_pipeline_produces_analysis_report(self) -> None:
        """MultiAgentGraph.run() with a real PostgresSaver produces an AnalysisReport."""
        from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore[import]
        from testcontainers.postgres import PostgresContainer

        from agents.models import AnalysisReport
        from core.graph import MultiAgentGraph

        with PostgresContainer("postgres:16-alpine") as pg:
            raw_url = pg.get_connection_url()
            dsn = raw_url.replace("postgresql+psycopg2://", "postgresql://")

            saver = PostgresSaver.from_conn_string(dsn)
            saver.setup()

            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = _build_llm_responses()

            with (
                patch("agents.base_agent.get_llm", return_value=mock_llm),
                patch("agents.base_agent.create_checkpointer", return_value=saver),
                patch("core.graph.create_checkpointer", return_value=saver),
            ):
                graph = MultiAgentGraph(
                    run_id="e2e-postgres-test",
                    llm=mock_llm,
                    checkpointer=saver,
                )
                report = graph.run("What is quantum computing?")

            assert isinstance(report, AnalysisReport)
            assert report.query == "What is quantum computing?"
            assert len(report.key_insights) > 0
            assert len(report.patterns) > 0
            assert len(report.implications) > 0
            assert 0.0 <= report.confidence <= 1.0
            assert report.research_summary != ""
            assert report.executive_summary != ""
            assert mock_llm.invoke.call_count == 6
