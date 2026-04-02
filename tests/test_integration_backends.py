"""
tests/test_integration_backends.py — Integration tests for Postgres and Redis
checkpointer code paths, and a full pipeline E2E integration test.

These tests exercise the real ``create_checkpointer`` factory and the
``ConversationMemory`` round-trip against mocked backend modules (so no
real Postgres/Redis instances are required), ensuring the code paths are
actually exercised rather than just the happy-path unit tests.

The pipeline E2E test verifies the full Research → Analysis flow with a
mocked LLM, ensuring all orchestration wiring holds together.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from agents.models import AnalysisReport
from core.memory import ConversationMemory, create_checkpointer

# ---------------------------------------------------------------------------
# Postgres checkpointer integration
# ---------------------------------------------------------------------------


class TestPostgresCheckpointerIntegration:
    """Exercise the full Postgres checkpointer creation path."""

    def test_postgres_checkpointer_lifecycle(self) -> None:
        """create_checkpointer(postgres) → PostgresSaver with setup() called."""
        mock_saver = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_saver)
        mock_pg_module = MagicMock()
        mock_pg_module.PostgresSaver.from_conn_string.return_value = mock_ctx

        mock_settings = MagicMock()
        mock_settings.memory_backend = "postgres"
        mock_settings.postgres_url = "postgresql+psycopg://user:pass@db:5432/app"
        mock_settings.environment = "production"

        with patch.dict(
            "sys.modules",
            {"langgraph.checkpoint.postgres": mock_pg_module},
        ):
            result = create_checkpointer(mock_settings)

        assert result is mock_saver
        mock_pg_module.PostgresSaver.from_conn_string.assert_called_once_with(
            "postgresql+psycopg://user:pass@db:5432/app"
        )
        mock_saver.setup.assert_called_once()

    def test_postgres_missing_url_falls_back_in_dev(self) -> None:
        """Missing POSTGRES_URL in dev environment falls back to MemorySaver."""
        mock_settings = MagicMock()
        mock_settings.memory_backend = "postgres"
        mock_settings.postgres_url = None
        mock_settings.environment = "development"

        with patch("core.config.get_settings", return_value=mock_settings):
            result = create_checkpointer(mock_settings)

        assert isinstance(result, MemorySaver)

    def test_postgres_missing_url_raises_in_production(self) -> None:
        """Missing POSTGRES_URL in production raises RuntimeError."""
        mock_settings = MagicMock()
        mock_settings.memory_backend = "postgres"
        mock_settings.postgres_url = None
        mock_settings.environment = "production"

        with patch("core.config.get_settings", return_value=mock_settings):
            with pytest.raises(RuntimeError, match="POSTGRES_URL"):
                create_checkpointer(mock_settings)

    def test_postgres_import_failure_falls_back_in_dev(self) -> None:
        """Import error for langgraph-checkpoint-postgres falls back in dev."""
        mock_settings = MagicMock()
        mock_settings.memory_backend = "postgres"
        mock_settings.postgres_url = "postgresql+psycopg://u:p@localhost:5432/db"
        mock_settings.environment = "development"

        with (
            patch.dict("sys.modules", {"langgraph.checkpoint.postgres": None}),
            patch("core.config.get_settings", return_value=mock_settings),
        ):
            result = create_checkpointer(mock_settings)

        assert isinstance(result, MemorySaver)


# ---------------------------------------------------------------------------
# Redis checkpointer integration
# ---------------------------------------------------------------------------


class TestRedisCheckpointerIntegration:
    """Exercise the full Redis checkpointer creation path."""

    def test_redis_checkpointer_lifecycle(self) -> None:
        """create_checkpointer(redis) → RedisSaver with correct URL."""
        mock_saver = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_saver)
        mock_redis_module = MagicMock()
        mock_redis_module.RedisSaver.from_conn_string.return_value = mock_ctx

        mock_settings = MagicMock()
        mock_settings.memory_backend = "redis"
        mock_settings.redis_url = "redis://:secret@redis-host:6379/0"
        mock_settings.environment = "production"

        with patch.dict(
            "sys.modules",
            {"langgraph.checkpoint.redis": mock_redis_module},
        ):
            result = create_checkpointer(mock_settings)

        assert result is mock_saver
        mock_redis_module.RedisSaver.from_conn_string.assert_called_once_with(
            "redis://:secret@redis-host:6379/0"
        )

    def test_redis_import_failure_falls_back_in_dev(self) -> None:
        """Import error for langgraph-checkpoint-redis falls back in dev."""
        mock_settings = MagicMock()
        mock_settings.memory_backend = "redis"
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_settings.environment = "development"

        with (
            patch.dict("sys.modules", {"langgraph.checkpoint.redis": None}),
            patch("core.config.get_settings", return_value=mock_settings),
        ):
            result = create_checkpointer(mock_settings)

        assert isinstance(result, MemorySaver)


# ---------------------------------------------------------------------------
# ConversationMemory round-trip integration
# ---------------------------------------------------------------------------


class TestConversationMemoryIntegration:
    """Full round-trip tests for ConversationMemory with session filtering."""

    def test_full_round_trip_with_sessions(self) -> None:
        """Save runs across sessions, verify session isolation and ordering."""
        with ConversationMemory(":memory:") as mem:
            mem.save_run(
                "run-a1",
                "What is ML?",
                {"summary": "ML overview"},
                {"session_id": "sess-alpha"},
            )
            mem.save_run(
                "run-b1",
                "What is DL?",
                {"summary": "DL overview"},
                {"session_id": "sess-beta"},
            )
            mem.save_run(
                "run-a2",
                "What is NLP?",
                {"summary": "NLP overview"},
                {"session_id": "sess-alpha"},
            )

            alpha_runs = mem.list_runs_by_session("sess-alpha")
            assert len(alpha_runs) == 2
            assert all(r["metadata"]["session_id"] == "sess-alpha" for r in alpha_runs)
            assert alpha_runs[0]["run_id"] == "run-a2"

            beta_runs = mem.list_runs_by_session("sess-beta")
            assert len(beta_runs) == 1
            assert beta_runs[0]["run_id"] == "run-b1"

            all_runs = mem.list_runs(limit=50)
            assert len(all_runs) == 3

    def test_overwrite_preserves_session(self) -> None:
        """Overwriting a run preserves metadata including session_id."""
        with ConversationMemory(":memory:") as mem:
            mem.save_run(
                "run-x",
                "original query",
                {"v": 1},
                {"session_id": "sess-1"},
            )
            mem.save_run(
                "run-x",
                "updated query",
                {"v": 2},
                {"session_id": "sess-1"},
            )

            record = mem.get_run("run-x")
            assert record is not None
            assert record["query"] == "updated query"
            assert record["result"]["v"] == 2
            assert record["metadata"]["session_id"] == "sess-1"

            runs = mem.list_runs_by_session("sess-1")
            assert len(runs) == 1


# ---------------------------------------------------------------------------
# Pipeline E2E integration (mocked LLM, real agents + graph)
# ---------------------------------------------------------------------------


class TestPipelineE2E:
    """Full Research → Analysis pipeline with real agents wired through
    ``MultiAgentGraph``, using a mocked LLM."""

    @pytest.fixture()
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM that returns plausible JSON for both agents."""
        llm = MagicMock(spec=BaseChatModel)
        llm.bind_tools.return_value = llm

        research_response = json.dumps({"sub_queries": ["What is quantum computing?"]})
        analysis_response = json.dumps(
            {
                "insights": [
                    {"title": "Quantum Advantage", "description": "Speed gains"}
                ],
                "confidence": 0.88,
            }
        )

        llm.invoke.side_effect = [
            AIMessage(content=research_response),
            AIMessage(content="Quantum computing uses qubits for computation."),
            AIMessage(content="Quantum computing is a new paradigm."),
            AIMessage(content=analysis_response),
            AIMessage(content=analysis_response),
        ]
        return llm

    def test_pipeline_produces_analysis_report(self, mock_llm: MagicMock) -> None:
        """The full pipeline returns a valid AnalysisReport."""
        from core.graph import MultiAgentGraph

        with MultiAgentGraph(
            run_id="e2e-test-run",
            llm=mock_llm,
            checkpointer=MemorySaver(),
        ) as graph:
            report = graph.run("What is quantum computing?")

        assert isinstance(report, AnalysisReport)
        assert report.confidence > 0
        assert len(report.executive_summary) > 0
