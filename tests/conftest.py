"""
tests/conftest.py — Shared pytest fixtures for the langgraph-agent-stack test suite.

All fixtures use mocks so that no real LLM API calls are made during tests.
The FastAPI TestClient is wired to a patched application that replaces
MultiAgentGraph and ResearchAgent with MagicMock instances.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agents.models import AnalysisReport, ResearchResult

# ---------------------------------------------------------------------------
# Environment must be patched BEFORE the settings singleton is imported.
# LLM_PROVIDER=anthropic with a dummy key so Settings loads without error.
# ---------------------------------------------------------------------------
os.environ.setdefault("LLM_PROVIDER", "anthropic")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test123456789012345")
os.environ.setdefault("MEMORY_BACKEND", "sqlite")
os.environ.setdefault("SQLITE_PATH", ":memory:")
os.environ.setdefault("ENVIRONMENT", "development")


# ---------------------------------------------------------------------------
# Domain-object fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_research_result() -> ResearchResult:
    """Return a real ``ResearchResult`` instance for testing.

    Scope: function — each test gets a fresh copy to avoid shared state.
    """
    return ResearchResult(
        query="What is quantum computing?",
        summary="Quantum computing uses qubits to perform computations.",
        findings=[
            "Quantum computers leverage superposition and entanglement.",
            "Current hardware is still in the NISQ era.",
        ],
        sources=[
            "https://example.com/quantum",
            "https://news.example.com/quantum-computing",
        ],
        confidence=0.85,
        metadata={"agent": "ResearchAgent", "thread_id": "test-thread-001"},
    )


@pytest.fixture()
def mock_analysis_report() -> AnalysisReport:
    """Return a real ``AnalysisReport`` instance for testing.

    Scope: function — each test gets a fresh copy to avoid shared state.
    """
    return AnalysisReport(
        query="What is quantum computing?",
        executive_summary=(
            "Quantum computing represents a paradigm shift in computational power, "
            "leveraging quantum mechanics to solve problems intractable for classical "
            "computers."
        ),
        key_insights=[
            "Qubits enable superposition, exponentially expanding the solution space.",
            "Error correction remains the dominant engineering challenge.",
        ],
        patterns=[
            "Rapid hardware iteration across multiple qubit modalities.",
            "Growing investment from both public and private sectors.",
        ],
        implications=[
            "Cryptographic systems based on integer factorisation will need replacement.",
            "Drug discovery and material science stand to benefit most in the near term.",
        ],
        confidence=0.82,
        research_summary="Quantum computing uses qubits to perform computations.",
        metadata={"run_id": "test-run-001", "agent": "AnalystAgent"},
    )


# ---------------------------------------------------------------------------
# TestClient fixture — agents fully mocked, no Anthropic calls
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def test_client(
    mock_research_result: ResearchResult,
    mock_analysis_report: AnalysisReport,
) -> Generator[TestClient, None, None]:
    """
    Return a FastAPI TestClient with MultiAgentGraph and ResearchAgent mocked.

    A fresh mock is constructed for every test function so that call counts
    and side effects are isolated between tests.

    Patches applied:
    - ``api.main.MultiAgentGraph`` — ``run()`` returns ``mock_analysis_report``.
    - ``api.main.ResearchAgent``   — ``run_structured()`` returns ``mock_research_result``.
    - ``api.main._rate_limiter``   — replaced with a permissive limiter (max 10 000
      requests) so individual endpoint tests are never blocked.
    """
    from core.security import RateLimiter

    permissive_limiter = RateLimiter(max_requests=10_000, window_seconds=60.0)

    mock_graph_instance = MagicMock()
    mock_graph_instance.run.return_value = mock_analysis_report

    mock_researcher_instance = MagicMock()
    mock_researcher_instance.run_structured.return_value = mock_research_result

    mock_analyst_instance = MagicMock()
    mock_analyst_instance.run_structured.return_value = mock_analysis_report

    mock_graph_cls = MagicMock(return_value=mock_graph_instance)
    mock_researcher_cls = MagicMock(return_value=mock_researcher_instance)
    mock_analyst_cls = MagicMock(return_value=mock_analyst_instance)

    mock_llm = MagicMock(spec=True)
    mock_checkpointer = MagicMock()

    with (
        patch("api.main.MultiAgentGraph", mock_graph_cls),
        patch("api.main.ResearchAgent", mock_researcher_cls),
        patch("api.main.AnalystAgent", mock_analyst_cls),
        patch("api.main._rate_limiter", permissive_limiter),
        patch("api.main.get_shared_llm", return_value=mock_llm),
        patch("api.main.get_shared_checkpointer", return_value=mock_checkpointer),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


# ---------------------------------------------------------------------------
# Settings fixture (unit tests that need a Settings instance)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_settings():
    """
    Return a Settings instance configured for testing.

    Uses an in-memory SQLite database so no files are written to disk.
    """
    from core.config import Settings

    return Settings(
        llm_provider="anthropic",
        anthropic_api_key="sk-ant-test123456789012345",
        memory_backend="sqlite",
        sqlite_path=":memory:",
        environment="development",
    )
