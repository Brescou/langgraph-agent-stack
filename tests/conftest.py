"""
tests/conftest.py — Shared pytest fixtures for the langgraph-agent-stack test suite.

All fixtures use mocks so that no real Anthropic API calls are made during tests.
The FastAPI TestClient is wired to a patched application that replaces
MultiAgentGraph and ResearchAgent with MagicMock instances.
"""

from __future__ import annotations

import os
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Environment must be patched BEFORE the settings singleton is imported.
# We set ANTHROPIC_API_KEY to a value that passes the format validator.
# ---------------------------------------------------------------------------
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test123456789012345")
os.environ.setdefault("MEMORY_BACKEND", "sqlite")
os.environ.setdefault("SQLITE_PATH", ":memory:")
os.environ.setdefault("ENVIRONMENT", "development")


# ---------------------------------------------------------------------------
# Domain-object fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mock_research_result() -> MagicMock:
    """
    Return a MagicMock that behaves like a ResearchResult dataclass.

    Scope: session — the object is immutable across all tests.
    """
    result = MagicMock()
    result.query = "What is quantum computing?"
    result.summary = "Quantum computing uses qubits to perform computations."
    result.findings = [
        "Quantum computers leverage superposition and entanglement.",
        "Current hardware is still in the NISQ era.",
    ]
    result.sources = [
        "https://example.com/quantum",
        "https://news.example.com/quantum-computing",
    ]
    result.confidence = 0.85
    result.metadata = {"agent": "ResearchAgent", "thread_id": "test-thread-001"}
    return result


@pytest.fixture(scope="session")
def mock_analysis_report() -> MagicMock:
    """
    Return a MagicMock that behaves like an AnalysisReport dataclass.

    Scope: session — the object is immutable across all tests.
    """
    report = MagicMock()
    report.query = "What is quantum computing?"
    report.executive_summary = (
        "Quantum computing represents a paradigm shift in computational power, "
        "leveraging quantum mechanics to solve problems intractable for classical computers."
    )
    report.key_insights = [
        "Qubits enable superposition, exponentially expanding the solution space.",
        "Error correction remains the dominant engineering challenge.",
    ]
    report.patterns = [
        "Rapid hardware iteration across multiple qubit modalities.",
        "Growing investment from both public and private sectors.",
    ]
    report.implications = [
        "Cryptographic systems based on integer factorisation will need replacement.",
        "Drug discovery and material science stand to benefit most in the near term.",
    ]
    report.confidence = 0.82
    report.research_summary = "Quantum computing uses qubits to perform computations."
    report.metadata = {"run_id": "test-run-001", "agent": "AnalystAgent"}
    return report


# ---------------------------------------------------------------------------
# TestClient fixture — agents fully mocked, no Anthropic calls
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def test_client(
    mock_research_result: MagicMock,
    mock_analysis_report: MagicMock,
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

    mock_agent_instance = MagicMock()
    mock_agent_instance.run_structured.return_value = mock_research_result

    mock_graph_cls = MagicMock(return_value=mock_graph_instance)
    mock_agent_cls = MagicMock(return_value=mock_agent_instance)

    with (
        patch("api.main.MultiAgentGraph", mock_graph_cls),
        patch("api.main.ResearchAgent", mock_agent_cls),
        patch("api.main._rate_limiter", permissive_limiter),
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
        anthropic_api_key="sk-ant-test123456789012345",
        memory_backend="sqlite",
        sqlite_path=":memory:",
        environment="development",
    )
