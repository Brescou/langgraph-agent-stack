"""
tests/test_auth_error_propagation.py — Invalid API keys must fail loudly.

Regression tests for the v0.6.2 fix: an LLM-provider authentication failure
(HTTP 401/403, e.g. a placeholder ``ANTHROPIC_API_KEY`` from ``.env.example``)
must surface as ``AgentAuthenticationError`` and an HTTP 502 with an
actionable message — never as an HTTP 200 with degraded placeholder content
("Summary unavailable.", "structured extraction failed").
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agents.analyst import AnalystAgent
from agents.base_agent import AgentAuthenticationError
from agents.llm_retry import is_auth_llm_error
from agents.models import ResearchResult
from agents.researcher import ResearchAgent
from tests.legacy_pack_override import override_legacy_pack_cls


class _FakeAuthError(Exception):
    """Mimics an SDK exception carrying an HTTP status code."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------------------


class TestIsAuthLLMError:
    def test_401_status_is_auth_error(self) -> None:
        assert is_auth_llm_error(_FakeAuthError("invalid x-api-key", 401)) is True

    def test_403_status_is_auth_error(self) -> None:
        assert is_auth_llm_error(_FakeAuthError("forbidden", 403)) is True

    def test_transient_statuses_are_not_auth_errors(self) -> None:
        for status in (408, 429, 500, 503, 529):
            assert is_auth_llm_error(_FakeAuthError("boom", status)) is False

    def test_plain_exception_is_not_auth_error(self) -> None:
        assert is_auth_llm_error(RuntimeError("boom")) is False

    def test_anthropic_authentication_error_type(self) -> None:
        anthropic = pytest.importorskip("anthropic")
        httpx = pytest.importorskip("httpx")

        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(401, request=request)
        exc = anthropic.AuthenticationError(
            "invalid x-api-key", response=response, body=None
        )
        assert is_auth_llm_error(exc) is True


# ---------------------------------------------------------------------------
# BaseAgent._invoke_llm_with_retry
# ---------------------------------------------------------------------------


class TestInvokeRaisesAuthError:
    def test_auth_error_is_not_retried_and_raises_typed_error(self) -> None:
        from langchain_core.messages import HumanMessage

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = _FakeAuthError("invalid x-api-key", 401)
        mock_llm.bind_tools.return_value = mock_llm

        with patch("agents.base_agent.get_llm", return_value=mock_llm):
            agent = ResearchAgent()
            with pytest.raises(AgentAuthenticationError) as exc_info:
                agent._invoke_llm_with_retry([HumanMessage(content="test")])

        # Not retried: a credential problem never recovers on its own.
        assert mock_llm.invoke.call_count == 1
        # The message must be actionable for a quickstart user.
        message = str(exc_info.value)
        assert "LLM_PROVIDER=mock" in message
        assert ".env" in message


# ---------------------------------------------------------------------------
# Agents must not degrade auth errors into placeholder output
# ---------------------------------------------------------------------------


class TestAgentsPropagateAuthError:
    def test_research_agent_propagates_auth_error(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = _FakeAuthError("invalid x-api-key", 401)
        mock_llm.bind_tools.return_value = mock_llm

        with patch("agents.base_agent.get_llm", return_value=mock_llm):
            agent = ResearchAgent()
            with pytest.raises(AgentAuthenticationError):
                agent.run_structured("What is quantum computing?")

    def test_analyst_agent_propagates_auth_error(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = _FakeAuthError("invalid x-api-key", 401)
        mock_llm.bind_tools.return_value = mock_llm

        research = ResearchResult(
            query="q",
            summary="s",
            findings=["f"],
            sources=["src"],
            confidence=0.9,
        )
        with patch("agents.base_agent.get_llm", return_value=mock_llm):
            agent = AnalystAgent()
            with pytest.raises(AgentAuthenticationError):
                agent.run_structured(research)


# ---------------------------------------------------------------------------
# API mapping — HTTP 502 with an actionable detail
# ---------------------------------------------------------------------------


def test_run_auth_error_returns_502(test_client: TestClient) -> None:
    """POST /run returns 502 (not a degraded 200) on provider auth failure."""
    detail = (
        "[ResearchAgent] LLM provider 'anthropic' rejected the request "
        "credentials: 401 invalid x-api-key. Check ANTHROPIC_API_KEY in your "
        ".env (see .env.example), or set LLM_PROVIDER=mock to run without an "
        "API key."
    )
    mock_graph_instance = MagicMock()
    mock_graph_instance.run.side_effect = AgentAuthenticationError(detail)
    mock_graph_instance.__enter__ = MagicMock(return_value=mock_graph_instance)
    mock_graph_instance.__exit__ = MagicMock(return_value=False)
    mock_graph_cls = MagicMock(return_value=mock_graph_instance)

    with override_legacy_pack_cls(mock_graph_cls):
        response = test_client.post(
            "/run", json={"query": "What is quantum computing?"}
        )

    assert response.status_code == 502
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_research_auth_error_returns_502(test_client: TestClient) -> None:
    """POST /research returns 502 on provider auth failure."""
    with patch("api.endpoints.pipeline.ResearchAgent") as mock_cls:
        inst = MagicMock()
        inst.run_structured.side_effect = AgentAuthenticationError(
            "credentials rejected — check ANTHROPIC_API_KEY"
        )
        mock_cls.return_value = inst
        response = test_client.post(
            "/research", json={"query": "Explain distributed systems."}
        )

    assert response.status_code == 502
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]
