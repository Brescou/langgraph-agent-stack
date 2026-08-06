"""
tests/test_examples.py — Execute every shipped example graph.

``examples/`` is documentation people copy from, but nothing exercised it:
``examples/parallel`` shipped broken for months because ``fan_out_node``
returned ``list[Send]`` from a node body instead of a routing function, which
LangGraph rejects with ``InvalidUpdateError``. Reading the file does not
catch that — only compiling and invoking the graph does.

These tests build each example graph with the deterministic mock chat model
and invoke it, so a wiring regression fails CI instead of a user's first run.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.mock_llm import MockProviderChatModel


@pytest.fixture()
def mock_llm() -> MockProviderChatModel:
    """Deterministic chat model — no network, no API key."""
    return MockProviderChatModel()


def test_sequential_example_runs(mock_llm: MockProviderChatModel) -> None:
    """The sequential example compiles and produces both node outputs."""
    from examples.sequential.graph import build_sequential_graph

    graph = build_sequential_graph(mock_llm)
    result: dict[str, Any] = graph.invoke(
        {"query": "What is quantum computing?", "messages": []}
    )

    assert result["research_output"]
    assert result["analysis_output"]


def test_parallel_example_fans_out_to_every_branch(
    mock_llm: MockProviderChatModel,
) -> None:
    """The Send fan-out runs all three analyst branches and reduces them.

    Regression guard: this raised ``InvalidUpdateError`` while ``fan_out_node``
    was wired with ``add_edge`` instead of ``add_conditional_edges``.
    """
    from examples.parallel.graph import build_parallel_graph

    graph = build_parallel_graph(mock_llm)
    result: dict[str, Any] = graph.invoke(
        {
            "query": "Trade-offs of microservices?",
            "analyses": [],
            "final_report": "",
        }
    )

    assert len(result["analyses"]) == 3, "each Send branch must contribute one analysis"
    roles = {"Technology Analyst", "Market Analyst", "Risk Analyst"}
    assert all(any(role in a for a in result["analyses"]) for role in roles)
    assert result["final_report"]


def test_supervisor_example_runs(mock_llm: MockProviderChatModel) -> None:
    """The supervisor example routes and terminates without looping forever."""
    from examples.supervisor.graph import build_supervisor_graph

    graph = build_supervisor_graph(mock_llm)
    result: dict[str, Any] = graph.invoke(
        {
            "query": "Explain binary search.",
            "next_agent": "",
            "agent_output": "",
            "messages": [],
        }
    )

    assert "next_agent" in result


def test_human_in_loop_example_interrupts_then_resumes(
    mock_llm: MockProviderChatModel,
) -> None:
    """The HITL example pauses on ``interrupt()`` and resumes via ``Command``."""
    from langgraph.types import Command

    from examples.human_in_loop.graph import build_human_loop_graph

    graph, _checkpointer = build_human_loop_graph(mock_llm)
    config = {"configurable": {"thread_id": "test-hitl"}}

    outcome = graph.invoke({"query": "Delete stale rows."}, config=config)
    assert "__interrupt__" in outcome, "graph must pause for human approval"

    resumed: dict[str, Any] = graph.invoke(
        Command(resume={"approved": True}), config=config
    )
    assert resumed["result"]
