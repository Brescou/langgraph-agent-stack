"""
Parallel pattern: multiple agents run simultaneously via the Send API.

When to use: independent analyses of the same topic where order does not
matter. Maximises throughput by running branches concurrently rather than
sequentially. Ideal for multi-perspective reports, ensemble scoring, or
parallel data enrichment.

Architecture::

    START
      |
    fan_out_node  (issues one Send per analyst role)
     /     |      \\
  tech  market   risk    (all run concurrently)
     \\     |      /
    consolidate_node
      |
     END

Run: uv run python examples/parallel/graph.py
"""

from __future__ import annotations

import operator
from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# State schemas
# ---------------------------------------------------------------------------


class ParallelState(TypedDict):
    """
    Top-level state for the parallel pipeline.

    Attributes:
        query: The topic all three analysts will investigate.
        analyses: List of analysis strings accumulated from all branches.
            The ``operator.add`` reducer appends each branch result.
        final_report: The consolidated report produced at the end.
    """

    query: str
    analyses: Annotated[list[str], operator.add]
    final_report: str


class AnalystBranchState(TypedDict):
    """
    Per-branch state passed to each parallel analyst node via ``Send``.

    Attributes:
        query: The original topic (copied from ``ParallelState``).
        role: The analyst persona (e.g. ``"Technology Analyst"``).
        focus: A short description of this branch's analytical lens.
    """

    query: str
    role: str
    focus: str


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def fan_out_node(state: ParallelState) -> list[Send]:
    """
    Node: emit one ``Send`` command per analyst role to trigger parallel execution.

    LangGraph will schedule all three analyst nodes concurrently.  The
    results are merged back into ``ParallelState.analyses`` by the
    ``operator.add`` reducer.

    Args:
        state: Current pipeline state containing the query.

    Returns:
        A list of ``Send`` objects, one per analyst branch.
    """
    branches = [
        {
            "query": state["query"],
            "role": "Technology Analyst",
            "focus": (
                "technical feasibility, architectural patterns, tooling ecosystem, "
                "and engineering trade-offs"
            ),
        },
        {
            "query": state["query"],
            "role": "Market Analyst",
            "focus": (
                "market adoption, competitive landscape, business drivers, "
                "and growth trends"
            ),
        },
        {
            "query": state["query"],
            "role": "Risk Analyst",
            "focus": (
                "operational risks, security concerns, compliance requirements, "
                "and mitigation strategies"
            ),
        },
    ]

    return [Send("analyst_node", branch) for branch in branches]


def analyst_node(state: AnalystBranchState, llm: BaseChatModel) -> dict[str, list[str]]:
    """
    Node: run a single analyst persona on the query.

    Called three times in parallel (once per Send).  Each invocation
    returns a partial update that LangGraph merges into ``ParallelState``
    via the ``operator.add`` reducer on ``analyses``.

    Args:
        state: Branch-specific state containing role, focus, and query.
        llm: Configured LangChain chat model.

    Returns:
        A partial state update with a single-element ``analyses`` list.
    """
    system_prompt = (
        f"You are a {state['role']}. Analyse the given topic from the perspective of "
        f"{state['focus']}. Structure your analysis with clear sections. "
        "Be specific, evidence-based, and actionable."
    )
    human_prompt = (
        f"Topic: {state['query']}\n\n" f"Provide a focused {state['role']} analysis."
    )

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )

    formatted = f"## {state['role']}\n\n{response.content}"
    return {"analyses": [formatted]}


def consolidate_node(state: ParallelState, llm: BaseChatModel) -> dict[str, str]:
    """
    Node: merge the three parallel analyses into a single cohesive report.

    Receives all accumulated analyses from the three branches and uses
    the LLM to synthesise them into an integrated executive summary.

    Args:
        state: Pipeline state with all three analyses populated.
        llm: Configured LangChain chat model.

    Returns:
        A partial state update with ``final_report`` set.
    """
    analyses_text = "\n\n---\n\n".join(state["analyses"])

    system_prompt = (
        "You are a chief strategy officer. You receive three specialist analyses "
        "(technology, market, and risk) on the same topic. Synthesise them into "
        "a cohesive executive report that: highlights cross-cutting themes, "
        "reconciles conflicting views, and ends with prioritised recommendations."
    )
    human_prompt = (
        f"Topic: {state['query']}\n\n"
        f"Specialist analyses:\n\n{analyses_text}\n\n"
        "Produce the consolidated executive report."
    )

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )

    return {"final_report": str(response.content)}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_parallel_graph(llm: BaseChatModel) -> object:
    """
    Build and compile the parallel multi-analyst pipeline.

    The ``fan_out_node`` uses LangGraph's ``Send`` API to dispatch three
    independent ``analyst_node`` executions concurrently.  Results are
    aggregated by the ``operator.add`` reducer before ``consolidate_node``
    merges them.

    Args:
        llm: A configured LangChain ``BaseChatModel`` instance.

    Returns:
        A compiled LangGraph ``StateGraph`` ready for ``.invoke()``.
    """
    graph: StateGraph = StateGraph(ParallelState)

    graph.add_node("fan_out", fan_out_node)
    graph.add_node("analyst_node", lambda state: analyst_node(state, llm))
    graph.add_node("consolidate", lambda state: consolidate_node(state, llm))

    # START -> fan_out -> [analyst_node x3 via Send] -> consolidate -> END
    graph.add_edge(START, "fan_out")
    graph.add_edge("fan_out", "analyst_node")
    graph.add_edge("analyst_node", "consolidate")
    graph.add_edge("consolidate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from core.config import get_settings

    settings = get_settings()
    from core.llm import get_llm

    llm = get_llm(settings.llm_config)
    graph = build_parallel_graph(llm)

    query = "What are the trade-offs of adopting a microservices architecture?"
    print(f"Query: {query}")
    print("-" * 60)

    result: ParallelState = graph.invoke(
        {
            "query": query,
            "analyses": [],
            "final_report": "",
        }
    )

    print("=== INDIVIDUAL ANALYSES ===")
    for analysis in result["analyses"]:
        print(analysis)
        print()

    print("=== CONSOLIDATED REPORT ===")
    print(result["final_report"])
