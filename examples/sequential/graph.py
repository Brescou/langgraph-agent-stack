"""
Sequential pattern: ResearchAgent -> AnalystAgent (linear pipeline).

When to use: tasks with clear dependencies where each step builds upon
the previous output. Ideal for structured workflows like research-then-analysis,
extract-then-transform, or draft-then-review pipelines.

Run: uv run python examples/sequential/graph.py
"""

from __future__ import annotations

import operator
from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class SequentialState(TypedDict):
    """
    State passed between nodes in the sequential pipeline.

    Attributes:
        query: The original user question or topic.
        research_output: Raw findings produced by the research node.
        analysis_output: Final analysis produced by the analyst node.
        messages: Accumulated conversation/event log (append-only via operator.add).
    """

    query: str
    research_output: str
    analysis_output: str
    messages: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def research_node(state: SequentialState, llm: BaseChatModel) -> SequentialState:
    """
    Node 1 — Research: gather and summarise information on the query.

    Calls the LLM with a researcher persona to produce a structured
    information brief.

    Args:
        state: Current pipeline state.
        llm: Configured LangChain chat model.

    Returns:
        Updated state with ``research_output`` populated.
    """
    query = state["query"]

    system_prompt = (
        "You are a thorough research assistant. Given a topic, produce a "
        "structured information brief covering: key concepts, current state of "
        "the art, main challenges, and notable examples. Be factual and concise."
    )
    human_prompt = f"Research the following topic in depth:\n\n{query}"

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )

    research_text: str = str(response.content)

    return {
        "query": query,
        "research_output": research_text,
        "analysis_output": "",
        "messages": [
            HumanMessage(content=human_prompt),
            AIMessage(content=research_text),
        ],
    }


def analyze_node(state: SequentialState, llm: BaseChatModel) -> SequentialState:
    """
    Node 2 — Analyze: derive insights from the research output.

    Receives the research brief from the previous node and uses the LLM
    to produce a strategic analysis with actionable conclusions.

    Args:
        state: Current pipeline state (must contain ``research_output``).
        llm: Configured LangChain chat model.

    Returns:
        Updated state with ``analysis_output`` populated.
    """
    query = state["query"]
    research = state["research_output"]

    system_prompt = (
        "You are a senior analyst. Given a research brief, produce a rigorous "
        "analysis that includes: key insights, implications, trade-offs, "
        "recommendations, and open questions. Use clear structure."
    )
    human_prompt = (
        f"Original question: {query}\n\n"
        f"Research brief:\n{research}\n\n"
        "Provide a strategic analysis of these findings."
    )

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )

    analysis_text: str = str(response.content)

    return {
        "query": query,
        "research_output": research,
        "analysis_output": analysis_text,
        "messages": [
            HumanMessage(content=human_prompt),
            AIMessage(content=analysis_text),
        ],
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_sequential_graph(llm: BaseChatModel) -> object:
    """
    Build and compile the sequential Research -> Analyze pipeline.

    Topology::

        research_node -> analyze_node -> END

    Nodes are wrapped in closures so they capture the ``llm`` instance
    without requiring it to be part of the state schema.

    Args:
        llm: A configured LangChain ``BaseChatModel`` instance.

    Returns:
        A compiled LangGraph ``StateGraph`` ready for ``.invoke()``.
    """
    graph: StateGraph = StateGraph(SequentialState)

    # Bind the LLM into each node via closure
    graph.add_node("research", lambda state: research_node(state, llm))
    graph.add_node("analyze", lambda state: analyze_node(state, llm))

    # Linear edges: research -> analyze -> END
    graph.set_entry_point("research")
    graph.add_edge("research", "analyze")
    graph.add_edge("analyze", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/home/brescou/Project/langgraph-agent-stack")

    from core.config import settings
    from core.llm import get_llm

    llm = get_llm(settings.llm_config)
    graph = build_sequential_graph(llm)

    query = "What are the benefits of microservices architecture?"
    print(f"Query: {query}")
    print("-" * 60)

    result: SequentialState = graph.invoke(
        {
            "query": query,
            "research_output": "",
            "analysis_output": "",
            "messages": [],
        }
    )

    print("=== RESEARCH OUTPUT ===")
    print(result["research_output"])
    print()
    print("=== ANALYSIS OUTPUT ===")
    print(result["analysis_output"])
