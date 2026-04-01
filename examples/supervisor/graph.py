"""
Supervisor pattern: a router agent delegates to specialised agents.

When to use: multi-domain assistants where the appropriate specialist
depends on the content of each query. The supervisor inspects the query,
decides which agent to call, receives its output, and either routes to
another agent or terminates.

Architecture::

    START
      |
    supervisor_node  <---+
     |  |  |  |          |
    research code data  (loop back to supervisor after each specialist)
     |  |  |  |          |
    supervisor_node  ----+
      |
     END  (when supervisor returns FINISH)

Run: uv run python examples/supervisor/graph.py
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

AgentName = Literal["research", "code", "data", "FINISH"]


class SupervisorState(TypedDict):
    """
    State shared across the supervisor pipeline.

    Attributes:
        query: The original user query.
        next_agent: Routing decision made by the supervisor node.
            One of ``"research"``, ``"code"``, ``"data"``, or ``"FINISH"``.
        agent_output: The last specialist agent's response string.
        messages: Full conversation history, append-only via ``operator.add``.
    """

    query: str
    next_agent: str
    agent_output: str
    messages: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# Shared LLM reference
# ---------------------------------------------------------------------------

_llm: BaseChatModel | None = None

# ---------------------------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------------------------

_SUPERVISOR_SYSTEM = """
You are a supervisor that routes user queries to the correct specialist agent.

Available agents:
- research: answers factual questions, explains concepts, summarises information
- code: writes, reviews, or debugs code; explains algorithms
- data: analyses datasets, creates SQL queries, interprets statistics

Return a JSON object with exactly one key "next":
- Set "next" to the name of the most appropriate agent for the current query.
- Set "next" to "FINISH" if the query has been fully answered by the previous agent output.

Previous agent output (if any) is included in the conversation history.
Respond ONLY with the JSON object, no other text.
""".strip()


def supervisor_node(state: SupervisorState) -> dict[str, str]:
    """
    Node: inspect the conversation and decide which agent to call next.

    Uses the LLM as a router.  The model receives the full message history
    and returns a JSON routing decision.

    Args:
        state: Current pipeline state.

    Returns:
        Partial state update with ``next_agent`` set to the routing decision.
    """
    assert _llm is not None, "LLM not initialised — call build_supervisor_graph() first."

    # Build the routing prompt from current conversation context
    history = state.get("messages", [])
    if not history:
        # First turn — seed with the original query
        history = [HumanMessage(content=state["query"])]

    response = _llm.invoke(
        [SystemMessage(content=_SUPERVISOR_SYSTEM)] + history
    )

    raw: str = str(response.content).strip()

    # Parse routing decision; default to FINISH on parse failure
    try:
        decision: dict = json.loads(raw)
        next_agent: str = decision.get("next", "FINISH")
        if next_agent not in ("research", "code", "data", "FINISH"):
            next_agent = "FINISH"
    except (json.JSONDecodeError, AttributeError):
        next_agent = "FINISH"

    return {"next_agent": next_agent}


# ---------------------------------------------------------------------------
# Specialist agent nodes
# ---------------------------------------------------------------------------

_AGENT_PERSONAS: dict[str, str] = {
    "research": (
        "You are a senior research analyst. Answer the query with thorough, "
        "well-structured information. Cite key concepts and provide context."
    ),
    "code": (
        "You are an expert software engineer. Provide clean, well-commented code "
        "or a clear technical explanation as appropriate. Follow best practices."
    ),
    "data": (
        "You are a data scientist and SQL expert. Provide rigorous analysis, "
        "SQL queries, or statistical interpretations with clear explanations."
    ),
}


def _make_specialist_node(role: str):
    """
    Factory: create a specialist node function for the given role.

    Args:
        role: One of ``"research"``, ``"code"``, or ``"data"``.

    Returns:
        A node function compatible with LangGraph's ``add_node``.
    """

    def specialist_node(state: SupervisorState) -> dict:
        assert _llm is not None, "LLM not initialised."

        persona = _AGENT_PERSONAS[role]
        query = state["query"]
        prior_output = state.get("agent_output", "")

        human_content = query
        if prior_output:
            human_content = (
                f"Original query: {query}\n\n"
                f"Prior agent output:\n{prior_output}\n\n"
                "Continue or refine the answer based on the above."
            )

        response = _llm.invoke(
            [
                SystemMessage(content=persona),
                HumanMessage(content=human_content),
            ]
        )

        output_text: str = str(response.content)

        return {
            "agent_output": output_text,
            "messages": [
                HumanMessage(content=f"[{role.upper()} AGENT] {human_content}"),
                AIMessage(content=output_text),
            ],
        }

    specialist_node.__name__ = f"{role}_node"
    return specialist_node


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------


def route_after_supervisor(state: SupervisorState) -> AgentName:
    """
    Conditional edge: read ``next_agent`` from state and return the edge key.

    Args:
        state: Current pipeline state.

    Returns:
        The name of the next node, or ``END`` sentinel when finished.
    """
    decision = state.get("next_agent", "FINISH")
    if decision == "FINISH":
        return "FINISH"
    return decision  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_supervisor_graph(llm: BaseChatModel) -> object:
    """
    Build and compile the supervisor routing pipeline.

    The supervisor runs first, then routes to a specialist agent.  After
    the specialist responds, control returns to the supervisor which may
    call another agent or terminate with ``FINISH``.

    Args:
        llm: A configured LangChain ``BaseChatModel`` instance.

    Returns:
        A compiled LangGraph ``StateGraph`` ready for ``.invoke()``.
    """
    global _llm
    _llm = llm

    graph: StateGraph = StateGraph(SupervisorState)

    # Add supervisor node
    graph.add_node("supervisor", supervisor_node)

    # Add specialist nodes
    for role in ("research", "code", "data"):
        graph.add_node(role, _make_specialist_node(role))

    # Routing: supervisor -> {research, code, data, END}
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "research": "research",
            "code": "code",
            "data": "data",
            "FINISH": END,
        },
    )

    # Each specialist loops back to supervisor
    for role in ("research", "code", "data"):
        graph.add_edge(role, "supervisor")

    graph.set_entry_point("supervisor")

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
    graph = build_supervisor_graph(llm)

    queries = [
        "Explain how transformer attention mechanisms work",
        "Write a Python function that computes the Fibonacci sequence iteratively",
        "What SQL query would I use to find the top 5 customers by total order value?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)

        result: SupervisorState = graph.invoke(
            {
                "query": query,
                "next_agent": "",
                "agent_output": "",
                "messages": [HumanMessage(content=query)],
            }
        )

        print(f"Routed to: {result['next_agent'] or 'FINISH'}")
        print()
        print("=== AGENT OUTPUT ===")
        print(result["agent_output"])
        print()
