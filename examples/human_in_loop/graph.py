"""
Human-in-the-loop pattern: agent pauses for human approval before critical actions.

When to use: agents that write to databases, send emails, deploy infrastructure,
or take any irreversible action. The ``interrupt()`` call suspends execution and
returns control to the caller; a ``Command(resume=...)`` resumes it.

Architecture::

    START
      |
    plan_node       (LLM proposes an action)
      |
    approval_node   (interrupt() — pauses here, waits for human input)
      |
    route_approval  (conditional: approved? -> execute, rejected -> END)
      |          \\
    execute_node   END
      |
     END

Run: uv run python examples/human_in_loop/graph.py
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class HumanLoopState(TypedDict):
    """
    State for the human-in-the-loop pipeline.

    Attributes:
        query: The user's original request.
        proposed_action: The action string the agent wishes to perform.
        human_approved: Whether the human approved the proposed action.
        result: The outcome string after execution (or a rejection message).
    """

    query: str
    proposed_action: str
    human_approved: bool
    result: str


# ---------------------------------------------------------------------------
# Shared LLM reference
# ---------------------------------------------------------------------------

_llm: BaseChatModel | None = None

# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def plan_node(state: HumanLoopState) -> dict[str, str]:
    """
    Node: use the LLM to propose a concrete action based on the user's request.

    The proposed action is stored in state so the approval node can
    present it to the human.

    Args:
        state: Current pipeline state.

    Returns:
        Partial state update with ``proposed_action`` set.
    """
    assert _llm is not None, "LLM not initialised — call build_human_loop_graph() first."

    system_prompt = (
        "You are an intelligent automation agent. The user gives you a task. "
        "Your job is to propose a single, specific, concrete action to fulfil it. "
        "Describe the action precisely: what will be done, what data will be changed, "
        "and what the expected outcome is. Be explicit about any side effects."
    )

    response = _llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["query"]),
        ]
    )

    proposed: str = str(response.content)
    return {"proposed_action": proposed}


def approval_node(state: HumanLoopState) -> dict[str, bool]:
    """
    Node: pause execution and request human approval via ``interrupt()``.

    ``interrupt()`` suspends the graph at this node and returns the
    ``proposed_action`` payload to the caller.  Execution resumes only
    when the graph is reinvoked with a ``Command(resume={"approved": bool})``.

    Args:
        state: Current pipeline state containing the proposed action.

    Returns:
        Partial state update with ``human_approved`` set from the human's decision.
    """
    # interrupt() suspends here; the dict is returned to the caller as the
    # interrupt value.  Execution resumes with Command(resume={"approved": ...}).
    human_response: dict = interrupt(
        {
            "message": "Human approval required before executing the following action.",
            "proposed_action": state["proposed_action"],
            "instructions": (
                "Resume with: Command(resume={'approved': True}) to approve, "
                "or Command(resume={'approved': False}) to reject."
            ),
        }
    )

    approved: bool = bool(human_response.get("approved", False))
    return {"human_approved": approved}


def execute_node(state: HumanLoopState) -> dict[str, str]:
    """
    Node: execute the approved action.

    Only reached when ``human_approved`` is ``True``.  In a real system this
    would perform the actual side-effectful operation (database write, API
    call, email send, etc.).  Here we simulate execution and return a result.

    Args:
        state: Current pipeline state with the approved action.

    Returns:
        Partial state update with ``result`` describing the execution outcome.
    """
    assert _llm is not None, "LLM not initialised — call build_human_loop_graph() first."

    system_prompt = (
        "You are an execution engine. The following action has been reviewed and "
        "approved by a human operator. Simulate its execution and report the outcome "
        "in detail, including any confirmation identifiers, timestamps, or affected "
        "records. Treat this as a successful dry-run."
    )
    human_prompt = (
        f"Execute the following approved action:\n\n{state['proposed_action']}"
    )

    response = _llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )

    return {"result": str(response.content)}


def rejection_node(state: HumanLoopState) -> dict[str, str]:
    """
    Node: record a rejection outcome without performing the action.

    Args:
        state: Current pipeline state.

    Returns:
        Partial state update with ``result`` set to a rejection message.
    """
    return {
        "result": (
            "Action rejected by human operator. "
            "No changes were made to any system."
        )
    }


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------


def route_after_approval(state: HumanLoopState) -> str:
    """
    Conditional edge: route to ``execute`` when approved, to ``reject`` otherwise.

    Args:
        state: Current pipeline state.

    Returns:
        Edge key ``"execute"`` or ``"reject"``.
    """
    if state.get("human_approved", False):
        return "execute"
    return "reject"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_human_loop_graph(llm: BaseChatModel) -> tuple[object, MemorySaver]:
    """
    Build and compile the human-in-the-loop pipeline.

    A ``MemorySaver`` checkpointer is required so that ``interrupt()`` can
    persist the suspended state between the initial invocation and the
    ``Command(resume=...)`` call.

    Args:
        llm: A configured LangChain ``BaseChatModel`` instance.

    Returns:
        A 2-tuple of ``(compiled_graph, memory_saver)``.  The caller must
        pass a ``config`` with a ``thread_id`` to both ``.invoke()`` calls
        so LangGraph can restore the suspended state on resume.
    """
    global _llm
    _llm = llm

    checkpointer = MemorySaver()

    graph: StateGraph = StateGraph(HumanLoopState)

    graph.add_node("plan", plan_node)
    graph.add_node("approval", approval_node)
    graph.add_node("execute", execute_node)
    graph.add_node("reject", rejection_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "approval")

    graph.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "execute": "execute",
            "reject": "reject",
        },
    )

    graph.add_edge("execute", END)
    graph.add_edge("reject", END)

    return graph.compile(checkpointer=checkpointer), checkpointer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/home/brescou/Project/langgraph-agent-stack")

    from core.config import settings
    from core.llm import get_llm

    llm = get_llm(settings.llm_config)
    graph, _ = build_human_loop_graph(llm)

    config = {"configurable": {"thread_id": "demo-thread-001"}}

    query = "Delete all records from the 'test_users' table that were created before 2023-01-01"
    print(f"Query: {query}")
    print("-" * 60)

    # --- Phase 1: Run until the interrupt ---
    print("\n[Phase 1] Running graph until human approval is required...")
    initial_state: HumanLoopState = {
        "query": query,
        "proposed_action": "",
        "human_approved": False,
        "result": "",
    }

    outcome = graph.invoke(initial_state, config=config)

    # When interrupted, LangGraph returns the interrupt payload rather than
    # the full final state.  We inspect the __interrupt__ key.
    if hasattr(outcome, "__interrupt__") or (
        isinstance(outcome, dict) and "__interrupt__" in outcome
    ):
        interrupt_payload = (
            outcome["__interrupt__"][0].value
            if isinstance(outcome, dict)
            else outcome.__interrupt__[0].value
        )
        print("\n[INTERRUPT] Approval required:")
        print(f"  Proposed action: {interrupt_payload['proposed_action'][:200]}...")
    else:
        # Fallback: graph may have completed without interrupting (no approval needed)
        print(f"\n[INFO] Proposed action:\n{outcome.get('proposed_action', 'N/A')}")

    # --- Phase 2: Simulate human approval (True = approve) ---
    print("\n[Phase 2] Simulating human APPROVAL (approved=True)...")
    approved_result = graph.invoke(
        Command(resume={"approved": True}),
        config=config,
    )

    print("\n=== EXECUTION RESULT (approved) ===")
    print(approved_result.get("result", "No result."))

    # --- Phase 3: Repeat with rejection to demonstrate the reject path ---
    print("\n" + "=" * 60)
    print("[Phase 3] Demonstrating REJECTION path...")

    config_reject = {"configurable": {"thread_id": "demo-thread-002"}}
    graph.invoke(initial_state, config=config_reject)

    rejected_result = graph.invoke(
        Command(resume={"approved": False}),
        config=config_reject,
    )

    print("\n=== EXECUTION RESULT (rejected) ===")
    print(rejected_result.get("result", "No result."))
