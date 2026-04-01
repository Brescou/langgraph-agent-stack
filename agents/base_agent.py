"""
agents/base_agent.py — Abstract base class for all LangGraph agents.

Every concrete agent in this stack MUST inherit from ``BaseAgent``.  The base
class provides:

* A typed ``AgentState`` (TypedDict) shared across all graph nodes.
* Pluggable memory/checkpointing (SQLite for development, Redis for production).
* Structured JSON logging via the standard ``logging`` module.
* A uniform ``run()`` interface so orchestrators can treat any agent
  polymorphically.
* Custom exception hierarchy for predictable error handling.
"""

from __future__ import annotations

import abc
import logging
import time
import uuid
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from typing_extensions import TypedDict

from core.config import get_settings
from core.llm import get_llm
from core.memory import create_checkpointer
from core.tools import get_default_tools

# ---------------------------------------------------------------------------
# Logging — structured, JSON-friendly via ``extra`` dict
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class AgentError(Exception):
    """Base exception for all agent-level errors."""


class AgentConfigurationError(AgentError):
    """Raised when an agent is misconfigured at startup."""


class AgentExecutionError(AgentError):
    """Raised when an agent fails during graph execution."""


class AgentTimeoutError(AgentError):
    """Raised when an agent exceeds its allotted step budget."""


class AgentValidationError(AgentError):
    """Raised when an agent receives or produces invalid data."""


# ---------------------------------------------------------------------------
# Shared state schema
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """
    Canonical state object passed between every graph node.

    Attributes:
        messages: Conversation history as LangChain message objects.
        context: Arbitrary key-value pairs accumulated during execution
            (e.g. retrieved documents, intermediate results).
        metadata: Run-level metadata: agent name, run_id, timestamps, etc.
        step_count: Monotonically increasing counter incremented by each node.
        error: Optional error message set when a node catches an exception.
        status: High-level execution status (``running`` | ``done`` | ``error``).
    """

    messages: list[BaseMessage]
    context: dict[str, Any]
    metadata: dict[str, Any]
    step_count: int
    error: str | None
    status: str


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------


class BaseAgent(abc.ABC):
    """
    Abstract base class for all LangGraph-powered agents.

    Subclasses MUST implement:

    * ``build_graph()`` — construct and return a compiled LangGraph ``StateGraph``.
    * ``run(input: str) -> str`` — public entry point that executes the graph.

    The constructor wires up the LLM client, checkpointer, and structured
    logger automatically from the shared ``Settings`` singleton.

    Args:
        name: Human-readable agent identifier (used in logs and metadata).
        thread_id: Optional stable ID for resuming a checkpointed conversation.
            A new UUID is generated when omitted.
        tools: Optional list of LangChain ``BaseTool`` instances made available
            to this agent.  When provided, the tools are stored on
            ``self.tools`` and subclasses may bind them to the LLM via
            ``self.llm.bind_tools(self.tools)`` inside ``build_graph()``.
            Defaults to an empty list when omitted.

    Raises:
        AgentConfigurationError: If the LLM client cannot be initialised
            (e.g. missing API key).
    """

    def __init__(
        self,
        name: str,
        thread_id: str | None = None,
        tools: list[BaseTool] | None = None,
        llm: BaseChatModel | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self.name: str = name
        self.thread_id: str = thread_id or str(uuid.uuid4())
        # Caller-supplied tools take precedence; fall back to the default tool set.
        self.tools: list[BaseTool] = tools if tools is not None else get_default_tools()
        self._start_time: float = time.monotonic()

        # Structured logger — include agent name and thread in every record
        self._log = logging.getLogger(f"{__name__}.{name}")

        _settings = get_settings()

        # Build LLM client — use injected instance or create from settings.
        if llm is not None:
            self.llm: BaseChatModel = llm
        else:
            try:
                self.llm = get_llm(_settings.llm_config)
            except (ImportError, ValueError) as exc:
                raise AgentConfigurationError(
                    f"[{self.name}] Failed to initialise LLM provider "
                    f"'{_settings.llm_provider}': {exc}"
                ) from exc

        # LLM variant with tools bound for structured tool-calling nodes.
        self.llm_with_tools: BaseChatModel = (
            self.llm.bind_tools(self.tools) if self.tools else self.llm
        )

        # Build checkpointer — use injected instance or create from settings.
        self.checkpointer = (
            checkpointer if checkpointer is not None else create_checkpointer(_settings)
        )

        # Compile graph (delegated to subclass)
        self._graph = self.build_graph()

        self._log.info(
            "Agent initialised",
            extra={
                "agent": self.name,
                "thread_id": self.thread_id,
                "llm_provider": _settings.llm_provider,
                "memory_backend": _settings.memory_backend.value,
                "tools": [t.name for t in self.tools],
            },
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def build_graph(self) -> Any:
        """
        Construct the compiled LangGraph ``StateGraph`` for this agent.

        Called once during ``__init__``.  The returned object is stored as
        ``self._graph`` and invoked by ``run()``.

        Returns:
            A compiled LangGraph graph (result of ``graph.compile(...)``).
        """

    @abc.abstractmethod
    def run(self, input: str) -> str:
        """
        Execute the agent graph for the given input string.

        Args:
            input: The user query or task description.

        Returns:
            A string representation of the final agent output.

        Raises:
            AgentExecutionError: If the graph encounters an unrecoverable error.
            AgentTimeoutError: If the step count exceeds ``settings.max_step_count``.
        """

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    def _make_initial_state(self, input: str) -> AgentState:
        """
        Build a fresh ``AgentState`` for the start of a new run.

        Args:
            input: The raw user input string.

        Returns:
            A fully populated ``AgentState`` dict ready to pass to the graph.
        """
        from langchain_core.messages import HumanMessage

        return AgentState(
            messages=[HumanMessage(content=input)],
            context={},
            metadata={
                "agent": self.name,
                "thread_id": self.thread_id,
                "run_id": str(uuid.uuid4()),
                "started_at": time.time(),
                "input": input,
            },
            step_count=0,
            error=None,
            status="running",
        )

    def _get_config(self) -> dict[str, Any]:
        """
        Return the LangGraph invocation config for checkpointing.

        Returns:
            A dict with ``configurable`` sub-dict expected by LangGraph.
        """
        return {"configurable": {"thread_id": self.thread_id}}

    def _increment_step(self, state: AgentState) -> AgentState:
        """
        Increment the step counter and raise if the budget is exhausted.

        Args:
            state: Current agent state.

        Returns:
            Updated state with ``step_count`` incremented by one.

        Raises:
            AgentTimeoutError: When ``step_count`` reaches ``max_step_count``.
        """
        new_count = state.get("step_count", 0) + 1
        max_steps = get_settings().max_step_count
        if new_count > max_steps:
            raise AgentTimeoutError(
                f"[{self.name}] Exceeded max_step_count={max_steps}"
            )
        return {**state, "step_count": new_count}  # type: ignore[return-value]

    def _log_step(self, node_name: str, state: AgentState) -> None:
        """
        Emit a structured DEBUG log entry for a graph node transition.

        Args:
            node_name: Name of the node being entered.
            state: Current agent state at entry.
        """
        self._log.debug(
            "Entering node",
            extra={
                "agent": self.name,
                "node": node_name,
                "step": state.get("step_count", 0),
                "status": state.get("status", "unknown"),
            },
        )

    def elapsed_seconds(self) -> float:
        """Return wall-clock seconds since this agent was instantiated."""
        return time.monotonic() - self._start_time

    def _invoke_llm_with_retry(
        self,
        messages: list[BaseMessage],
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> Any:
        """Invoke the LLM with exponential-backoff retry on transient errors.

        Retries on ``TimeoutError``, ``ConnectionError``, and generic
        ``Exception`` subclasses that indicate rate limiting (status 429).
        Non-transient errors are re-raised immediately.

        Args:
            messages: LangChain message list to send.
            max_retries: Maximum number of retry attempts.
            base_delay: Initial delay in seconds before the first retry.
            max_delay: Upper bound on the exponential delay.

        Returns:
            The LLM response (``AIMessage``).

        Raises:
            AgentExecutionError: When all retries are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return self.llm.invoke(messages)
            except (TimeoutError, ConnectionError) as exc:
                last_exc = exc
                if attempt < max_retries:
                    delay = min(base_delay * (2**attempt), max_delay)
                    self._log.warning(
                        "LLM call failed (attempt %d/%d), retrying in %.1fs",
                        attempt + 1,
                        max_retries + 1,
                        delay,
                        extra={"error": str(exc)},
                    )
                    time.sleep(delay)
            except Exception as exc:
                err_str = str(exc).lower()
                if "429" in err_str or "rate" in err_str:
                    last_exc = exc
                    if attempt < max_retries:
                        delay = min(base_delay * (2**attempt), max_delay)
                        self._log.warning(
                            "LLM rate limited (attempt %d/%d), retrying in %.1fs",
                            attempt + 1,
                            max_retries + 1,
                            delay,
                            extra={"error": str(exc)},
                        )
                        time.sleep(delay)
                    continue
                raise AgentExecutionError(
                    f"[{self.name}] LLM call failed: {exc}"
                ) from exc

        raise AgentExecutionError(
            f"[{self.name}] LLM call failed after {max_retries + 1} attempts: {last_exc}"
        ) from last_exc
