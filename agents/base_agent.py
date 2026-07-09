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
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from typing_extensions import TypedDict

from agents.llm_retry import (
    before_sleep_log_transient_error,
    is_auth_llm_error,
    record_retry_exhausted,
    retry_if_transient_llm_error,
)
from core.config import get_settings
from core.cost import BudgetExceededError, CostTracker, UnknownModelPricingError
from core.llm import get_llm
from core.memory import create_checkpointer
from core.observability import (
    llm_request_duration_seconds,
    llm_requests_total,
    llm_tokens_total,
)
from core.security import InputValidator
from core.tools import get_default_tools

input_validator = InputValidator()

# Env var to point the user at when a provider rejects the credentials.
_PROVIDER_KEY_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "bedrock": "your AWS credentials (AWS_REGION / AWS_ACCESS_KEY_ID)",
    "ollama": "OLLAMA_BASE_URL",
}


def make_auth_error(
    name: str, provider: str, exc: BaseException
) -> AgentAuthenticationError:
    """Build the actionable ``AgentAuthenticationError`` for a rejected credential."""
    key_hint = _PROVIDER_KEY_ENV_VARS.get(provider, "your provider credentials")
    return AgentAuthenticationError(
        f"[{name}] LLM provider '{provider}' rejected the request "
        f"credentials: {exc}. Check {key_hint} in your .env (see .env.example), "
        "or set LLM_PROVIDER=mock to run without an API key."
    )


def extract_text_content(content: Any) -> str:
    """Safely extract text from an LLM message content field.

    Multi-modal models may return ``list[dict]`` instead of ``str``.
    This helper normalises both representations to a plain string so
    that ``json.loads`` never receives an unexpected type.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class AgentError(Exception):
    """Base exception for all agent-level errors."""


class AgentConfigurationError(AgentError):
    """Raised when an agent is misconfigured at startup."""


class AgentExecutionError(AgentError):
    """Raised when an agent fails during graph execution."""


class AgentAuthenticationError(AgentExecutionError):
    """Raised when the LLM provider rejects the configured credentials.

    Subclasses ``AgentExecutionError`` so existing handlers still catch it,
    but is never converted into a degraded in-band response: nodes and packs
    re-raise it so the API can return an actionable HTTP error instead of a
    200 with placeholder content.
    """


class AgentTimeoutError(AgentError):
    """Raised when an agent exceeds its allotted step budget."""


class AgentValidationError(AgentError):
    """Raised when an agent receives or produces invalid data."""


class AgentBudgetExceededError(AgentError):
    """Raised when an agent run exceeds its configured USD cost budget."""


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
    * ``run(query: str) -> str`` — public entry point that executes the graph.

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
        budget_usd: Maximum USD cost allowed for this agent's LLM calls.
            Set to ``None`` to disable budget enforcement (default).
            Set to ``0.0`` is valid but will immediately raise
            ``AgentBudgetExceededError`` on the first LLM call that
            incurs any cost — use only for testing.
            Ignored when ``cost_tracker`` is provided.
        cost_tracker: Optional pre-built ``CostTracker`` shared across several
            agents of the same pipeline run.  When provided, the agent attaches
            this tracker to its LLM instead of creating its own, so the
            tracker's budget applies to the *cumulative* spend of all agents
            sharing it.  Enforcement semantics are unchanged — the same
            ``AgentBudgetExceededError`` is raised on overrun.

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
        budget_usd: float | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        self.name: str = name
        self.thread_id: str = thread_id or str(uuid.uuid4())
        self.tools: list[BaseTool] = tools if tools is not None else get_default_tools()
        self._start_time: float = time.monotonic()

        self._log = logging.getLogger(f"{__name__}.{name}")

        _settings = get_settings()

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

        # Resolve the effective tracker. An injected tracker (shared across a
        # pipeline run) takes precedence and is attached as-is so the budget
        # applies to the cumulative spend of every agent sharing it.
        # Otherwise an explicit budget_usd argument takes precedence over the
        # settings-level default; None on both sides disables tracking.
        self._cost_tracker: CostTracker | None = cost_tracker
        if self._cost_tracker is None:
            _effective_budget: float | None = (
                budget_usd
                if budget_usd is not None
                else _settings.pack_default_budget_usd
            )
            if _effective_budget is not None:
                self._cost_tracker = CostTracker(budget_usd=_effective_budget)

        if self._cost_tracker is not None:
            # with_config returns a new Runnable wrapper; it does not mutate the
            # underlying BaseChatModel object. If _shared_llm is passed from the
            # API layer, this call creates a per-agent view with the cost tracker
            # attached while leaving the shared instance unmodified.
            self.llm = cast(
                BaseChatModel,
                self.llm.with_config({"callbacks": [self._cost_tracker]}),
            )

        # Pre-bound LLM with tool schemas attached.  Not used by the
        # built-in agents (they call _invoke_llm_with_retry directly),
        # but available for subclasses that need tool-calling via
        # LangChain's native bind_tools() API. Some chat models (e.g.
        # FakeListChatModel, used by ``LLM_PROVIDER=mock``) do not implement
        # bind_tools() and raise NotImplementedError; fall back to the plain
        # LLM in that case instead of failing agent construction.
        self.llm_with_tools: Any = self.llm
        if self.tools:
            try:
                self.llm_with_tools = self.llm.bind_tools(self.tools)
            except NotImplementedError:
                self._log.debug(
                    "LLM does not support bind_tools(); using unbound LLM",
                    extra={"agent": self.name},
                )

        self.checkpointer = (
            checkpointer if checkpointer is not None else create_checkpointer(_settings)
        )

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

        self._graph = self.build_graph()

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
    def run(self, query: str) -> str:
        """
        Execute the agent graph for the given query string.

        Args:
            query: The user query or task description.

        Returns:
            A string representation of the final agent output.

        Raises:
            AgentExecutionError: If the graph encounters an unrecoverable error.
            AgentTimeoutError: If the step count exceeds ``settings.max_step_count``.
        """

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    def _make_initial_state(self, query: str) -> AgentState:
        """
        Build a fresh ``AgentState`` for the start of a new run.

        Args:
            query: The raw user query string.

        Returns:
            A fully populated ``AgentState`` dict ready to pass to the graph.
        """
        from langchain_core.messages import HumanMessage

        return AgentState(
            messages=[HumanMessage(content=query)],
            context={},
            metadata={
                "agent": self.name,
                "thread_id": self.thread_id,
                "run_id": str(uuid.uuid4()),
                "started_at": time.time(),
                "input": query,
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

    @property
    def cost_usd(self) -> float:
        """Return total USD cost for this agent's LLM calls so far."""
        return self._cost_tracker.total_cost_usd if self._cost_tracker else 0.0

    def _invoke_llm_with_retry(
        self,
        messages: list[BaseMessage],
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> Any:
        """Invoke the LLM with exponential-backoff retry on transient errors.

        Retries are handled here only — vendor SDK auto-retries are disabled in
        :func:`core.llm.get_llm` (``max_retries=0``) to avoid stacked backoff.
        Uses tenacity on typed SDK / HTTP exceptions (rate limits, timeouts, 5xx).

        Args:
            messages: LangChain message list to send.
            max_retries: Maximum number of retry attempts after the first failure.
            base_delay: Multiplier for exponential backoff (seconds).
            max_delay: Upper bound on wait between attempts (seconds).

        Returns:
            The LLM response (``AIMessage``).

        Raises:
            AgentExecutionError: When all retries are exhausted or error is fatal.
            AgentBudgetExceededError: When the cost budget is exceeded.
        """
        settings = get_settings()
        max_attempts = max_retries + 1

        def _invoke_once() -> Any:
            try:
                t0 = time.monotonic()
                result = self.llm.invoke(messages)
            except BudgetExceededError as exc:
                raise AgentBudgetExceededError(str(exc)) from exc
            except UnknownModelPricingError as exc:
                raise AgentBudgetExceededError(str(exc)) from exc
            except Exception as exc:
                if retry_if_transient_llm_error(exc):
                    if llm_requests_total is not None:
                        llm_requests_total.labels(
                            provider=settings.llm_provider,
                            status="retryable_error",
                        ).inc()
                    raise
                if llm_requests_total is not None:
                    llm_requests_total.labels(
                        provider=settings.llm_provider,
                        status="fatal_error",
                    ).inc()
                if is_auth_llm_error(exc):
                    raise make_auth_error(
                        self.name, settings.llm_provider, exc
                    ) from exc
                raise AgentExecutionError(
                    f"[{self.name}] LLM call failed: {exc}"
                ) from exc

            elapsed = time.monotonic() - t0
            if llm_request_duration_seconds is not None:
                llm_request_duration_seconds.labels(
                    provider=settings.llm_provider,
                ).observe(elapsed)
            if llm_requests_total is not None:
                llm_requests_total.labels(
                    provider=settings.llm_provider, status="success"
                ).inc()
            usage = getattr(result, "usage_metadata", None)
            if usage and llm_tokens_total is not None:
                if "input_tokens" in usage:
                    llm_tokens_total.labels(
                        provider=settings.llm_provider, direction="input"
                    ).inc(usage["input_tokens"])
                if "output_tokens" in usage:
                    llm_tokens_total.labels(
                        provider=settings.llm_provider, direction="output"
                    ).inc(usage["output_tokens"])
            return result

        retryer = Retrying(
            retry=retry_if_exception(retry_if_transient_llm_error),
            stop=stop_after_attempt(max_attempts),
            wait=wait_random_exponential(multiplier=base_delay, max=max_delay),
            reraise=True,
            before_sleep=before_sleep_log_transient_error(
                self._log, provider=settings.llm_provider
            ),
        )

        try:
            return retryer(_invoke_once)
        except AgentBudgetExceededError:
            raise
        except AgentExecutionError:
            raise
        except Exception as exc:
            # Only transient errors reach this branch (fatal ones are wrapped
            # into AgentExecutionError inside _invoke_once), so the retry
            # budget was exhausted without recovery.
            record_retry_exhausted(settings.llm_provider)
            if llm_requests_total is not None:
                llm_requests_total.labels(
                    provider=settings.llm_provider,
                    status="fatal_error",
                ).inc()
            raise AgentExecutionError(
                f"[{self.name}] LLM call failed after {max_attempts} attempts: {exc}"
            ) from exc
