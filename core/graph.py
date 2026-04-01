"""
core/graph.py — MultiAgentGraph: top-level orchestrator for the agent pipeline.

Wires ``ResearchAgent`` and ``AnalystAgent`` into a sequential pipeline:

    Input → ResearchAgent → AnalystAgent → AnalysisReport (output)

The orchestrator itself is built as a LangGraph ``StateGraph`` so the full
execution trace is checkpointed and observable.  Each agent runs as a
dedicated node; conditional edges allow the pipeline to short-circuit when a
step fails.

Async support
-------------
Both ``run()`` (sync) and ``arun()`` (async) are provided.  The async variant
runs the blocking LangGraph calls in a thread executor to avoid blocking the
event loop.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from agents.analyst import AnalysisReport, AnalystAgent
from agents.base_agent import AgentError, AgentExecutionError, AgentValidationError
from agents.researcher import ResearchAgent, ResearchResult
from core.config import get_settings
from core.memory import create_checkpointer
from core.observability import trace_span

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Orchestrator state schema
# ---------------------------------------------------------------------------


class OrchestratorState(TypedDict, total=False):
    """
    State shared across the MultiAgentGraph pipeline nodes.

    Attributes:
        query: The original user query string.
        research_result: Serialised ``ResearchResult`` dict from ResearchAgent.
        analysis_report: Serialised ``AnalysisReport`` dict from AnalystAgent.
        error: Error message if a node fails; ``None`` on success.
        status: Pipeline lifecycle: ``running`` | ``research_done``
            | ``analysis_done`` | ``error``.
        metadata: Arbitrary run-level metadata (run_id, timestamps, …).
    """

    query: str
    research_result: dict[str, Any] | None
    analysis_report: dict[str, Any] | None
    error: str | None
    status: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# MultiAgentGraph
# ---------------------------------------------------------------------------


class MultiAgentGraph:
    """
    Top-level orchestrator that sequences ``ResearchAgent`` and ``AnalystAgent``.

    Usage::

        from core.graph import MultiAgentGraph

        pipeline = MultiAgentGraph()
        report = pipeline.run("What are the latest trends in vector databases?")
        print(report.to_markdown())

    Args:
        run_id: Optional stable identifier for the pipeline run.  A UUID is
            generated when omitted.

    Raises:
        AgentExecutionError: When either agent fails during the pipeline run.
    """

    def __init__(
        self,
        run_id: str | None = None,
        llm: BaseChatModel | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self.run_id: str = run_id or str(uuid.uuid4())
        self._llm = llm
        self._checkpointer = checkpointer or create_checkpointer(get_settings())
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._research_agent: ResearchAgent | None = None
        self._analyst_agent: AnalystAgent | None = None
        self._graph = self._build_graph()

        logger.info(
            "MultiAgentGraph initialised",
            extra={"run_id": self.run_id},
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        """
        Construct and compile the orchestrator StateGraph.

        Nodes:
            - ``research_node``: runs ``ResearchAgent.run_structured()``.
            - ``analysis_node``: runs ``AnalystAgent.run_structured()``.

        Edges:
            research_node → (conditional) → analysis_node | END (on error)
            analysis_node → END

        Returns:
            Compiled LangGraph graph.
        """
        graph = StateGraph(OrchestratorState)

        graph.add_node("research_node", self._research_node)
        graph.add_node("analysis_node", self._analysis_node)

        graph.set_entry_point("research_node")

        graph.add_conditional_edges(
            "research_node",
            self._route_after_research,
            {
                "analysis_node": "analysis_node",
                "end_on_error": END,
            },
        )
        graph.add_edge("analysis_node", END)

        return graph.compile(checkpointer=self._checkpointer)

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _research_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        Orchestrator node: instantiate and run ``ResearchAgent``.

        Args:
            state: Current orchestrator state.

        Returns:
            Updated state with ``research_result`` populated (or ``error`` set).
        """
        query: str = state.get("query", "")
        logger.info(
            "Pipeline: starting research phase",
            extra={"run_id": self.run_id, "query": query[:120]},
        )

        with trace_span("research_node", {"run_id": self.run_id}):
            try:
                if self._research_agent is None:
                    self._research_agent = ResearchAgent(
                        thread_id=f"{self.run_id}-research",
                        llm=self._llm,
                        checkpointer=self._checkpointer,
                    )
                result: ResearchResult = self._research_agent.run_structured(query)
                logger.info(
                    "Pipeline: research phase complete",
                    extra={
                        "run_id": self.run_id,
                        "confidence": result.confidence,
                        "findings": len(result.findings),
                    },
                )
                return {
                    **state,
                    "research_result": result.to_dict(),
                    "status": "research_done",
                    "error": None,
                }  # type: ignore[return-value]

            except AgentError as exc:
                logger.error(
                    "Pipeline: research phase failed",
                    extra={"run_id": self.run_id, "error": str(exc)},
                )
                return {
                    **state,
                    "research_result": None,
                    "status": "error",
                    "error": str(exc),
                }  # type: ignore[return-value]

    def _analysis_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        Orchestrator node: instantiate and run ``AnalystAgent``.

        Reads the ``ResearchResult`` from state and passes it to the analyst.

        Args:
            state: Current orchestrator state (must contain ``research_result``).

        Returns:
            Updated state with ``analysis_report`` populated (or ``error`` set).
        """
        logger.info(
            "Pipeline: starting analysis phase",
            extra={"run_id": self.run_id},
        )

        research_dict = state.get("research_result")
        if not research_dict:
            err = "analysis_node received no research_result"
            logger.error("Pipeline: analysis phase skipped", extra={"reason": err})
            return {
                **state,
                "status": "error",
                "error": err,
            }  # type: ignore[return-value]

        try:
            research_result = ResearchResult(**research_dict)
        except (TypeError, KeyError, ValueError) as exc:
            err = f"Failed to deserialise ResearchResult: {exc}"
            logger.error(
                "Pipeline: analysis phase failed (bad research_result)",
                extra={"run_id": self.run_id, "error": err},
            )
            return {
                **state,
                "analysis_report": None,
                "status": "error",
                "error": err,
            }  # type: ignore[return-value]

        with trace_span("analysis_node", {"run_id": self.run_id}):
            try:
                if self._analyst_agent is None:
                    self._analyst_agent = AnalystAgent(
                        thread_id=f"{self.run_id}-analysis",
                        llm=self._llm,
                        checkpointer=self._checkpointer,
                    )
                report: AnalysisReport = self._analyst_agent.run_structured(
                    research_result
                )

                logger.info(
                    "Pipeline: analysis phase complete",
                    extra={
                        "run_id": self.run_id,
                        "confidence": report.confidence,
                        "insights": len(report.key_insights),
                    },
                )
                return {
                    **state,
                    "analysis_report": report.to_dict(),
                    "status": "analysis_done",
                    "error": None,
                }  # type: ignore[return-value]

            except AgentError as exc:
                logger.error(
                    "Pipeline: analysis phase failed",
                    extra={"run_id": self.run_id, "error": str(exc)},
                )
                return {
                    **state,
                    "analysis_report": None,
                    "status": "error",
                    "error": str(exc),
                }  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Conditional routing
    # ------------------------------------------------------------------

    def _route_after_research(
        self, state: OrchestratorState
    ) -> Literal["analysis_node", "end_on_error"]:
        """
        Route to analysis when research succeeded, otherwise short-circuit.

        Args:
            state: Current orchestrator state.

        Returns:
            Edge key ``"analysis_node"`` or ``"end_on_error"``.
        """
        if state.get("status") == "error" or state.get("error"):
            logger.warning(
                "Pipeline: routing to end due to research error",
                extra={"error": state.get("error", "")},
            )
            return "end_on_error"
        return "analysis_node"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, query: str) -> AnalysisReport:
        """
        Execute the full Research → Analysis pipeline synchronously.

        Args:
            query: The research question or topic to investigate.

        Returns:
            A populated ``AnalysisReport`` dataclass.

        Raises:
            AgentValidationError: When the query is empty.
            AgentExecutionError: When any pipeline stage fails.
        """
        if not query or not query.strip():
            raise AgentValidationError(
                "MultiAgentGraph.run() requires a non-empty query."
            )

        logger.info(
            "MultiAgentGraph.run() started",
            extra={"run_id": self.run_id, "query": query[:120]},
        )

        initial_state: OrchestratorState = {
            "query": query.strip(),
            "research_result": None,
            "analysis_report": None,
            "error": None,
            "status": "running",
            "metadata": {"run_id": self.run_id},
        }

        config = {"configurable": {"thread_id": self.run_id}}

        try:
            final_state: OrchestratorState = self._graph.invoke(
                initial_state, config=config
            )
        except Exception as exc:
            raise AgentExecutionError(
                f"[MultiAgentGraph] Pipeline execution failed: {exc}"
            ) from exc

        if final_state.get("status") == "error":
            raise AgentExecutionError(
                f"[MultiAgentGraph] Pipeline error: {final_state.get('error')}"
            )

        report_dict = final_state.get("analysis_report")
        if not report_dict:
            raise AgentExecutionError(
                "[MultiAgentGraph] Pipeline completed without an AnalysisReport."
            )

        try:
            report = AnalysisReport(**report_dict)
        except (TypeError, KeyError, ValueError) as exc:
            raise AgentExecutionError(
                f"[MultiAgentGraph] Failed to deserialise AnalysisReport: {exc}"
            ) from exc
        logger.info(
            "MultiAgentGraph.run() completed",
            extra={
                "run_id": self.run_id,
                "confidence": report.confidence,
                "insights": len(report.key_insights),
            },
        )
        return report

    async def arun(self, query: str) -> AnalysisReport:
        """
        Execute the full pipeline asynchronously.

        The underlying LangGraph calls are blocking; this method runs them in a
        ``ThreadPoolExecutor`` to avoid stalling an ``asyncio`` event loop.

        Args:
            query: The research question or topic to investigate.

        Returns:
            A populated ``AnalysisReport`` dataclass.

        Raises:
            AgentValidationError: When the query is empty.
            AgentExecutionError: When any pipeline stage fails.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.run, query)

    def close(self) -> None:
        """Shut down the thread pool executor."""
        self._executor.shutdown(wait=False)

    def get_research_result(self, query: str) -> ResearchResult:
        """
        Run only the research phase and return the raw ``ResearchResult``.

        Useful for ad-hoc research without triggering the full analysis
        pipeline.

        Args:
            query: The research question or topic.

        Returns:
            A populated ``ResearchResult`` dataclass.

        Raises:
            AgentValidationError: When the query is empty.
            AgentExecutionError: On research agent failure.
        """
        if not query or not query.strip():
            raise AgentValidationError(
                "MultiAgentGraph.get_research_result() requires a non-empty query."
            )

        agent = ResearchAgent(
            thread_id=f"{self.run_id}-research-only",
            llm=self._llm,
            checkpointer=self._checkpointer,
        )
        return agent.run_structured(query.strip())
