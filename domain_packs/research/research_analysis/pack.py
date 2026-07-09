"""
domain_packs/research_analysis/pack.py — Research + Analysis domain pack.

Migrated from core/graph.py.  The pipeline sequences ResearchAgent → AnalystAgent
inside a LangGraph StateGraph and exposes the standard BaseDomainPack interface:
run(), arun(), and stream_events().

core/graph.py keeps a backward-compat alias:
    MultiAgentGraph = ResearchAnalysisPack
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from agents.analyst import AnalysisReport, AnalystAgent
from agents.base_agent import (
    AgentAuthenticationError,
    AgentBudgetExceededError,
    AgentExecutionError,
    AgentTimeoutError,
    AgentValidationError,
)
from agents.researcher import ResearchAgent, ResearchResult
from connectors.base import (
    BaseConnector,
    ConnectorRequest,
    ConnectorResult,
    record_to_source_ref,
)
from core.config import get_settings
from core.cost import CostTracker
from core.memory import create_checkpointer
from core.observability import trace_span
from domain_packs.research.research_analysis.schemas import (
    ResearchAnalysisInput,
    ResearchAnalysisOutput,
)
from pack_kernel.base_pack import BaseDomainPack, pack_stream_event

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orchestrator state schema
# ---------------------------------------------------------------------------


class OrchestratorState(TypedDict, total=False):
    """State shared across the Research+Analysis pipeline nodes."""

    query: str
    research_result: dict[str, Any] | None
    analysis_report: dict[str, Any] | None
    error: str | None
    status: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# ResearchAnalysisPack
# ---------------------------------------------------------------------------


class ResearchAnalysisPack(BaseDomainPack):
    """
    Domain pack that sequences ResearchAgent → AnalystAgent.

    This is the first-party pack that ships with the platform kernel.
    It is wire-compatible with the former MultiAgentGraph class — the
    constructor signature and all public methods are identical.

    Usage (direct)::

        with ResearchAnalysisPack(run_id="abc") as pack:
            report = pack.run("What are trends in vector databases?")

    Usage (via registry)::

        from pack_kernel.registry import PackRegistry
        from pack_kernel.builtin_packs import register_builtin_packs

        register_builtin_packs()

        Pack = PackRegistry.get("research_analysis")
        with Pack(run_id="abc", llm=llm, checkpointer=cp) as pack:
            report = pack.run(query)
    """

    pack_id = "research_analysis"
    name = "Research + Analysis"
    description = (
        "Sequences a ResearchAgent and an AnalystAgent to turn a free-text query "
        "into a structured AnalysisReport."
    )
    input_schema = ResearchAnalysisInput
    output_schema = ResearchAnalysisOutput
    executor_thread_name_prefix = "agent-graph"

    def __init__(
        self,
        run_id: str | None = None,
        llm: Any | None = None,
        checkpointer: Any | None = None,
        budget_usd: float | None = None,
        connector: BaseConnector | None = None,
    ) -> None:
        """Initialise the Research + Analysis pipeline.

        Args:
            run_id:       Optional stable identifier for this pipeline run.
                          Defaults to a new UUID4.
            llm:          Optional pre-built LangChain chat model.
            checkpointer: Optional pre-built LangGraph checkpointer.
            budget_usd:   USD cost ceiling for the *whole pipeline run*.  A single
                          CostTracker is shared by every agent in the run, so the
                          limit applies to the cumulative spend of all agents and
                          ``cost_usd`` exposes the cumulative run cost.
                          Set to ``None`` to disable budget enforcement (default).
            connector:    Optional retrieval adapter; when set, records from
                          ``connector.fetch()`` are merged into the research phase output.
        """
        super().__init__(
            run_id=run_id, llm=llm, checkpointer=checkpointer, budget_usd=budget_usd
        )
        self.run_id = run_id or str(uuid.uuid4())
        self._checkpointer = checkpointer or create_checkpointer(get_settings())
        self._connector = connector
        # ONE tracker for the whole run, shared by every agent, so budget_usd
        # caps the cumulative spend of the pipeline (not per-agent).  The same
        # budget resolution as BaseAgent applies: explicit argument first,
        # then the settings-level default.  When neither is set, no tracker is
        # created and agents keep their legacy untracked behaviour.
        _effective_budget: float | None = (
            budget_usd
            if budget_usd is not None
            else get_settings().pack_default_budget_usd
        )
        self._cost_tracker: CostTracker | None = (
            CostTracker(budget_usd=_effective_budget)
            if _effective_budget is not None
            else None
        )
        self._research_agent: ResearchAgent | None = None
        self._analyst_agent: AnalystAgent | None = None
        self._graph = self._build_graph()

        logger.info(
            "ResearchAnalysisPack initialised",
            extra={"run_id": self.run_id},
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
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
    # Optional connector retrieval
    # ------------------------------------------------------------------

    def _fetch_connector_result(self, query: str) -> ConnectorResult:
        """Run the optional connector (sync graph node — uses asyncio.run when idle)."""
        if self._connector is None:
            return ConnectorResult()

        request = ConnectorRequest(query=query)

        async def _fetch() -> ConnectorResult:
            return await self._connector.fetch(request)  # type: ignore[union-attr]

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_fetch())

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _fetch())
            return future.result()

    def _merge_connector_into_result(
        self, result: ResearchResult, connector_result: ConnectorResult
    ) -> ResearchResult:
        """Append connector snippets to findings and record provenance in metadata."""
        if not connector_result.records:
            return result

        extra_findings: list[str] = []
        extra_sources: list[str] = []
        source_refs: list[dict[str, Any]] = []
        for index, record in enumerate(connector_result.records, start=1):
            ref = record_to_source_ref(record, index)
            source_refs.append(ref.model_dump())
            snippet = record.get("snippet") or record.get("text")
            if snippet:
                extra_findings.append(str(snippet))
            extra_sources.append(ref.citation())

        metadata = dict(result.metadata)
        metadata["connector"] = {
            "connector_id": getattr(self._connector, "connector_id", ""),
            "record_count": len(connector_result.records),
            "fetch_metadata": connector_result.metadata,
        }
        metadata["source_refs"] = [
            *metadata.get("source_refs", []),
            *source_refs,
        ]

        return ResearchResult(
            query=result.query,
            findings=[*result.findings, *extra_findings],
            summary=result.summary,
            sources=[*result.sources, *extra_sources],
            confidence=result.confidence,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _research_node(self, state: OrchestratorState) -> OrchestratorState:
        query: str = state.get("query", "")
        logger.info(
            "Pipeline: starting research phase",
            extra={"run_id": self.run_id, "query": query[:120]},
        )

        with trace_span("research_node", {"run_id": self.run_id}):
            try:
                if self._research_agent is None:
                    # Resolve ResearchAgent via sys.modules so that
                    # patch("core.graph.ResearchAgent", …) in tests is honoured.
                    import sys as _sys

                    _core_graph = _sys.modules.get("core.graph")
                    _agent_cls = (
                        getattr(_core_graph, "ResearchAgent", None)
                        if _core_graph is not None
                        else None
                    ) or ResearchAgent
                    self._research_agent = _agent_cls(
                        thread_id=f"{self.run_id}-research",
                        llm=self._llm,
                        checkpointer=self._checkpointer,
                        cost_tracker=self._cost_tracker,
                    )
                research_agent = self._research_agent
                if research_agent is None:
                    raise AgentExecutionError("Research agent failed to initialize")
                result: ResearchResult = research_agent.run_structured(query)
                if self._connector is not None:
                    connector_result = self._fetch_connector_result(query)
                    result = self._merge_connector_into_result(result, connector_result)
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

            except AgentAuthenticationError:
                raise
            except (
                AgentBudgetExceededError,
                AgentExecutionError,
                AgentTimeoutError,
                AgentValidationError,
            ) as exc:
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
        logger.info(
            "Pipeline: starting analysis phase",
            extra={"run_id": self.run_id},
        )

        research_dict = state.get("research_result")
        if not research_dict:
            err = "analysis_node received no research_result"
            logger.error("Pipeline: analysis phase skipped", extra={"reason": err})
            return {**state, "status": "error", "error": err}  # type: ignore[return-value]

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
                    # Resolve AnalystAgent via sys.modules so that
                    # patch("core.graph.AnalystAgent", …) in tests is honoured.
                    import sys as _sys

                    _core_graph = _sys.modules.get("core.graph")
                    _agent_cls = (
                        getattr(_core_graph, "AnalystAgent", None)
                        if _core_graph is not None
                        else None
                    ) or AnalystAgent
                    self._analyst_agent = _agent_cls(
                        thread_id=f"{self.run_id}-analysis",
                        llm=self._llm,
                        checkpointer=self._checkpointer,
                        cost_tracker=self._cost_tracker,
                    )
                analyst_agent = self._analyst_agent
                if analyst_agent is None:
                    raise AgentExecutionError("Analyst agent failed to initialize")
                report: AnalysisReport = analyst_agent.run_structured(research_result)
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

            except AgentAuthenticationError:
                raise
            except (
                AgentBudgetExceededError,
                AgentExecutionError,
                AgentTimeoutError,
                AgentValidationError,
            ) as exc:
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
    # Cost tracking
    # ------------------------------------------------------------------

    @property
    def cost_usd(self) -> float:
        """Total USD cost for this run across all agents (shared tracker)."""
        if self._cost_tracker is not None:
            return self._cost_tracker.total_cost_usd
        return 0.0

    # ------------------------------------------------------------------
    # Conditional routing
    # ------------------------------------------------------------------

    def _route_after_research(
        self, state: OrchestratorState
    ) -> Literal["analysis_node", "end_on_error"]:
        if state.get("status") == "error" or state.get("error"):
            logger.warning(
                "Pipeline: routing to end due to research error",
                extra={"error": state.get("error", "")},
            )
            return "end_on_error"
        return "analysis_node"

    # ------------------------------------------------------------------
    # BaseDomainPack interface
    # ------------------------------------------------------------------

    def run(self, query: str) -> AnalysisReport:
        """Execute the full Research → Analysis pipeline synchronously."""
        if not query or not query.strip():
            raise AgentValidationError(
                "ResearchAnalysisPack.run() requires a non-empty query."
            )

        logger.info(
            "ResearchAnalysisPack.run() started",
            extra={"run_id": self.run_id, "query": query[:120]},
        )

        from core.mock_llm import reset_mock_research_sequence

        reset_mock_research_sequence(start=0)

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
        except AgentAuthenticationError:
            raise
        except Exception as exc:
            raise AgentExecutionError(
                f"[ResearchAnalysisPack] Pipeline execution failed: {exc}"
            ) from exc

        if final_state.get("status") == "error":
            raise AgentExecutionError(
                f"[ResearchAnalysisPack] Pipeline error: {final_state.get('error')}"
            )

        report_dict = final_state.get("analysis_report")
        if not report_dict:
            raise AgentExecutionError(
                "[ResearchAnalysisPack] Pipeline completed without an AnalysisReport."
            )

        try:
            report = AnalysisReport(**report_dict)
        except (TypeError, KeyError, ValueError) as exc:
            raise AgentExecutionError(
                f"[ResearchAnalysisPack] Failed to deserialise AnalysisReport: {exc}"
            ) from exc

        logger.info(
            "ResearchAnalysisPack.run() completed",
            extra={
                "run_id": self.run_id,
                "confidence": report.confidence,
                "insights": len(report.key_insights),
            },
        )
        return report

    async def arun(self, query: str) -> AnalysisReport:
        """Execute the full pipeline asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._get_executor(), self.run, query)

    async def _iter_stream_events(self, query: str) -> AsyncIterator[dict[str, Any]]:
        """Stream pipeline execution events in real time."""
        if not query or not query.strip():
            raise AgentValidationError(
                "ResearchAnalysisPack.stream_events() requires a non-empty query."
            )

        from core.mock_llm import reset_mock_research_sequence

        reset_mock_research_sequence(start=0)

        initial_state: OrchestratorState = {
            "query": query.strip(),
            "research_result": None,
            "analysis_report": None,
            "error": None,
            "status": "running",
            "metadata": {"run_id": self.run_id},
        }
        config = {"configurable": {"thread_id": self.run_id}}

        final_report: AnalysisReport | None = None

        async for event in self._graph.astream_events(
            initial_state, config=config, version="v2"
        ):
            kind = event.get("event", "")
            name = event.get("name", "")

            if kind == "on_chain_start" and name in ("research_node", "analysis_node"):
                phase = "research" if name == "research_node" else "analysis"
                yield pack_stream_event("phase_started", phase=phase)

            elif kind == "on_chain_end" and name in ("research_node", "analysis_node"):
                phase = "research" if name == "research_node" else "analysis"
                yield pack_stream_event("phase_completed", phase=phase)

                if name == "analysis_node":
                    output = event.get("data", {}).get("output", {})
                    report_dict = None
                    if isinstance(output, dict):
                        report_dict = output.get("analysis_report")
                    if report_dict:
                        try:
                            final_report = AnalysisReport(**report_dict)
                        except (TypeError, KeyError, ValueError):
                            pass

            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield pack_stream_event(
                        "token",
                        content=chunk.content,
                        node=event.get("metadata", {}).get("langgraph_node", ""),
                    )

        if final_report is None:
            raise AgentExecutionError(
                "[ResearchAnalysisPack] Stream completed without an AnalysisReport."
            )

        yield pack_stream_event("pipeline_completed", report=final_report)

    # ------------------------------------------------------------------
    # Additional method kept for API backward-compat
    # ------------------------------------------------------------------

    def get_research_result(self, query: str) -> ResearchResult:
        """Run only the research phase and return the raw ResearchResult."""
        if not query or not query.strip():
            raise AgentValidationError(
                "ResearchAnalysisPack.get_research_result() requires a non-empty query."
            )
        import sys as _sys

        _core_graph = _sys.modules.get("core.graph")
        _agent_cls = (
            getattr(_core_graph, "ResearchAgent", None)
            if _core_graph is not None
            else None
        ) or ResearchAgent
        agent = _agent_cls(
            thread_id=f"{self.run_id}-research-only",
            llm=self._llm,
            checkpointer=self._checkpointer,
            cost_tracker=self._cost_tracker,
        )
        return agent.run_structured(query.strip())
