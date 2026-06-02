"""
domain_packs/analysis_only/pack.py — Analysis-only domain pack.

Runs AnalystAgent on pre-supplied research material (findings + summary).
Symmetric counterpart to ``ResearchOnlyPack``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from agents.analyst import AnalysisReport, AnalystAgent
from agents.base_agent import (
    AgentBudgetExceededError,
    AgentExecutionError,
    AgentTimeoutError,
    AgentValidationError,
)
from agents.researcher import ResearchResult
from core.config import get_settings
from core.memory import create_checkpointer
from core.observability import trace_span
from domain_packs.research.analysis_only.schemas import (
    AnalysisOnlyInput,
    AnalysisOnlyOutput,
)
from pack_kernel.base_pack import (
    BaseDomainPack,
    normalize_pack_stream_event,
    pack_stream_event,
)

logger = logging.getLogger(__name__)


class AnalysisOnlyState(TypedDict, total=False):
    research_result: dict[str, Any] | None
    analysis_report: dict[str, Any] | None
    error: str | None
    status: str


class AnalysisOnlyPack(BaseDomainPack):
    """Domain pack that runs AnalystAgent on supplied research context."""

    pack_id = "analysis_only"
    name = "Analysis Only"
    description = (
        "Runs AnalystAgent on pre-collected research findings and summary "
        "without a live research phase."
    )
    input_schema = AnalysisOnlyInput
    output_schema = AnalysisOnlyOutput

    def __init__(
        self,
        run_id: str | None = None,
        llm: Any | None = None,
        checkpointer: Any | None = None,
        budget_usd: float | None = None,
    ) -> None:
        super().__init__(
            run_id=run_id, llm=llm, checkpointer=checkpointer, budget_usd=budget_usd
        )
        self.run_id = run_id or str(uuid.uuid4())
        self._checkpointer = checkpointer or create_checkpointer(get_settings())
        self._executor: ThreadPoolExecutor | None = None
        self._executor_lock = threading.Lock()
        self._analyst_agent: AnalystAgent | None = None
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(AnalysisOnlyState)
        graph.add_node("analysis_node", self._analysis_node)
        graph.set_entry_point("analysis_node")
        graph.add_edge("analysis_node", END)
        return graph.compile(checkpointer=self._checkpointer)

    def _analysis_node(self, state: AnalysisOnlyState) -> AnalysisOnlyState:
        research_dict = state.get("research_result")
        if not research_dict:
            return {
                **state,
                "status": "error",
                "error": "analysis_node received no research_result",
            }  # type: ignore[return-value]

        with trace_span("analysis_only_node", {"run_id": self.run_id}):
            try:
                research_result = ResearchResult(**research_dict)
                if self._analyst_agent is None:
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
                        budget_usd=self._budget_usd,
                    )
                analyst_agent = self._analyst_agent
                if analyst_agent is None:
                    raise AgentExecutionError("Analyst agent failed to initialize")
                report: AnalysisReport = analyst_agent.run_structured(research_result)
                return {
                    **state,
                    "analysis_report": report.to_dict(),
                    "status": "done",
                    "error": None,
                }  # type: ignore[return-value]
            except (
                AgentBudgetExceededError,
                AgentExecutionError,
                AgentTimeoutError,
                AgentValidationError,
            ) as exc:
                return {
                    **state,
                    "analysis_report": None,
                    "status": "error",
                    "error": str(exc),
                }  # type: ignore[return-value]

    @property
    def cost_usd(self) -> float:
        if self._analyst_agent is not None:
            return self._analyst_agent.cost_usd
        return 0.0

    @staticmethod
    def _input_to_research(inp: AnalysisOnlyInput) -> ResearchResult:
        return ResearchResult(
            query=inp.query,
            findings=list(inp.findings),
            summary=inp.summary,
            sources=list(inp.sources),
            confidence=inp.confidence,
            metadata={"pack": "analysis_only"},
        )

    def run_from_input(self, body: BaseModel) -> AnalysisReport:
        if not isinstance(body, AnalysisOnlyInput):
            body = AnalysisOnlyInput.model_validate(body)
        return self._run_research(self._input_to_research(body))

    def run(self, query: str) -> AnalysisReport:
        if not query or not query.strip():
            raise AgentValidationError(
                "AnalysisOnlyPack.run() requires a non-empty query."
            )
        inp = AnalysisOnlyInput(query=query.strip(), summary=query.strip())
        return self.run_from_input(inp)

    def _run_research(self, research: ResearchResult) -> AnalysisReport:
        initial: AnalysisOnlyState = {
            "research_result": research.to_dict(),
            "analysis_report": None,
            "error": None,
            "status": "running",
        }
        config = {"configurable": {"thread_id": self.run_id}}
        try:
            final = self._graph.invoke(initial, config=config)
        except Exception as exc:
            raise AgentExecutionError(
                f"[AnalysisOnlyPack] Pipeline execution failed: {exc}"
            ) from exc

        if final.get("status") == "error":
            raise AgentExecutionError(
                f"[AnalysisOnlyPack] Pipeline error: {final.get('error')}"
            )
        report_dict = final.get("analysis_report")
        if not report_dict:
            raise AgentExecutionError(
                "[AnalysisOnlyPack] completed without AnalysisReport."
            )
        return AnalysisReport(**report_dict)

    async def arun(self, query: str) -> AnalysisReport:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._get_executor(), self.run, query)

    async def stream_events_from_input(
        self, body: BaseModel
    ) -> AsyncIterator[dict[str, Any]]:
        async for raw in self._iter_stream_events_from_input(body):
            yield normalize_pack_stream_event(raw)

    async def _iter_stream_events_from_input(
        self, body: BaseModel
    ) -> AsyncIterator[dict[str, Any]]:
        if not isinstance(body, AnalysisOnlyInput):
            body = AnalysisOnlyInput.model_validate(body)
        research = self._input_to_research(body)
        initial: AnalysisOnlyState = {
            "research_result": research.to_dict(),
            "analysis_report": None,
            "error": None,
            "status": "running",
        }
        config = {"configurable": {"thread_id": self.run_id}}
        final_report: AnalysisReport | None = None

        async for event in self._graph.astream_events(
            initial, config=config, version="v2"
        ):
            kind = event.get("event", "")
            name = event.get("name", "")
            if kind == "on_chain_start" and name == "analysis_node":
                yield pack_stream_event("phase_started", phase="analysis")
            elif kind == "on_chain_end" and name == "analysis_node":
                yield pack_stream_event("phase_completed", phase="analysis")
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    rd = output.get("analysis_report")
                    if rd:
                        try:
                            final_report = AnalysisReport(**rd)
                        except (TypeError, KeyError, ValueError):
                            pass

        if final_report is None:
            raise AgentExecutionError(
                "[AnalysisOnlyPack] Stream completed without AnalysisReport."
            )
        yield pack_stream_event("pipeline_completed", report=final_report)

    async def _iter_stream_events(self, query: str) -> AsyncIterator[dict[str, Any]]:
        inp = AnalysisOnlyInput(query=query, summary=query)
        async for event in self._iter_stream_events_from_input(inp):
            yield event

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            with self._executor_lock:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=get_settings().thread_pool_max_workers,
                        thread_name_prefix="analysis-only-pack",
                    )
        return self._executor
