"""
domain_packs/research_only/pack.py — Research-only domain pack.

Runs a single ResearchAgent inside a minimal LangGraph and returns a
``ResearchResult``. Demonstrates registering a second pack_id via PackRegistry.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from agents.base_agent import (
    AgentBudgetExceededError,
    AgentExecutionError,
    AgentTimeoutError,
    AgentValidationError,
)
from agents.researcher import ResearchAgent, ResearchResult
from core.config import get_settings
from core.memory import create_checkpointer
from core.observability import trace_span
from domain_packs.research.research_only.schemas import (
    ResearchOnlyInput,
    ResearchOnlyOutput,
)
from pack_kernel.base_pack import BaseDomainPack, pack_stream_event

logger = logging.getLogger(__name__)


class ResearchOnlyState(TypedDict, total=False):
    """State for the research-only pipeline."""

    query: str
    research_result: dict[str, Any] | None
    error: str | None
    status: str


class ResearchOnlyPack(BaseDomainPack):
    """Domain pack that runs ResearchAgent only (no analysis phase)."""

    pack_id = "research_only"
    name = "Research Only"
    description = (
        "Runs a single ResearchAgent to collect findings and a summary "
        "without an analysis phase."
    )
    input_schema = ResearchOnlyInput
    output_schema = ResearchOnlyOutput
    executor_thread_name_prefix = "research-only-pack"

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
        self._research_agent: ResearchAgent | None = None
        self._graph = self._build_graph()

        logger.info(
            "ResearchOnlyPack initialised",
            extra={"run_id": self.run_id},
        )

    def _build_graph(self) -> Any:
        graph = StateGraph(ResearchOnlyState)
        graph.add_node("research_node", self._research_node)
        graph.set_entry_point("research_node")
        graph.add_edge("research_node", END)
        return graph.compile(checkpointer=self._checkpointer)

    def _research_node(self, state: ResearchOnlyState) -> ResearchOnlyState:
        query: str = state.get("query", "")
        logger.info(
            "ResearchOnlyPack: starting research",
            extra={"run_id": self.run_id, "query": query[:120]},
        )

        with trace_span("research_only_node", {"run_id": self.run_id}):
            try:
                if self._research_agent is None:
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
                        budget_usd=self._budget_usd,
                    )
                research_agent = self._research_agent
                if research_agent is None:
                    raise AgentExecutionError("Research agent failed to initialize")
                result: ResearchResult = research_agent.run_structured(query)
                return {
                    **state,
                    "research_result": result.to_dict(),
                    "status": "done",
                    "error": None,
                }  # type: ignore[return-value]

            except (
                AgentBudgetExceededError,
                AgentExecutionError,
                AgentTimeoutError,
                AgentValidationError,
            ) as exc:
                logger.error(
                    "ResearchOnlyPack: research failed",
                    extra={"run_id": self.run_id, "error": str(exc)},
                )
                return {
                    **state,
                    "research_result": None,
                    "status": "error",
                    "error": str(exc),
                }  # type: ignore[return-value]

    @property
    def cost_usd(self) -> float:
        if self._research_agent is not None:
            return self._research_agent.cost_usd
        return 0.0

    def run(self, query: str) -> ResearchResult:
        if not query or not query.strip():
            raise AgentValidationError(
                "ResearchOnlyPack.run() requires a non-empty query."
            )

        from core.mock_llm import reset_mock_research_sequence

        reset_mock_research_sequence(start=0)

        initial_state: ResearchOnlyState = {
            "query": query.strip(),
            "research_result": None,
            "error": None,
            "status": "running",
        }
        config = {"configurable": {"thread_id": self.run_id}}

        try:
            final_state: ResearchOnlyState = self._graph.invoke(
                initial_state, config=config
            )
        except Exception as exc:
            raise AgentExecutionError(
                f"[ResearchOnlyPack] Pipeline execution failed: {exc}"
            ) from exc

        if final_state.get("status") == "error":
            raise AgentExecutionError(
                f"[ResearchOnlyPack] Pipeline error: {final_state.get('error')}"
            )

        result_dict = final_state.get("research_result")
        if not result_dict:
            raise AgentExecutionError(
                "[ResearchOnlyPack] Pipeline completed without a ResearchResult."
            )

        try:
            return ResearchResult(**result_dict)
        except (TypeError, KeyError, ValueError) as exc:
            raise AgentExecutionError(
                f"[ResearchOnlyPack] Failed to deserialise ResearchResult: {exc}"
            ) from exc

    async def arun(self, query: str) -> ResearchResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._get_executor(), self.run, query)

    async def _iter_stream_events(self, query: str) -> AsyncIterator[dict[str, Any]]:
        if not query or not query.strip():
            raise AgentValidationError(
                "ResearchOnlyPack.stream_events() requires a non-empty query."
            )

        initial_state: ResearchOnlyState = {
            "query": query.strip(),
            "research_result": None,
            "error": None,
            "status": "running",
        }
        config = {"configurable": {"thread_id": self.run_id}}

        final_result: ResearchResult | None = None

        async for event in self._graph.astream_events(
            initial_state, config=config, version="v2"
        ):
            kind = event.get("event", "")
            name = event.get("name", "")

            if kind == "on_chain_start" and name == "research_node":
                yield pack_stream_event("phase_started", phase="research")

            elif kind == "on_chain_end" and name == "research_node":
                yield pack_stream_event("phase_completed", phase="research")
                output = event.get("data", {}).get("output", {})
                result_dict = None
                if isinstance(output, dict):
                    result_dict = output.get("research_result")
                if result_dict:
                    try:
                        final_result = ResearchResult(**result_dict)
                    except (TypeError, KeyError, ValueError):
                        pass

        if final_result is None:
            raise AgentExecutionError(
                "[ResearchOnlyPack] Stream completed without a ResearchResult."
            )

        yield pack_stream_event("pipeline_completed", result=final_result)
