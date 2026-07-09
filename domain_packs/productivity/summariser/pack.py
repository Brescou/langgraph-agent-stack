"""
domain_packs/summariser/pack.py — Single-agent summariser domain pack.

Uses one LLM call (no LangGraph sub-agents) inside a minimal one-node graph for
consistency with other packs. Accepts structured ``SummaryInput`` via
``run_from_input`` for typed API routes.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from agents.base_agent import (
    AgentAuthenticationError,
    AgentBudgetExceededError,
    AgentExecutionError,
    AgentTimeoutError,
    AgentValidationError,
    make_auth_error,
)
from agents.llm_retry import is_auth_llm_error
from core.config import get_settings
from core.memory import create_checkpointer
from core.observability import trace_span
from domain_packs.productivity.summariser.schemas import SummaryInput, SummaryOutput
from pack_kernel.base_pack import (
    BaseDomainPack,
    normalize_pack_stream_event,
    pack_stream_event,
)

logger = logging.getLogger(__name__)


class SummariserState(TypedDict, total=False):
    text: str
    bullet_count: int
    output: dict[str, Any] | None
    error: str | None
    status: str


class SummariserPack(BaseDomainPack):
    """Summarise free text into bullet points via the configured LLM."""

    pack_id = "summariser"
    name = "Text Summariser"
    description = "Summarises text into a configurable number of bullet points."
    input_schema = SummaryInput
    output_schema = SummaryOutput

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
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(SummariserState)
        graph.add_node("summarise_node", self._summarise_node)
        graph.set_entry_point("summarise_node")
        graph.add_edge("summarise_node", END)
        return graph.compile(checkpointer=self._checkpointer)

    def _build_prompt(self, inp: SummaryInput) -> str:
        return (
            f"Summarise the following text into exactly {inp.bullet_count} concise "
            f"bullet points.\nReturn only the bullet points, one per line, each "
            f"starting with '- '.\n\nText:\n{inp.text}"
        )

    def _parse_bullets(self, raw: str, expected: int) -> list[str]:
        lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        bullets = [line.lstrip("- ").strip() for line in lines if line.startswith("-")]
        if not bullets:
            bullets = lines
        return bullets[:expected]

    def _summarise_node(self, state: SummariserState) -> SummariserState:
        text = state.get("text", "")
        bullet_count = int(state.get("bullet_count") or 3)
        inp = SummaryInput(text=text, bullet_count=bullet_count)

        with trace_span("summarise_node", {"run_id": self.run_id}):
            try:
                if self._llm is None:
                    raise AgentExecutionError(
                        "SummariserPack requires an LLM instance on the pack."
                    )
                prompt = self._build_prompt(inp)
                from core.mock_llm import mock_plain_bullets_context

                try:
                    with mock_plain_bullets_context(inp.bullet_count):
                        response = self._llm.invoke(prompt)
                except Exception as exc:
                    if is_auth_llm_error(exc):
                        raise make_auth_error(
                            self.__class__.__name__,
                            get_settings().llm_provider,
                            exc,
                        ) from exc
                    raise
                raw = (
                    response.content if hasattr(response, "content") else str(response)
                )
                bullets = self._parse_bullets(raw, inp.bullet_count)
                output = SummaryOutput(
                    original_length=len(inp.text),
                    bullets=bullets,
                )
                return {
                    **state,
                    "output": output.model_dump(),
                    "status": "done",
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
                return {
                    **state,
                    "output": None,
                    "status": "error",
                    "error": str(exc),
                }  # type: ignore[return-value]

    def run_from_input(self, body: BaseModel) -> SummaryOutput:
        if not isinstance(body, SummaryInput):
            body = SummaryInput.model_validate(body)
        return self._run_state(body.text, body.bullet_count)

    def run(self, query: str) -> SummaryOutput:
        if not query or not query.strip():
            raise AgentValidationError("SummariserPack.run() requires non-empty text.")
        return self._run_state(query.strip(), 3)

    def _run_state(self, text: str, bullet_count: int) -> SummaryOutput:
        initial: SummariserState = {
            "text": text,
            "bullet_count": bullet_count,
            "output": None,
            "error": None,
            "status": "running",
        }
        config = {"configurable": {"thread_id": self.run_id}}
        try:
            final = self._graph.invoke(initial, config=config)
        except AgentAuthenticationError:
            raise
        except Exception as exc:
            raise AgentExecutionError(
                f"[SummariserPack] Pipeline execution failed: {exc}"
            ) from exc

        if final.get("status") == "error":
            raise AgentExecutionError(
                f"[SummariserPack] Pipeline error: {final.get('error')}"
            )
        out = final.get("output")
        if not out:
            raise AgentExecutionError("[SummariserPack] completed without output.")
        return SummaryOutput(**out)

    async def arun(self, query: str) -> SummaryOutput:
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
        if not isinstance(body, SummaryInput):
            body = SummaryInput.model_validate(body)
        yield pack_stream_event("phase_started", phase="summarise")
        # run_from_input drives a sync graph.invoke; keep it off the event
        # loop thread so async checkpointers (AsyncSqliteSaver, ...) work.
        result = await asyncio.to_thread(self.run_from_input, body)
        yield pack_stream_event("phase_completed", phase="summarise")
        yield pack_stream_event("pipeline_completed", result=result)

    async def _iter_stream_events(self, query: str) -> AsyncIterator[dict[str, Any]]:
        inp = SummaryInput(text=query, bullet_count=3)
        async for event in self._iter_stream_events_from_input(inp):
            yield event
