"""
agents/analyst.py — AnalystAgent: synthesise research into actionable reports.

The ``AnalystAgent`` consumes the output of ``ResearchAgent`` (a
``ResearchResult``) and drives a three-node LangGraph pipeline:

1. ``analyze``    — deep-dive into the research findings, extract insights.
2. ``synthesize`` — connect insights, identify patterns and implications.
3. ``report``     — produce a structured ``AnalysisReport``.

Collaboration pattern
---------------------
``AnalystAgent`` is designed to operate after ``ResearchAgent`` in the
``MultiAgentGraph`` pipeline.  The two agents share no in-process state;
they communicate through the serialised ``ResearchResult`` dict that the
orchestrator passes as part of the input payload.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agents.base_agent import (
    AgentExecutionError,
    AgentState,
    AgentValidationError,
    BaseAgent,
    _input_validator,
)
from agents.models import (
    AnalysisReport,  # backward-compat re-export
    ResearchResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AnalystAgent
# ---------------------------------------------------------------------------


class AnalystAgent(BaseAgent):
    """
    LangGraph agent that analyses research findings and produces a report.

    Graph topology::

        analyze → synthesize → report → END

    The agent expects the ``ResearchResult`` to be embedded in the input
    JSON string (see ``run()`` and ``run_structured()``).

    Args:
        thread_id: Optional stable ID for resuming a checkpointed session.
    """

    _CTX_RESEARCH = "research_result"
    _CTX_INSIGHTS = "raw_insights"
    _CTX_PATTERNS = "raw_patterns"
    _CTX_REPORT = "analysis_report"

    def __init__(
        self,
        thread_id: str | None = None,
        llm: BaseChatModel | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        super().__init__(
            name="AnalystAgent",
            thread_id=thread_id,
            llm=llm,
            checkpointer=checkpointer,
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(self) -> Any:
        """
        Construct and compile the analysis StateGraph.

        Returns:
            Compiled LangGraph graph ready for ``.invoke()`` / ``.stream()``.
        """
        graph = StateGraph(AgentState)

        graph.add_node("analyze", self._node_analyze)
        graph.add_node("synthesize", self._node_synthesize)
        graph.add_node("report", self._node_report)

        graph.set_entry_point("analyze")
        graph.add_edge("analyze", "synthesize")
        graph.add_edge("synthesize", "report")
        graph.add_edge("report", END)

        return graph.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _node_analyze(self, state: AgentState) -> dict[str, Any]:
        """
        Node: extract key insights from the research findings.

        Reads the ``ResearchResult`` from ``state["context"][_CTX_RESEARCH]``
        and writes a list of insight strings to
        ``state["context"][_CTX_INSIGHTS]``.
        """
        state = self._increment_step(state)
        self._log_step("analyze", state)

        research: dict[str, Any] = state.get("context", {}).get(self._CTX_RESEARCH, {})
        query: str = research.get("query", state["messages"][0].content)  # type: ignore[union-attr]
        summary: str = research.get("summary", "")
        findings: list[str] = research.get("findings", [])

        analysis_prompt = (
            "You are a senior data analyst. Perform a rigorous analysis.\n\n"
            f"Research query: {query}\n\n"
            f"Research summary:\n{summary}\n\n"
            f"Raw findings ({len(findings)} total, showing first 8):\n"
            + "\n".join(f"[{i+1}] {f[:300]}" for i, f in enumerate(findings[:8]))
            + "\n\nExtract the KEY INSIGHTS. Return JSON: "
            '{"insights": ["...", ...], "confidence": 0.0-1.0}'
        )

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content="You are a rigorous analytical thinker."),
                    HumanMessage(content=analysis_prompt),
                ]
            )
            parsed = json.loads(response.content)  # type: ignore[arg-type]
            insights: list[str] = parsed.get("insights", [])
            confidence: float = float(parsed.get("confidence", 0.7))
        except Exception:
            insights = ["Analysis completed — structured extraction failed."]
            confidence = 0.5

        self._log.info(
            "Analyze node completed",
            extra={"insights_count": len(insights), "confidence": confidence},
        )

        return {
            "step_count": state["step_count"],
            "context": {
                **state.get("context", {}),
                self._CTX_INSIGHTS: insights,
                "analyze_confidence": confidence,
            },
        }

    def _node_synthesize(self, state: AgentState) -> dict[str, Any]:
        """
        Node: connect insights and identify cross-cutting patterns.

        Reads ``state["context"][_CTX_INSIGHTS]`` and writes pattern strings
        to ``state["context"][_CTX_PATTERNS]``.
        """
        state = self._increment_step(state)
        self._log_step("synthesize", state)

        insights: list[str] = state.get("context", {}).get(self._CTX_INSIGHTS, [])
        research: dict[str, Any] = state.get("context", {}).get(self._CTX_RESEARCH, {})
        query: str = research.get("query", "")

        synthesis_prompt = (
            "You are a strategic synthesiser. Your task is to identify overarching "
            "patterns and structural themes across a set of insights.\n\n"
            f"Research topic: {query}\n\n"
            "Insights to synthesise:\n"
            + "\n".join(f"- {ins}" for ins in insights)
            + "\n\nIdentify PATTERNS and IMPLICATIONS. Return JSON:\n"
            '{"patterns": ["...", ...], "implications": ["...", ...]}'
        )

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content="You are a strategic pattern synthesiser."),
                    HumanMessage(content=synthesis_prompt),
                ]
            )
            parsed = json.loads(response.content)  # type: ignore[arg-type]
            patterns: list[str] = parsed.get("patterns", [])
            implications: list[str] = parsed.get("implications", [])
        except Exception:
            patterns = ["Pattern synthesis unavailable."]
            implications = ["Implication extraction unavailable."]

        self._log.info(
            "Synthesize node completed",
            extra={
                "patterns_count": len(patterns),
                "implications_count": len(implications),
            },
        )

        return {
            "step_count": state["step_count"],
            "context": {
                **state.get("context", {}),
                self._CTX_PATTERNS: patterns,
                "implications": implications,
            },
        }

    def _node_report(self, state: AgentState) -> dict[str, Any]:
        """
        Node: compile all intermediate results into a final ``AnalysisReport``.

        Serialises the report dict into ``state["context"][_CTX_REPORT]`` and
        appends the Markdown rendering as an ``AIMessage`` to
        ``state["messages"]``.
        """
        state = self._increment_step(state)
        self._log_step("report", state)

        ctx = state.get("context", {})
        research: dict[str, Any] = ctx.get(self._CTX_RESEARCH, {})
        query: str = research.get("query", "")
        research_summary: str = research.get("summary", "")
        insights: list[str] = ctx.get(self._CTX_INSIGHTS, [])
        patterns: list[str] = ctx.get(self._CTX_PATTERNS, [])
        implications: list[str] = ctx.get("implications", [])
        analyze_confidence: float = ctx.get("analyze_confidence", 0.7)

        exec_summary_prompt = (
            "You are a C-suite report writer. Write a concise executive summary "
            "(2-3 sentences) based on the following analysis.\n\n"
            f"Topic: {query}\n"
            f"Research baseline: {research_summary[:500]}\n"
            f"Key insights: {'; '.join(insights[:5])}\n"
            f"Patterns: {'; '.join(patterns[:3])}\n"
            "Return ONLY the executive summary paragraph — no JSON, no headers."
        )

        try:
            exec_response = self.llm.invoke(
                [
                    SystemMessage(content="You are a precise executive report writer."),
                    HumanMessage(content=exec_summary_prompt),
                ]
            )
            exec_summary: str = str(exec_response.content).strip()
        except Exception:
            exec_summary = (
                f"Analysis of '{query}' completed with "
                f"{len(insights)} insights identified."
            )

        report = AnalysisReport(
            query=query,
            executive_summary=exec_summary,
            key_insights=insights,
            patterns=patterns,
            implications=implications,
            confidence=min(1.0, max(0.0, analyze_confidence)),
            research_summary=research_summary,
            metadata=state.get("metadata", {}),
        )

        report_markdown = report.to_markdown()

        self._log.info(
            "Report node completed",
            extra={
                "confidence": report.confidence,
                "insights_count": len(report.key_insights),
            },
        )

        updated_messages = list(state.get("messages", []))
        updated_messages.append(AIMessage(content=report_markdown))

        return {
            "step_count": state["step_count"],
            "messages": updated_messages,
            "context": {
                **ctx,
                self._CTX_REPORT: report.to_dict(),
            },
            "status": "done",
        }

    # ------------------------------------------------------------------
    # Public run interface
    # ------------------------------------------------------------------

    def _execute(self, query: str, research_dict: dict[str, Any]) -> AgentState:
        """Execute the analysis graph and return the final state."""
        if not query or not query.strip():
            raise AgentValidationError(f"[{self.name}] Query must not be empty.")
        query = _input_validator.validate(query)
        initial_state = self._make_initial_state(query)
        initial_state["context"][self._CTX_RESEARCH] = research_dict
        self._log.info(
            "Starting analysis run",
            extra={"query": research_dict.get("query", "")[:120]},
        )
        try:
            final_state: AgentState = self._graph.invoke(
                initial_state, config=self._get_config()
            )
        except Exception as exc:
            raise AgentExecutionError(
                f"[{self.name}] Analysis pipeline failed: {exc}"
            ) from exc
        return final_state

    def run(self, query: str) -> str:
        """
        Execute the analysis pipeline.

        ``query`` is expected to be either a plain question string or a
        JSON-encoded ``ResearchResult`` dict.  When a plain string is
        supplied the agent will analyse it directly without research context.

        Args:
            query: Research summary / topic string, or JSON ``ResearchResult``.

        Returns:
            Markdown-formatted analysis report as a string.

        Raises:
            AgentExecutionError: On unrecoverable graph errors.
        """
        research_dict: dict[str, Any] = {}
        try:
            candidate = json.loads(query)
            if isinstance(candidate, dict) and "query" in candidate:
                research_dict = candidate
        except (json.JSONDecodeError, ValueError):
            research_dict = {
                "query": query,
                "summary": query,
                "findings": [],
                "sources": [],
                "confidence": 0.5,
                "metadata": {},
            }

        final_state = self._execute(query, research_dict)

        for msg in reversed(final_state.get("messages", [])):
            if isinstance(msg, AIMessage):
                return str(msg.content)

        return "Analysis completed but no output was produced."

    def run_structured(self, research_result: ResearchResult) -> AnalysisReport:
        """
        Execute the analysis pipeline with a typed ``ResearchResult`` input.

        This is the preferred method when collaborating with ``ResearchAgent``
        in the ``MultiAgentGraph`` orchestrator.

        Args:
            research_result: Structured output from ``ResearchAgent``.

        Returns:
            A populated ``AnalysisReport`` dataclass.

        Raises:
            AgentExecutionError: On unrecoverable graph errors.
            AgentValidationError: When no structured report is produced.
        """
        final_state = self._execute(research_result.query, research_result.to_dict())

        report_dict = final_state.get("context", {}).get(self._CTX_REPORT)
        if not report_dict:
            raise AgentValidationError(
                "AnalystAgent: graph completed without producing an AnalysisReport."
            )

        return AnalysisReport(**report_dict)
