"""
agents/researcher.py — ResearchAgent: collect and structure information.

The ``ResearchAgent`` drives a three-node LangGraph pipeline:

1. ``research``  — query expansion and (mock/real) web/document retrieval.
2. ``validate``  — quality check on retrieved snippets; re-queries if needed.
3. ``summarize`` — distil validated findings into a ``ResearchResult``.

The mock search tool is intentionally swappable: replace
``_mock_web_search`` with any ``langchain_community`` tool (Tavily, Bing, …)
without touching the graph structure.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agents.base_agent import (
    AgentExecutionError,
    AgentState,
    AgentValidationError,
    BaseAgent,
    _extract_text_content,
    input_validator,
)
from agents.models import ResearchResult  # backward-compat re-export
from core.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------


class ResearchAgent(BaseAgent):
    """
    LangGraph agent that researches a topic and returns structured findings.

    Graph topology::

        research → validate → summarize → END

    The ``validate`` node can route back to ``research`` for a second pass
    when retrieved content is deemed insufficient (up to
    ``settings.max_research_iterations`` rounds).

    Args:
        thread_id: Optional stable ID for resuming a checkpointed session.
    """

    _CTX_FINDINGS = "findings"
    _CTX_SOURCES = "sources"
    _CTX_ITERATIONS = "research_iterations"
    _CTX_RESULT = "research_result"

    def __init__(
        self,
        thread_id: str | None = None,
        llm: BaseChatModel | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        super().__init__(
            name="ResearchAgent",
            thread_id=thread_id,
            llm=llm,
            checkpointer=checkpointer,
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(self) -> Any:
        """
        Construct and compile the research StateGraph.

        Returns:
            Compiled LangGraph graph ready for ``.invoke()`` / ``.stream()``.
        """
        graph = StateGraph(AgentState)

        graph.add_node("research", self._node_research)
        graph.add_node("validate", self._node_validate)
        graph.add_node("summarize", self._node_summarize)

        graph.set_entry_point("research")
        graph.add_edge("research", "validate")
        graph.add_conditional_edges(
            "validate",
            self._route_after_validate,
            {
                "research": "research",
                "summarize": "summarize",
            },
        )
        graph.add_edge("summarize", END)

        return graph.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _node_research(self, state: AgentState) -> dict[str, Any]:
        """
        Node: expand the query and retrieve information snippets.

        Calls the (mock) search tool and accumulates findings in
        ``state["context"][_CTX_FINDINGS]``.
        """
        state = self._increment_step(state)
        self._log_step("research", state)

        query = state["messages"][0].content  # type: ignore[union-attr]
        iterations: int = state.get("context", {}).get(self._CTX_ITERATIONS, 0)
        existing: list[str] = state.get("context", {}).get(self._CTX_FINDINGS, [])
        sources: list[str] = state.get("context", {}).get(self._CTX_SOURCES, [])

        expansion_prompt = (
            f"You are a research assistant. The user wants to research: '{query}'.\n"
            "Provide 3 focused sub-queries that would help gather comprehensive "
            "information. Return ONLY a JSON array of strings."
        )
        try:
            expansion_msg = self._invoke_llm_with_retry(
                [
                    SystemMessage(content="You are a precise research query expander."),
                    HumanMessage(content=expansion_prompt),
                ]
            )
            parsed = json.loads(_extract_text_content(expansion_msg.content))
            sub_queries: list[str] = (
                list(map(str, parsed)) if isinstance(parsed, list) else [query]
            )  # type: ignore[assignment]
        except json.JSONDecodeError:
            logger.warning(
                "Query expansion returned non-JSON, falling back to original query"
            )
            sub_queries: list[str] = [query]  # type: ignore[no-redef]
        except Exception:
            logger.warning(
                "Query expansion failed unexpectedly, falling back to original query",
                exc_info=True,
            )
            sub_queries: list[str] = [query]  # type: ignore[no-redef]

        search_tool = next((t for t in self.tools if t.name == "web_search"), None)
        new_findings: list[str] = []
        new_sources: list[str] = []
        for sq in sub_queries[:3]:
            if search_tool is not None:
                result = str(search_tool.invoke(sq))
                new_findings.append(result)
                for line in result.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("Source:"):
                        src_url = stripped[len("Source:") :].strip()
                        if src_url:
                            new_sources.append(src_url)

        updated_context = {
            **state.get("context", {}),
            self._CTX_FINDINGS: existing + new_findings,
            self._CTX_SOURCES: list(dict.fromkeys(sources + new_sources)),
            self._CTX_ITERATIONS: iterations + 1,
        }

        self._log.info(
            "Research node completed",
            extra={
                "iteration": iterations + 1,
                "new_snippets": len(new_findings),
                "total_snippets": len(updated_context[self._CTX_FINDINGS]),
            },
        )

        return {"step_count": state["step_count"], "context": updated_context}

    def _node_validate(self, state: AgentState) -> dict[str, Any]:
        """
        Node: assess the quality and sufficiency of collected findings.

        Writes ``state["context"]["validation_ok"]`` (bool) and
        ``state["context"]["validation_reason"]`` (str).
        """
        state = self._increment_step(state)
        self._log_step("validate", state)

        findings: list[str] = state.get("context", {}).get(self._CTX_FINDINGS, [])
        query = state["messages"][0].content  # type: ignore[union-attr]

        if not findings:
            return {
                "step_count": state["step_count"],
                "context": {
                    **state.get("context", {}),
                    "validation_ok": False,
                    "validation_reason": "No findings retrieved.",
                },
            }

        validation_prompt = (
            f"You are a research quality assessor.\n"
            f"Query: {query}\n"
            f"Retrieved {len(findings)} snippets. First 3 snippets:\n"
            + "\n".join(f"- {s[:200]}" for s in findings[:3])
            + "\n\nAre these findings sufficient to write a comprehensive answer? "
            'Reply with a JSON object: {"sufficient": true/false, "reason": "..."}'
        )

        try:
            validation_msg = self._invoke_llm_with_retry(
                [
                    SystemMessage(
                        content="You are a strict research quality assessor."
                    ),
                    HumanMessage(content=validation_prompt),
                ]
            )
            result = json.loads(_extract_text_content(validation_msg.content))
            is_sufficient: bool = bool(result.get("sufficient", True))
            reason: str = result.get("reason", "")
        except json.JSONDecodeError:
            is_sufficient = False
            reason = (
                "Validation parsing failed — defaulting to insufficient for safety."
            )
        except Exception:
            is_sufficient = False
            reason = "Validation failed unexpectedly — defaulting to insufficient."

        self._log.info(
            "Validate node completed",
            extra={"sufficient": is_sufficient, "reason": reason[:120]},
        )

        return {
            "step_count": state["step_count"],
            "context": {
                **state.get("context", {}),
                "validation_ok": is_sufficient,
                "validation_reason": reason,
            },
        }

    def _node_summarize(self, state: AgentState) -> dict[str, Any]:
        """
        Node: distil validated findings into a final ``ResearchResult``.

        Serialises the result into ``state["context"][_CTX_RESULT]`` as a dict
        so that downstream agents (e.g. ``AnalystAgent``) can deserialise it.
        """
        state = self._increment_step(state)
        self._log_step("summarize", state)

        query = state["messages"][0].content  # type: ignore[union-attr]
        findings: list[str] = state.get("context", {}).get(self._CTX_FINDINGS, [])
        sources: list[str] = state.get("context", {}).get(self._CTX_SOURCES, [])

        summary_prompt = (
            f"You are a professional research analyst.\n"
            f"Research query: {query}\n\n"
            f"Based on the following {len(findings)} retrieved snippets, write a "
            "comprehensive, structured summary. Include key facts, main themes, and "
            "any conflicting information.\n\n"
            "Snippets:\n"
            + "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(findings[:10]))
            + "\n\nProvide a JSON object with keys: "
            '{"summary": "...", "confidence": 0.0-1.0}'
        )

        summary_text = "Summary unavailable."
        confidence = 0.5
        try:
            summary_msg = self._invoke_llm_with_retry(
                [
                    SystemMessage(content="You are a precise research summariser."),
                    HumanMessage(content=summary_prompt),
                ]
            )
            parsed = json.loads(_extract_text_content(summary_msg.content))
            summary_text = parsed.get("summary", str(summary_msg.content))
            confidence = float(parsed.get("confidence", 0.7))
        except json.JSONDecodeError:
            summary_text = _extract_text_content(summary_msg.content)
        except Exception:
            logger.warning(
                "Summarize node parsing failed — using defaults",
                exc_info=True,
            )

        result = ResearchResult(
            query=str(query),
            findings=findings,
            summary=summary_text,
            sources=sources,
            confidence=min(1.0, max(0.0, confidence)),
            metadata=state.get("metadata", {}),
        )

        self._log.info(
            "Summarize node completed",
            extra={
                "confidence": result.confidence,
                "summary_length": len(result.summary),
                "sources_count": len(result.sources),
            },
        )

        return {
            "step_count": state["step_count"],
            "context": {
                **state.get("context", {}),
                self._CTX_RESULT: result.to_dict(),
            },
            "status": "done",
        }

    # ------------------------------------------------------------------
    # Conditional routing
    # ------------------------------------------------------------------

    def _route_after_validate(
        self, state: AgentState
    ) -> Literal["research", "summarize"]:
        """
        Route logic after the ``validate`` node.

        Returns ``"research"`` when the findings are insufficient AND the
        iteration budget has not been exhausted; ``"summarize"`` otherwise.
        """
        ctx = state.get("context", {})
        is_ok: bool = ctx.get("validation_ok", True)
        iterations: int = ctx.get(self._CTX_ITERATIONS, 0)

        max_iter: int = get_settings().max_research_iterations
        if not is_ok and iterations < max_iter:
            self._log.info(
                "Routing back to research",
                extra={"iterations": iterations, "max": max_iter},
            )
            return "research"

        return "summarize"

    # ------------------------------------------------------------------
    # Public run interface
    # ------------------------------------------------------------------

    def _execute(self, query: str) -> AgentState:
        """Execute the research graph and return the final state."""
        if not query or not query.strip():
            raise AgentValidationError(f"[{self.name}] Query must not be empty.")
        query = input_validator.validate(query)
        self._log.info("Starting research run", extra={"query": query[:120]})
        initial_state = self._make_initial_state(query)
        try:
            final_state: AgentState = self._graph.invoke(
                initial_state, config=self._get_config()
            )
        except Exception as exc:
            raise AgentExecutionError(
                f"[{self.name}] Research pipeline failed: {exc}"
            ) from exc
        return final_state

    def run(self, query: str) -> str:
        """
        Execute the research pipeline for a given query.

        Args:
            query: The research question or topic.

        Returns:
            The LLM-generated research summary as a string.

        Raises:
            AgentExecutionError: On unrecoverable graph errors.
        """
        final_state = self._execute(query)

        result_dict = final_state.get("context", {}).get(self._CTX_RESULT)
        if result_dict:
            return result_dict.get("summary", "No summary generated.")

        for msg in reversed(final_state.get("messages", [])):
            if isinstance(msg, AIMessage):
                return str(msg.content)

        return "Research completed but no output was produced."

    def run_structured(self, query: str) -> ResearchResult:
        """
        Execute the research pipeline and return the full ``ResearchResult``.

        Args:
            query: The research question or topic.

        Returns:
            A populated ``ResearchResult`` dataclass.

        Raises:
            AgentExecutionError: On unrecoverable graph errors.
            AgentValidationError: When no structured result is present.
        """
        final_state = self._execute(query)

        result_dict = final_state.get("context", {}).get(self._CTX_RESULT)
        if not result_dict:
            raise AgentValidationError(
                "ResearchAgent: graph completed without producing a ResearchResult."
            )

        return ResearchResult(**result_dict)
