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
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agents.base_agent import (
    AgentError,
    AgentExecutionError,
    AgentState,
    AgentValidationError,
    BaseAgent,
)
from core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------


@dataclass
class ResearchResult:
    """
    Structured output produced by the ResearchAgent.

    Attributes:
        query: Original research query.
        findings: List of raw text snippets collected during retrieval.
        summary: LLM-generated summary of the consolidated findings.
        sources: List of source identifiers (URLs, doc IDs, …).
        confidence: Self-reported confidence score between 0.0 and 1.0.
        metadata: Arbitrary run-level metadata forwarded from ``AgentState``.
    """

    query: str
    findings: list[str] = field(default_factory=list)
    summary: str = ""
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a plain dictionary."""
        return {
            "query": self.query,
            "findings": self.findings,
            "summary": self.summary,
            "sources": self.sources,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


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

    # Key under which we store intermediate research data in AgentState.context
    _CTX_FINDINGS = "findings"
    _CTX_SOURCES = "sources"
    _CTX_ITERATIONS = "research_iterations"
    _CTX_RESULT = "research_result"

    def __init__(self, thread_id: Optional[str] = None) -> None:
        super().__init__(name="ResearchAgent", thread_id=thread_id)

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
                "research": "research",   # insufficient — loop back
                "summarize": "summarize", # sufficient — move on
            },
        )
        graph.add_edge("summarize", END)

        return graph.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _node_research(self, state: AgentState) -> AgentState:
        """
        Node: expand the query and retrieve information snippets.

        Calls the (mock) search tool and accumulates findings in
        ``state["context"][_CTX_FINDINGS]``.

        Args:
            state: Current graph state.

        Returns:
            Updated state with new findings appended to context.
        """
        state = self._increment_step(state)
        self._log_step("research", state)

        query = state["messages"][0].content  # type: ignore[union-attr]
        iterations: int = state.get("context", {}).get(self._CTX_ITERATIONS, 0)
        existing: list[str] = state.get("context", {}).get(self._CTX_FINDINGS, [])
        sources: list[str] = state.get("context", {}).get(self._CTX_SOURCES, [])

        # --- Query expansion via LLM ---
        expansion_prompt = (
            f"You are a research assistant. The user wants to research: '{query}'.\n"
            "Provide 3 focused sub-queries that would help gather comprehensive "
            "information. Return ONLY a JSON array of strings."
        )
        try:
            expansion_msg = self.llm.invoke(
                [SystemMessage(content="You are a precise research query expander."),
                 HumanMessage(content=expansion_prompt)]
            )
            sub_queries: list[str] = json.loads(expansion_msg.content)  # type: ignore[arg-type]
            if not isinstance(sub_queries, list):
                sub_queries = [query]
        except (json.JSONDecodeError, Exception):
            sub_queries = [query]

        # --- Retrieval ---
        new_findings: list[str] = []
        new_sources: list[str] = []
        for sq in sub_queries[:3]:
            snippets, src = self._mock_web_search(sq)
            new_findings.extend(snippets)
            new_sources.extend(src)

        document_snippets = self._mock_document_retrieval(query)
        new_findings.extend(document_snippets)

        updated_context = {
            **state.get("context", {}),
            self._CTX_FINDINGS: existing + new_findings,
            self._CTX_SOURCES: list(set(sources + new_sources)),
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

        return {**state, "context": updated_context}  # type: ignore[return-value]

    def _node_validate(self, state: AgentState) -> AgentState:
        """
        Node: assess the quality and sufficiency of collected findings.

        Writes ``state["context"]["validation_ok"]`` (bool) and
        ``state["context"]["validation_reason"]`` (str).

        Args:
            state: Current graph state.

        Returns:
            Updated state with validation result stored in context.
        """
        state = self._increment_step(state)
        self._log_step("validate", state)

        findings: list[str] = state.get("context", {}).get(self._CTX_FINDINGS, [])
        query = state["messages"][0].content  # type: ignore[union-attr]

        if not findings:
            return {
                **state,
                "context": {
                    **state.get("context", {}),
                    "validation_ok": False,
                    "validation_reason": "No findings retrieved.",
                },
            }  # type: ignore[return-value]

        validation_prompt = (
            f"You are a research quality assessor.\n"
            f"Query: {query}\n"
            f"Retrieved {len(findings)} snippets. First 3 snippets:\n"
            + "\n".join(f"- {s[:200]}" for s in findings[:3])
            + "\n\nAre these findings sufficient to write a comprehensive answer? "
            "Reply with a JSON object: {\"sufficient\": true/false, \"reason\": \"...\"}"
        )

        try:
            validation_msg = self.llm.invoke(
                [SystemMessage(content="You are a strict research quality assessor."),
                 HumanMessage(content=validation_prompt)]
            )
            result = json.loads(validation_msg.content)  # type: ignore[arg-type]
            is_sufficient: bool = bool(result.get("sufficient", True))
            reason: str = result.get("reason", "")
        except Exception:
            # When in doubt, proceed to summarize
            is_sufficient = True
            reason = "Validation parsing failed — defaulting to sufficient."

        self._log.info(
            "Validate node completed",
            extra={"sufficient": is_sufficient, "reason": reason[:120]},
        )

        return {
            **state,
            "context": {
                **state.get("context", {}),
                "validation_ok": is_sufficient,
                "validation_reason": reason,
            },
        }  # type: ignore[return-value]

    def _node_summarize(self, state: AgentState) -> AgentState:
        """
        Node: distil validated findings into a final ``ResearchResult``.

        Serialises the result into ``state["context"][_CTX_RESULT]`` as a dict
        so that downstream agents (e.g. ``AnalystAgent``) can deserialise it.

        Args:
            state: Current graph state.

        Returns:
            Updated state containing the serialised ``ResearchResult``.
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
            "Snippets:\n" + "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(findings[:10]))
            + "\n\nProvide a JSON object with keys: "
            "{\"summary\": \"...\", \"confidence\": 0.0-1.0}"
        )

        try:
            summary_msg = self.llm.invoke(
                [SystemMessage(content="You are a precise research summariser."),
                 HumanMessage(content=summary_prompt)]
            )
            parsed = json.loads(summary_msg.content)  # type: ignore[arg-type]
            summary_text: str = parsed.get("summary", summary_msg.content)
            confidence: float = float(parsed.get("confidence", 0.7))
        except Exception:
            summary_text = str(summary_msg.content) if 'summary_msg' in dir() else "Summary unavailable."
            confidence = 0.5

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
            **state,
            "context": {
                **state.get("context", {}),
                self._CTX_RESULT: result.to_dict(),
            },
            "status": "done",
        }  # type: ignore[return-value]

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

        Args:
            state: Current graph state.

        Returns:
            Name of the next node to execute.
        """
        ctx = state.get("context", {})
        is_ok: bool = ctx.get("validation_ok", True)
        iterations: int = ctx.get(self._CTX_ITERATIONS, 0)
        max_iter: int = settings.max_research_iterations

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

    def run(self, input: str) -> str:
        """
        Execute the research pipeline for a given query.

        Args:
            input: The research question or topic.

        Returns:
            The LLM-generated research summary as a string.

        Raises:
            AgentExecutionError: On unrecoverable graph errors.
        """
        if not input or not input.strip():
            raise AgentValidationError("ResearchAgent.run() requires a non-empty input.")

        self._log.info("Starting research run", extra={"query": input[:120]})
        initial_state = self._make_initial_state(input)

        try:
            final_state: AgentState = self._graph.invoke(
                initial_state, config=self._get_config()
            )
        except Exception as exc:
            raise AgentExecutionError(
                f"[ResearchAgent] Graph execution failed: {exc}"
            ) from exc

        result_dict = final_state.get("context", {}).get(self._CTX_RESULT)
        if result_dict:
            return result_dict.get("summary", "No summary generated.")

        # Fallback: last AI message
        for msg in reversed(final_state.get("messages", [])):
            if isinstance(msg, AIMessage):
                return str(msg.content)

        return "Research completed but no output was produced."

    def run_structured(self, input: str) -> ResearchResult:
        """
        Execute the research pipeline and return the full ``ResearchResult``.

        Args:
            input: The research question or topic.

        Returns:
            A populated ``ResearchResult`` dataclass.

        Raises:
            AgentExecutionError: On unrecoverable graph errors.
            AgentValidationError: When no structured result is present.
        """
        if not input or not input.strip():
            raise AgentValidationError(
                "ResearchAgent.run_structured() requires a non-empty input."
            )

        initial_state = self._make_initial_state(input)

        try:
            final_state: AgentState = self._graph.invoke(
                initial_state, config=self._get_config()
            )
        except Exception as exc:
            raise AgentExecutionError(
                f"[ResearchAgent] Graph execution failed: {exc}"
            ) from exc

        result_dict = final_state.get("context", {}).get(self._CTX_RESULT)
        if not result_dict:
            raise AgentValidationError(
                "ResearchAgent: graph completed without producing a ResearchResult."
            )

        return ResearchResult(**result_dict)

    # ------------------------------------------------------------------
    # Mock tools (replace with real implementations as needed)
    # ------------------------------------------------------------------

    def _mock_web_search(self, query: str) -> tuple[list[str], list[str]]:
        """
        Placeholder web search tool.

        Replace the body of this method with a call to a real search provider
        (Tavily, Bing, SerpAPI, …) without altering the signature.

        Args:
            query: The search query string.

        Returns:
            A 2-tuple of (snippets, sources) where snippets are text excerpts
            and sources are URL/identifier strings.
        """
        snippets = [
            f"[MOCK] Web result for '{query}': "
            "This is a representative paragraph of text that a real search engine "
            "would return. It contains relevant information about the query topic.",
            f"[MOCK] Secondary result for '{query}': "
            "Additional context and supporting details that complement the primary "
            "finding with different perspectives.",
        ]
        sources = [
            f"https://example.com/search?q={query.replace(' ', '+')}",
            f"https://news.example.com/{query.replace(' ', '-').lower()}",
        ]
        return snippets, sources

    def _mock_document_retrieval(self, query: str) -> list[str]:
        """
        Placeholder document retrieval tool.

        Replace with a real vector-store lookup (Chroma, Pinecone, Weaviate, …).

        Args:
            query: The semantic search query.

        Returns:
            A list of relevant document excerpts.
        """
        return [
            f"[MOCK] Document excerpt related to '{query}': "
            "This paragraph represents a chunk retrieved from an internal knowledge "
            "base or vector store. It provides domain-specific context.",
        ]
