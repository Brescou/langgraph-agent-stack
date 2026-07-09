"""
domain_packs/common/structured_llm.py — Reusable single-node LLM pack base.

Vertical packs subclass ``StructuredLLMPack``, declare schemas, and implement
``build_prompt`` (+ optional ``parse_llm_output``). Supports optional
``connector`` for RAG enrichment on packs that accept reference documents.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import anyio.from_thread
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from agents.base_agent import (
    AgentAuthenticationError,
    AgentBudgetExceededError,
    AgentExecutionError,
    AgentTimeoutError,
    AgentValidationError,
    extract_text_content,
    make_auth_error,
)
from agents.llm_retry import is_auth_llm_error
from connectors.base import (
    BaseConnector,
    ConnectorRequest,
    ConnectorResult,
    SourceRef,
    record_to_source_ref,
)
from core.config import get_settings
from core.memory import create_checkpointer
from core.observability import trace_span
from core.security import InputValidator
from domain_packs.common.compliance import REGULATED_PACK_IDS, apply_compliance_output
from domain_packs.common.output_guard import (
    cross_check_output_if_enabled,
    guard_llm_output,
)
from pack_kernel.base_pack import (
    BaseDomainPack,
    normalize_pack_stream_event,
    pack_stream_event,
)

logger = logging.getLogger(__name__)

_CONNECTOR_SNIPPET_VALIDATOR = InputValidator(max_length=100_000)


class _StructuredState(TypedDict, total=False):
    input_json: str
    reference_text: str
    output: dict[str, Any] | None
    error: str | None
    status: str


def _fetch_connector_result_sync(
    connector: BaseConnector, query: str
) -> ConnectorResult:
    """Run async ``connector.fetch`` from a sync LangGraph node.

    Pack nodes run in plain ``ThreadPoolExecutor`` threads, which are not
    anyio worker threads — ``anyio.from_thread.run`` only works in the
    latter, so fall back to a private event loop otherwise.
    """
    request = ConnectorRequest(query=query)

    async def _fetch() -> ConnectorResult:
        return await connector.fetch(request)

    try:
        return anyio.from_thread.run(_fetch)
    except RuntimeError:
        return anyio.run(_fetch)


# Maximum raw JSON blob size accepted from LLM responses (512 KiB).
_MAX_JSON_RESPONSE_BYTES = 512 * 1024

_json_output_parser = JsonOutputParser()


def extract_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response using LangChain's JSON parser."""
    if len(raw.encode("utf-8")) > _MAX_JSON_RESPONSE_BYTES:
        raise ValueError("LLM JSON response exceeds maximum allowed size.")
    try:
        parsed = _json_output_parser.parse(raw.strip())
    except OutputParserException as exc:
        raise ValueError(f"Could not parse JSON from LLM response: {exc}") from exc

    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                return item
        raise ValueError("LLM JSON array contains no object.")
    raise ValueError("LLM response must decode to a JSON object.")


class StructuredLLMPack(BaseDomainPack):
    """Base class for vertical packs driven by one structured LLM call."""

    input_schema: ClassVar[type[BaseModel]]
    output_schema: ClassVar[type[BaseModel]]
    primary_field: ClassVar[str]
    node_name: ClassVar[str] = "structured_llm_node"

    def __init__(
        self,
        run_id: str | None = None,
        llm: Any | None = None,
        checkpointer: Any | None = None,
        budget_usd: float | None = None,
        connector: BaseConnector | None = None,
    ) -> None:
        super().__init__(
            run_id=run_id, llm=llm, checkpointer=checkpointer, budget_usd=budget_usd
        )
        self.run_id = run_id or str(uuid.uuid4())
        self._checkpointer = checkpointer or create_checkpointer(get_settings())
        self._connector = connector
        #: SourceRefs for the connector snippets used on the last reference-text
        #: resolution (audit trail for citations). Reset on each resolution.
        self.last_source_refs: list[SourceRef] = []
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(_StructuredState)
        graph.add_node(self.node_name, self._run_node)
        graph.set_entry_point(self.node_name)
        graph.add_edge(self.node_name, END)
        return graph.compile(checkpointer=self._checkpointer)

    @classmethod
    def _coerce_input(cls, inp: BaseModel) -> BaseModel:
        """Return a validated instance of ``input_schema`` (no ``assert``)."""
        if isinstance(inp, cls.input_schema):
            return inp
        return cls.input_schema.model_validate(
            inp.model_dump() if isinstance(inp, BaseModel) else inp
        )

    @classmethod
    def build_prompt(cls, inp: BaseModel, *, reference_text: str = "") -> str:
        """Build the LLM prompt from validated input and optional reference text."""
        raise NotImplementedError

    @classmethod
    def parse_llm_output(
        cls,
        raw: str,
        inp: BaseModel,
        *,
        reference_text: str = "",
        run_id: str | None = None,
        llm: Any | None = None,
    ) -> BaseModel:
        """Parse LLM text into ``output_schema`` (default: JSON extraction)."""
        data = extract_json_object(raw)
        data = guard_llm_output(cls.pack_id, raw, data, run_id=run_id)
        data = apply_compliance_output(cls.pack_id, data)
        strict = cls.pack_id in REGULATED_PACK_IDS
        output = cls.output_schema.model_validate(data, strict=strict)
        cross_check_output_if_enabled(
            cls.pack_id,
            task_summary=cls.primary_text(inp),
            output_json=output.model_dump(),
            llm=llm,
            run_id=run_id,
        )
        return output

    @classmethod
    def primary_text(cls, inp: BaseModel) -> str:
        """Text used for policy validation and run-history query field."""
        coerced = cls._coerce_input(inp)
        value = getattr(coerced, cls.primary_field, None)
        if value:
            return str(value)
        return str(coerced.model_dump())[:500]

    def _resolve_reference_text(self, inp: BaseModel) -> str:
        parts: list[str] = []
        for field in (
            "reference_text",
            "rfp_text",
            "contract_text",
            "document_text",
            "job_description",
            "resume_text",
        ):
            if hasattr(inp, field):
                value = getattr(inp, field)
                if value:
                    parts.append(str(value))
        self.last_source_refs = []
        if self._connector is not None:
            query = self.primary_text(inp)
            try:
                result = _fetch_connector_result_sync(self._connector, query)
                for record in result.records:
                    snippet = (
                        record.get("snippet")
                        or record.get("text")
                        or record.get("content")
                        or str(record)
                    )
                    text = str(snippet)
                    try:
                        _CONNECTOR_SNIPPET_VALIDATOR.check_content_safety(
                            text, max_length=100_000
                        )
                    except ValueError:
                        logger.warning(
                            "Skipping connector snippet that failed validation "
                            "for pack %s",
                            self.pack_id,
                        )
                        continue
                    ref = record_to_source_ref(record, len(self.last_source_refs) + 1)
                    self.last_source_refs.append(ref)
                    parts.append(f"[{ref.id}] {text}")
            except Exception as exc:
                logger.warning("Connector fetch failed for %s: %s", self.pack_id, exc)
        return "\n\n---\n\n".join(parts)

    def _inject_source_citations(self, output: BaseModel) -> BaseModel:
        """Fill an existing ``sources`` output field with citation strings.

        Only applies when the pack's ``output_schema`` already declares a
        ``sources`` field (no schema change) and connector snippets were used
        for the last reference-text resolution. Citations use the stable
        ``[id] title — url`` format from :meth:`SourceRef.citation`.
        """
        if not self.last_source_refs:
            return output
        if "sources" not in self.output_schema.model_fields:
            return output
        current = getattr(output, "sources", None)
        existing = [str(item) for item in current] if isinstance(current, list) else []
        citations = [ref.citation() for ref in self.last_source_refs]
        return output.model_copy(
            update={"sources": list(dict.fromkeys([*existing, *citations]))}
        )

    def _run_node(self, state: _StructuredState) -> _StructuredState:
        inp = self.input_schema.model_validate_json(state.get("input_json", "{}"))
        reference_text = state.get("reference_text") or self._resolve_reference_text(
            inp
        )

        with trace_span(
            self.node_name, {"run_id": self.run_id, "pack_id": self.pack_id}
        ):
            try:
                if self._llm is None:
                    raise AgentExecutionError(
                        f"{self.__class__.__name__} requires an LLM instance."
                    )
                prompt = self.build_prompt(inp, reference_text=reference_text)
                from core.mock_llm import mock_output_schema_context

                try:
                    with mock_output_schema_context(self.output_schema):
                        response = self._llm.invoke(prompt)
                except Exception as exc:
                    if is_auth_llm_error(exc):
                        raise make_auth_error(
                            self.__class__.__name__,
                            get_settings().llm_provider,
                            exc,
                        ) from exc
                    raise
                raw = extract_text_content(
                    response.content if hasattr(response, "content") else response
                )
                output = self.parse_llm_output(
                    raw,
                    inp,
                    reference_text=reference_text,
                    run_id=self.run_id,
                    llm=self._llm,
                )
                output = self._inject_source_citations(output)
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
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                return {
                    **state,
                    "output": None,
                    "status": "error",
                    "error": str(exc),
                }  # type: ignore[return-value]

    def run_from_input(self, body: BaseModel) -> BaseModel:
        if not isinstance(body, self.input_schema):
            body = self.input_schema.model_validate(body)
        reference_text = self._resolve_reference_text(body)
        initial: _StructuredState = {
            "input_json": body.model_dump_json(),
            "reference_text": reference_text,
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
                f"[{self.__class__.__name__}] Pipeline failed: {exc}"
            ) from exc

        if final.get("status") == "error":
            raise AgentExecutionError(
                f"[{self.__class__.__name__}] {final.get('error')}"
            )
        out = final.get("output")
        if not out:
            raise AgentExecutionError(
                f"[{self.__class__.__name__}] completed without output."
            )
        return self.output_schema.model_validate(out)

    def run(self, query: str) -> BaseModel:
        if not query or not query.strip():
            raise AgentValidationError(f"{self.__class__.__name__}.run() needs text.")
        fields = self.input_schema.model_fields
        if self.primary_field not in fields:
            raise AgentValidationError(
                f"{self.__class__.__name__}.primary_field "
                f"'{self.primary_field}' is not on {self.input_schema.__name__}."
            )
        payload: dict[str, Any] = {self.primary_field: query.strip()}
        if self.primary_field == "ticket_subject" and "body" in fields:
            payload["body"] = query.strip()
        return self.run_from_input(self.input_schema(**payload))

    async def arun(self, query: str) -> BaseModel:
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
        yield pack_stream_event("phase_started", phase=self.pack_id)
        # run_from_input drives a sync graph.invoke; keep it off the event
        # loop thread so async checkpointers (AsyncSqliteSaver, ...) work.
        result = await asyncio.to_thread(self.run_from_input, body)
        yield pack_stream_event("phase_completed", phase=self.pack_id)
        yield pack_stream_event("pipeline_completed", result=result)

    async def _iter_stream_events(self, query: str) -> AsyncIterator[dict[str, Any]]:
        if not query or not query.strip():
            raise AgentValidationError(f"{self.__class__.__name__}.run() needs text.")
        fields = self.input_schema.model_fields
        if self.primary_field not in fields:
            raise AgentValidationError(
                f"{self.__class__.__name__}.primary_field "
                f"'{self.primary_field}' is not on {self.input_schema.__name__}."
            )
        payload: dict[str, Any] = {self.primary_field: query.strip()}
        if self.primary_field == "ticket_subject" and "body" in fields:
            payload["body"] = query.strip()
        body = self.input_schema(**payload)
        async for event in self._iter_stream_events_from_input(body):
            yield event
