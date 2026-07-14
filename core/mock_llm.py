"""Mock LLM helpers for ``LLM_PROVIDER=mock``.

Provides a schema-aware chat model so structured vertical packs receive JSON
that validates against their ``output_schema``, while the research pipeline
still receives its fixed multi-step response sequence.
"""

from __future__ import annotations

import json
import threading
import types
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime
from enum import Enum
from typing import Any, Union, get_args, get_origin

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

_mock_output_schema: ContextVar[type[BaseModel] | None] = ContextVar(
    "mock_output_schema", default=None
)
_mock_plain_bullets: ContextVar[int | None] = ContextVar(
    "mock_plain_bullets", default=None
)

_research_local = threading.local()

# Stable model id for CostTracker / pricing table (see core/cost.py).
MOCK_MODEL_ID = "mock-provider"


def _message_text(message: BaseMessage) -> str:
    """Flatten a chat message's content to a string for token estimation."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _synthetic_token_count(text: str) -> int:
    """Deterministic token estimate from character length (≈4 chars/token)."""
    return max(1, len(text) // 4)


# Ordered responses for ResearchAgent (3) + AnalystAgent (3) — see agents/*.py.
RESEARCH_PIPELINE_RESPONSES: tuple[str, ...] = (
    json.dumps(["sub-query 1", "sub-query 2", "sub-query 3"]),
    json.dumps({"sufficient": True, "reason": "Mock validation passed."}),
    json.dumps(
        {
            "summary": "Mock research summary based on findings.",
            "confidence": 0.85,
        }
    ),
    json.dumps(
        {
            "insights": [
                "Mock insight 1: Key trend identified.",
                "Mock insight 2: Pattern detected.",
            ],
            "confidence": 0.82,
        }
    ),
    json.dumps(
        {
            "patterns": ["Mock pattern: Consistent growth."],
            "implications": ["Mock implication: Continued adoption expected."],
        }
    ),
    "Mock executive summary: The analysis reveals significant trends across the research domain.",
)


def reset_mock_research_sequence(*, start: int = 0) -> None:
    """Reset the per-thread research pipeline mock response index."""
    _research_local.index = start


def _research_index() -> int:
    if not hasattr(_research_local, "index"):
        _research_local.index = 0
    return int(_research_local.index)


def _advance_research_index() -> None:
    _research_local.index = _research_index() + 1


@contextmanager
def mock_output_schema_context(schema: type[BaseModel]) -> Iterator[None]:
    """Tell the mock LLM to emit JSON valid for ``schema`` on the next invoke."""
    token = _mock_output_schema.set(schema)
    try:
        yield
    finally:
        _mock_output_schema.reset(token)


@contextmanager
def mock_plain_bullets_context(count: int) -> Iterator[None]:
    """Tell the mock LLM to return plain-text bullet lines on the next invoke."""
    token = _mock_plain_bullets.set(count)
    try:
        yield
    finally:
        _mock_plain_bullets.reset(token)


def _unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


def _fake_scalar(field_name: str, annotation: Any) -> Any:
    annotation = _unwrap_optional(annotation)
    if annotation is str:
        return f"Mock {field_name.replace('_', ' ')}"
    if annotation is int:
        return 1
    if annotation is float:
        if field_name == "confidence":
            return 0.85
        return 0.5
    if annotation is bool:
        return False
    if annotation is date:
        return date(2026, 1, 1)
    if annotation is datetime:
        return datetime(2026, 1, 1, tzinfo=datetime.now().astimezone().tzinfo)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return next(iter(annotation))
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return generate_mock_payload(annotation)
    origin = get_origin(annotation)
    if origin in (list, Sequence):
        args = get_args(annotation)
        inner = args[0] if args else str
        inner = _unwrap_optional(inner)
        if inner is str:
            return [f"Mock {field_name} item"]
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return [generate_mock_payload(inner)]
        return [f"Mock {field_name} item"]
    if origin in (dict, Mapping):
        key_type, value_type = get_args(annotation) or (str, str)
        if value_type is str or _unwrap_optional(value_type) is str:
            return {"mock_section": f"Mock {field_name} content"}
        return {"mock_key": "mock_value"}
    if annotation is Any:
        return f"Mock {field_name}"
    return f"Mock {field_name}"


def generate_mock_payload(model: type[BaseModel]) -> dict[str, Any]:
    """Build a minimal JSON-serialisable dict that validates against ``model``."""
    payload: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        if field.is_required():
            payload[name] = _fake_scalar(name, field.annotation)
            continue
        if field.default is not PydanticUndefined:
            payload[name] = field.default
        elif field.default_factory is not None:
            payload[name] = field.default_factory()  # type: ignore[misc]
        else:
            payload[name] = _fake_scalar(name, field.annotation)
    return model.model_validate(payload).model_dump(mode="json")


def minimal_valid_input(model: type[BaseModel]) -> dict[str, Any]:
    """Return a minimal JSON body that validates against a pack ``input_schema``."""
    return generate_mock_payload(model)


class MockProviderChatModel(BaseChatModel):
    """Deterministic mock chat model with schema-aware and pipeline modes."""

    model_name: str = MOCK_MODEL_ID

    @property
    def _llm_type(self) -> str:
        return MOCK_MODEL_ID

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}

    def bind_tools(self, tools: Any, **kwargs: Any) -> MockProviderChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = self._next_content()
        input_text = "".join(_message_text(message) for message in messages)
        input_tokens = _synthetic_token_count(input_text)
        output_tokens = _synthetic_token_count(content)
        usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        message = AIMessage(
            content=content,
            usage_metadata=usage_metadata,
            response_metadata={
                "model_name": self.model_name,
                "model_id": self.model_name,
            },
        )
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={"model_name": self.model_name},
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _next_content(self) -> str:
        schema = _mock_output_schema.get()
        if schema is not None:
            return json.dumps(generate_mock_payload(schema))

        bullet_count = _mock_plain_bullets.get()
        if bullet_count is not None:
            return "\n".join(
                f"- Mock bullet {index + 1}" for index in range(bullet_count)
            )

        index = _research_index()
        if index < len(RESEARCH_PIPELINE_RESPONSES):
            content = RESEARCH_PIPELINE_RESPONSES[index]
            _advance_research_index()
            return content

        return json.dumps({"mock": True, "id": str(uuid.uuid4())})
