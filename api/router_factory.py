"""api/router_factory.py — Per-pack APIRouter factory.

Called during lifespan startup for each pack registered in PackRegistry.
Each router exposes typed /run and /run/stream endpoints for one pack.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse

import api.state as state
from agents.base_agent import (
    AgentBudgetExceededError,
    AgentExecutionError,
    AgentTimeoutError,
    AgentValidationError,
)
from api.dependencies import (
    _rate_limit_key,
    pack_primary_text,
    pack_runtime_kwargs,
    validate_pack_body_fields,
    validate_pack_query,
    verify_api_key,
)
from control_plane.enforce import effective_stream_timeout_seconds
from core.config import get_settings
from pack_kernel.base_pack import normalize_pack_stream_event
from pack_kernel.registry import PackRegistry

logger = logging.getLogger(__name__)


def _pack_has_structured_input(pack_cls: type) -> bool:
    return "run_from_input" in pack_cls.__dict__


def _pack_has_structured_stream(pack_cls: type) -> bool:
    return "stream_events_from_input" in pack_cls.__dict__


def _invoke_pack_run(pack_cls: type, pipeline: Any, body: Any) -> Any:
    if _pack_has_structured_input(pack_cls):
        return pipeline.run_from_input(body)
    return pipeline.run(pack_primary_text(body))


async def _iter_pack_stream_events(
    pack_cls: type, pipeline: Any, body: Any
) -> AsyncIterator[Any]:
    if _pack_has_structured_stream(pack_cls):
        events = pipeline.stream_events_from_input(body)
        async for event in cast(AsyncIterator[dict[str, Any]], events):
            yield normalize_pack_stream_event(event)
        return
    async for event in cast(
        AsyncIterator[dict[str, Any]],
        pipeline.stream_events(pack_primary_text(body)),
    ):
        yield event


def _serialize_pack_result(
    result: Any, output_model: type, cost_usd: float | None
) -> Any:
    if hasattr(output_model, "from_analysis_report"):
        return output_model.from_analysis_report(result, cost_usd=cost_usd)
    if hasattr(output_model, "from_research_result"):
        return output_model.from_research_result(result, cost_usd=cost_usd)
    if hasattr(output_model, "from_summary_result"):
        return output_model.from_summary_result(result, cost_usd=cost_usd)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result


async def _run_in_executor(fn: Any, *args: Any) -> Any:
    """Execute a blocking callable in the application thread pool."""
    import contextvars
    import functools

    from core.observability import active_pipelines

    if state.executor is None:
        raise RuntimeError("Application not started — call during lifespan only")
    if active_pipelines is not None:
        active_pipelines.inc()
    try:
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(
            state.executor, functools.partial(ctx.run, fn, *args)
        )
    finally:
        if active_pipelines is not None:
            active_pipelines.dec()


def build_pack_router(
    pack_id: str,
    pack_cls: type,
    input_model: type,
    output_model: type,
) -> APIRouter:
    """Build a per-pack APIRouter with typed /run and /run/stream endpoints."""
    router = APIRouter(prefix=f"/packs/{pack_id}", tags=[pack_id])

    async def run_pack(
        body: input_model,  # type: ignore[valid-type]
        request: Request,
        response: Response,
        _auth: Annotated[None, Depends(verify_api_key)],
    ) -> Any:
        if state.shutting_down.is_set():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server is shutting down.",
            )
        run_id = str(uuid.uuid4())
        requested_version = request.headers.get("X-Pack-Version") or None

        if requested_version is None and state.shared_memory is not None:
            session_id = getattr(body, "session_id", None) or None
            if session_id and hasattr(
                state.shared_memory, "get_pack_version_for_session"
            ):
                requested_version = state.shared_memory.get_pack_version_for_session(
                    session_id, pack_id
                )

        try:
            pack_cls_to_use = PackRegistry.get(
                pack_id,
                version=requested_version,
                affinity_key=_rate_limit_key(request),
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc

        used_version = next(
            (
                pv.version
                for pv in PackRegistry._get_versions(pack_id)
                if pv.pack_cls is pack_cls_to_use
            ),
            "unknown",
        )
        response.headers["X-Pack-Version-Used"] = used_version

        validate_pack_body_fields(pack_id, body)
        query = validate_pack_query(pack_id, pack_primary_text(body))

        settings = get_settings()
        from domain_packs.common.compliance import assert_regulated_pack_runtime_enabled

        try:
            assert_regulated_pack_runtime_enabled(
                pack_id, regulated_packs_enabled=settings.regulated_packs_enabled
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)
            ) from exc

        def _execute() -> Any:
            with pack_cls_to_use(
                run_id=run_id,
                llm=state.get_shared_llm(),
                checkpointer=state.get_shared_checkpointer(),
                **pack_runtime_kwargs(pack_cls_to_use),
            ) as pipeline:
                result = _invoke_pack_run(pack_cls_to_use, pipeline, body)
                cost_usd = getattr(pipeline, "cost_usd", None)
                result_payload = (
                    result.to_dict()
                    if hasattr(result, "to_dict")
                    else result.model_dump()
                    if hasattr(result, "model_dump")
                    else {}
                )
                if state.shared_memory is not None:
                    session_id_for_history = getattr(body, "session_id", None) or None
                    state.shared_memory.save_run(
                        run_id=run_id,
                        query=query,
                        result=result_payload,
                        metadata={
                            "pack_id": pack_id,
                            "pack_version": used_version,
                            **(
                                {"session_id": session_id_for_history}
                                if session_id_for_history
                                else {}
                            ),
                        },
                    )
                return _serialize_pack_result(result, output_model, cost_usd)

        try:
            return await _run_in_executor(_execute)
        except AgentBudgetExceededError as exc:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED, detail=str(exc)
            ) from exc
        except AgentTimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)
            ) from exc
        except (AgentExecutionError, AgentValidationError) as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc

    run_pack.__annotations__["body"] = input_model
    router.add_api_route(
        "/run",
        run_pack,
        methods=["POST"],
        summary=f"Run {pack_id} pipeline",
        response_model=output_model,
    )

    async def stream_pack(
        body: input_model,  # type: ignore[valid-type]
        request: Request,
        _auth: Annotated[None, Depends(verify_api_key)],
    ) -> StreamingResponse:
        if state.shutting_down.is_set():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server is shutting down.",
            )
        run_id = str(uuid.uuid4())
        requested_version = request.headers.get("X-Pack-Version") or None
        if requested_version is None and state.shared_memory is not None:
            session_id = getattr(body, "session_id", None) or None
            if session_id and hasattr(
                state.shared_memory, "get_pack_version_for_session"
            ):
                requested_version = state.shared_memory.get_pack_version_for_session(
                    session_id, pack_id
                )

        try:
            pack_cls_to_use = PackRegistry.get(
                pack_id,
                version=requested_version,
                affinity_key=_rate_limit_key(request),
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc

        used_version = next(
            (
                pv.version
                for pv in PackRegistry._get_versions(pack_id)
                if pv.pack_cls is pack_cls_to_use
            ),
            "unknown",
        )

        validate_pack_body_fields(pack_id, body)
        validate_pack_query(pack_id, pack_primary_text(body))

        async def _event_generator() -> AsyncGenerator[str, None]:
            pack = pack_cls_to_use(
                run_id=run_id,
                llm=state.get_shared_llm(),
                checkpointer=state.get_shared_checkpointer(),
                **pack_runtime_kwargs(pack_cls_to_use),
            )
            try:
                async for event in _iter_pack_stream_events(
                    pack_cls_to_use, pack, body
                ):
                    yield f"data: {json.dumps(event, default=str)}\n\n"
            finally:
                pack.close()

        stream_timeout = effective_stream_timeout_seconds(pack_id, get_settings())

        async def _timed_event_generator() -> AsyncGenerator[str, None]:
            try:
                async with asyncio.timeout(stream_timeout):
                    async for chunk in _event_generator():
                        yield chunk
            except TimeoutError:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Stream timed out after {stream_timeout}s'})}\n\n"

        return StreamingResponse(
            _timed_event_generator(),
            media_type="text/event-stream",
            headers={
                "X-Pack-Version-Used": used_version,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    stream_pack.__annotations__["body"] = input_model
    router.add_api_route(
        "/run/stream",
        stream_pack,
        methods=["POST"],
        summary=f"Stream {pack_id} pipeline",
    )

    return router
