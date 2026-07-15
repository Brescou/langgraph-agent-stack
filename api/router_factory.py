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
    AgentAuthenticationError,
    AgentExecutionError,
    AgentTimeoutError,
    AgentValidationError,
    make_auth_error,
)
from agents.llm_retry import find_auth_cause
from api.dependencies import (
    _rate_limit_key,
    pack_primary_text,
    pack_runtime_kwargs,
    validate_pack_body_fields,
    validate_pack_query,
    verify_api_key,
)
from api.pack_execution import (
    SESSION_IN_FLIGHT_DETAIL,
    create_review_best_effort,
    execute_typed_pack_run,
    pack_has_structured_stream,
)
from control_plane.enforce import effective_stream_timeout_seconds
from core.config import get_settings
from pack_kernel.base_pack import normalize_pack_stream_event
from pack_kernel.registry import PackRegistry

logger = logging.getLogger(__name__)


async def _iter_pack_stream_events(
    pack_cls: type, pipeline: Any, body: Any
) -> AsyncIterator[Any]:
    if pack_has_structured_stream(pack_cls):
        events = pipeline.stream_events_from_input(body)
        async for event in cast(AsyncIterator[dict[str, Any]], events):
            yield normalize_pack_stream_event(event)
        return
    async for event in cast(
        AsyncIterator[dict[str, Any]],
        pipeline.stream_events(pack_primary_text(body)),
    ):
        yield event


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
        requested_version = request.headers.get("X-Pack-Version") or None
        outcome = await execute_typed_pack_run(
            pack_id,
            body,
            affinity_key=_rate_limit_key(request),
            requested_version=requested_version,
        )
        response.headers["X-Pack-Version-Used"] = outcome.used_version
        return outcome.serialized

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
        session_id = getattr(body, "session_id", None) or None
        if (
            requested_version is None
            and state.shared_memory is not None
            and session_id
            and hasattr(state.shared_memory, "get_pack_version_for_session")
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
                last_event: dict[str, Any] | None = None
                async for event in _iter_pack_stream_events(
                    pack_cls_to_use, pack, body
                ):
                    last_event = event
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                await create_review_best_effort(
                    run_id=run_id,
                    pack_id=pack_id,
                    session_id=session_id,
                    result_payload=last_event or {},
                )
            except AgentTimeoutError as exc:
                logger.error(
                    "Pack stream — timeout",
                    extra={"run_id": run_id, "pack_id": pack_id, "error": str(exc)},
                )
                yield f"data: {json.dumps({'type': 'error', 'message': 'The pipeline timed out.'})}\n\n"
            except AgentAuthenticationError as exc:
                logger.error(
                    "Pack stream — LLM authentication error",
                    extra={"run_id": run_id, "pack_id": pack_id, "error": str(exc)},
                )
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            except (AgentExecutionError, AgentValidationError) as exc:
                logger.error(
                    "Pack stream — error",
                    extra={"run_id": run_id, "pack_id": pack_id, "error": str(exc)},
                )
                auth_cause = find_auth_cause(exc)
                if auth_cause is not None:
                    message = str(
                        make_auth_error(
                            pack_id, get_settings().llm_provider, auth_cause
                        )
                    )
                else:
                    message = "The pipeline encountered an error."
                yield f"data: {json.dumps({'type': 'error', 'message': message})}\n\n"
            except Exception:
                logger.exception(
                    "Pack stream — unexpected error",
                    extra={"run_id": run_id, "pack_id": pack_id},
                )
                yield f"data: {json.dumps({'type': 'error', 'message': 'Internal error'})}\n\n"
            finally:
                try:
                    pack.close()
                except Exception as close_exc:
                    logger.warning(
                        "Pack close failed after stream",
                        extra={
                            "run_id": run_id,
                            "pack_id": pack_id,
                            "error": str(close_exc),
                        },
                    )

        stream_timeout = effective_stream_timeout_seconds(pack_id, get_settings())

        if session_id and not state.try_acquire_session(session_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=SESSION_IN_FLIGHT_DETAIL,
            )

        async def _timed_event_generator() -> AsyncGenerator[str, None]:
            try:
                async with asyncio.timeout(stream_timeout):
                    async for chunk in _event_generator():
                        yield chunk
            except TimeoutError:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Stream timed out after {stream_timeout}s'})}\n\n"
            finally:
                if session_id:
                    state.release_session(session_id)

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
