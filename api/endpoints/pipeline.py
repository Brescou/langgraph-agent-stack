"""api/endpoints/pipeline.py — Legacy /run, /run/stream, and /research endpoints."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

import api.state as state
from agents.analyst import AnalysisReport
from agents.base_agent import (
    AgentBudgetExceededError,
    AgentExecutionError,
    AgentTimeoutError,
    AgentValidationError,
)
from agents.researcher import ResearchAgent
from api.dependencies import (
    get_legacy_pack_cls,
    pack_runtime_kwargs,
    validate_pack_query,
)
from api.models import ResearchRequest, ResearchResponse, RunRequest, RunResponse
from control_plane.enforce import effective_stream_timeout_seconds
from core.config import Settings, get_settings
from core.observability import active_pipelines

router = APIRouter(tags=["Pipeline"])
logger = logging.getLogger(__name__)


async def _run_in_executor(fn: Any, *args: Any) -> Any:
    """Execute a blocking callable in the application thread pool."""
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


@router.post(
    "/run",
    response_model=RunResponse,
    status_code=status.HTTP_200_OK,
    summary="Run the full Research + Analysis pipeline",
    response_description="Structured AnalysisReport produced by the full agent pipeline.",
)
async def run_pipeline(
    body: RunRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    pack_cls: Annotated[type[Any], Depends(get_legacy_pack_cls)],
) -> RunResponse:
    """Execute the complete multi-agent pipeline for a given query."""
    if state.shutting_down.is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is shutting down.",
        )

    try:
        query = validate_pack_query(settings.default_pack_id, body.query)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Query must not be empty."
        )

    if state.get_shared_llm() is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider is not configured. Check server logs.",
        )

    session_id = body.session_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    logger.info(
        "POST /run — pipeline started",
        extra={
            "run_id": run_id,
            "session_id": session_id,
            "query_preview": query[:120],
        },
    )

    def _execute() -> RunResponse:
        with pack_cls(
            run_id=run_id,
            llm=state.shared_llm,
            checkpointer=state.shared_checkpointer,
            **pack_runtime_kwargs(pack_cls),
        ) as pipeline:
            report = pipeline.run(query)
            cost_usd = getattr(pipeline, "cost_usd", None)
            if state.shared_memory is not None:
                state.shared_memory.save_run(
                    run_id=run_id,
                    query=query,
                    result=report.to_dict() if hasattr(report, "to_dict") else {},
                    metadata={"session_id": session_id, "agent": "MultiAgentGraph"},
                )
            return RunResponse.from_analysis_report(
                report, session_id=session_id, cost_usd=cost_usd
            )

    try:
        response = await _run_in_executor(_execute)
    except AgentValidationError as exc:
        logger.warning(
            "POST /run — validation error", extra={"run_id": run_id, "error": str(exc)}
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except AgentTimeoutError as exc:
        logger.error(
            "POST /run — pipeline timeout", extra={"run_id": run_id, "error": str(exc)}
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The agent pipeline exceeded its step budget. Try a simpler query.",
        ) from exc
    except AgentBudgetExceededError as exc:
        logger.warning(
            "POST /run — budget exceeded", extra={"run_id": run_id, "error": str(exc)}
        )
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Run cost budget exceeded.",
        ) from exc
    except AgentExecutionError as exc:
        logger.error(
            "POST /run — execution error", extra={"run_id": run_id, "error": str(exc)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The agent pipeline encountered an internal error.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "POST /run — unexpected error", extra={"run_id": run_id, "error": str(exc)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        ) from exc

    logger.info(
        "POST /run — pipeline completed",
        extra={
            "run_id": run_id,
            "session_id": session_id,
            "confidence": response.confidence,
            "insights_count": len(response.key_insights),
        },
    )
    return response


async def _stream_pipeline(
    query: str,
    session_id: str,
    run_id: str,
    pack_cls: type[Any],
) -> AsyncGenerator[str, None]:
    """Async generator that streams the pipeline execution as SSE events."""
    if active_pipelines is not None:
        active_pipelines.inc()
    try:
        yield f"data: {json.dumps({'type': 'status', 'message': 'Starting pipeline...'})}\n\n"

        pipeline = pack_cls(
            run_id=run_id,
            llm=state.get_shared_llm(),
            checkpointer=state.get_shared_checkpointer(),
            **pack_runtime_kwargs(pack_cls),
        )
        report = None

        try:
            async for event in pipeline.stream_events(query):
                kind = event["type"]
                if kind in ("phase_started", "phase_completed", "token"):
                    yield f"data: {json.dumps(event)}\n\n"
                elif kind == "pipeline_completed":
                    report_raw = event.get("report")
                    report = (
                        AnalysisReport(**report_raw)
                        if isinstance(report_raw, dict)
                        else report_raw
                    )
        finally:
            pipeline.close()

        if report is None:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Pipeline completed without a report.'})}\n\n"
            return

        logger.info(
            "POST /run/stream — pipeline completed",
            extra={
                "run_id": run_id,
                "session_id": session_id,
                "confidence": report.confidence,
            },
        )

        if state.shared_memory is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                state.executor,
                functools.partial(
                    state.shared_memory.save_run,
                    run_id=run_id,
                    query=query,
                    result=report.to_dict(),
                    metadata={"session_id": session_id, "agent": "stream_pipeline"},
                ),
            )

        done_payload = {
            "type": "done",
            "run_id": run_id,
            "session_id": session_id,
            "executive_summary": report.executive_summary,
            "key_insights": report.key_insights,
            "patterns": report.patterns,
            "implications": report.implications,
            "confidence": report.confidence,
            "research_summary": report.research_summary,
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    except AgentTimeoutError as exc:
        logger.error(
            "POST /run/stream — timeout", extra={"run_id": run_id, "error": str(exc)}
        )
        yield f"data: {json.dumps({'type': 'error', 'message': 'The pipeline timed out.'})}\n\n"
    except (AgentExecutionError, AgentValidationError) as exc:
        logger.error(
            "POST /run/stream — error", extra={"run_id": run_id, "error": str(exc)}
        )
        yield f"data: {json.dumps({'type': 'error', 'message': 'The pipeline encountered an error.'})}\n\n"
    except Exception as exc:
        logger.exception(
            "POST /run/stream — unexpected error",
            extra={"run_id": run_id, "error": str(exc)},
        )
        yield f"data: {json.dumps({'type': 'error', 'message': 'An unexpected error occurred.'})}\n\n"
    finally:
        if active_pipelines is not None:
            active_pipelines.dec()


@router.post(
    "/run/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream the full Research + Analysis pipeline as Server-Sent Events",
)
async def run_stream(
    body: RunRequest,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    pack_cls: Annotated[type[Any], Depends(get_legacy_pack_cls)],
) -> StreamingResponse:
    """Execute the complete multi-agent pipeline and stream progress as SSE."""
    if state.shutting_down.is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is shutting down.",
        )

    try:
        query = validate_pack_query(settings.default_pack_id, body.query)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Query must not be empty."
        )

    if state.get_shared_llm() is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider is not configured. Check server logs.",
        )

    session_id = body.session_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    stream_timeout = effective_stream_timeout_seconds(
        settings.default_pack_id, settings
    )

    logger.info(
        "POST /run/stream — pipeline started",
        extra={
            "run_id": run_id,
            "session_id": session_id,
            "query_preview": query[:120],
        },
    )

    async def _guarded_stream() -> AsyncGenerator[str, None]:
        try:
            async with asyncio.timeout(stream_timeout):
                async for event in _stream_pipeline(
                    query, session_id, run_id, pack_cls
                ):
                    yield event
        except TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Stream timed out after {stream_timeout}s'})}\n\n"

    return StreamingResponse(
        _guarded_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post(
    "/research",
    response_model=ResearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Run the Research-only pipeline",
    response_description="Structured ResearchResult produced by the ResearchAgent.",
)
async def run_research(
    body: ResearchRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ResearchResponse:
    """Execute only the research phase of the pipeline."""
    if state.shutting_down.is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is shutting down.",
        )

    try:
        query = state.input_validator.validate(body.query)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Query must not be empty."
        )

    if state.get_shared_llm() is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider is not configured. Check server logs.",
        )

    session_id = body.session_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    logger.info(
        "POST /research — started",
        extra={
            "run_id": run_id,
            "session_id": session_id,
            "query_preview": query[:120],
        },
    )

    def _execute() -> ResearchResponse:
        agent = ResearchAgent(
            thread_id=run_id,
            llm=state.shared_llm,
            checkpointer=state.shared_checkpointer,
        )
        result = agent.run_structured(query)
        cost_usd = getattr(agent, "cost_usd", None)
        if state.shared_memory is not None:
            state.shared_memory.save_run(
                run_id=run_id,
                query=query,
                result=result.to_dict(),
                metadata={"session_id": session_id, "agent": "ResearchAgent"},
            )
        return ResearchResponse.from_research_result(
            result, session_id=session_id, cost_usd=cost_usd
        )

    try:
        response = await _run_in_executor(_execute)
    except AgentValidationError as exc:
        logger.warning(
            "POST /research — validation error",
            extra={"run_id": run_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except AgentTimeoutError as exc:
        logger.error(
            "POST /research — timeout", extra={"run_id": run_id, "error": str(exc)}
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The research agent exceeded its step budget.",
        ) from exc
    except AgentBudgetExceededError as exc:
        logger.warning(
            "POST /research — budget exceeded",
            extra={"run_id": run_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Run cost budget exceeded.",
        ) from exc
    except AgentExecutionError as exc:
        logger.error(
            "POST /research — execution error",
            extra={"run_id": run_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The research agent encountered an internal error.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "POST /research — unexpected error",
            extra={"run_id": run_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        ) from exc

    logger.info(
        "POST /research — completed",
        extra={
            "run_id": run_id,
            "session_id": session_id,
            "confidence": response.confidence,
            "findings_count": len(response.findings),
        },
    )
    return response
