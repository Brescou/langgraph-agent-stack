"""api/pack_execution.py — Shared typed-pack run path for REST and MCP.

Both ``api/router_factory.py`` and ``api/mcp_server.py`` call
:func:`execute_typed_pack_run` so budget, compliance, validation, and
session semantics stay in one place.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from time import monotonic
from typing import Any

from fastapi import HTTPException, status

import api.state as state
from agents.base_agent import (
    AgentAuthenticationError,
    AgentBudgetExceededError,
    AgentExecutionError,
    AgentTimeoutError,
    AgentValidationError,
    make_auth_error,
)
from agents.llm_retry import find_auth_cause
from api.dependencies import (
    pack_primary_text,
    pack_runtime_kwargs,
    validate_pack_body_fields,
    validate_pack_query,
)
from core.config import get_settings
from core.observability import set_span_attributes
from core.security import (
    IdempotencyConflictError,
    IdempotencyStatus,
    IdempotencyStore,
)
from pack_kernel.registry import PackRegistry

logger = logging.getLogger(__name__)

#: Maximum time (seconds) granted to history persistence before giving up.
SAVE_RUN_TIMEOUT_SECONDS = 5.0

#: Detail message returned when a session already has a run in flight.
SESSION_IN_FLIGHT_DETAIL = "A run is already in progress for this session."

#: Polling interval (seconds) while waiting for an in-flight idempotent
#: request to complete.
IDEMPOTENCY_POLL_INTERVAL_SECONDS = 0.1


@dataclass(frozen=True, slots=True)
class PackRunResult:
    """Outcome of a successful typed pack run."""

    serialized: Any
    used_version: str
    run_id: str


def pack_has_structured_input(pack_cls: type) -> bool:
    """Return True when the pack defines ``run_from_input`` on the class body."""
    return "run_from_input" in pack_cls.__dict__


def pack_has_structured_stream(pack_cls: type) -> bool:
    """Return True when the pack defines ``stream_events_from_input``."""
    return "stream_events_from_input" in pack_cls.__dict__


def invoke_pack_run(pack_cls: type, pipeline: Any, body: Any) -> Any:
    """Invoke sync pack run (structured input or free-text)."""
    if pack_has_structured_input(pack_cls):
        return pipeline.run_from_input(body)
    return pipeline.run(pack_primary_text(body))


def serialize_pack_result(
    result: Any, output_model: type, cost_usd: float | None
) -> Any:
    """Serialize a pack result into the pack's output schema / dict.

    ``cost_usd`` is only injected when the pack's ``output_schema`` actually
    declares that field. Plugin packs are *required* to use strict schemas
    (``extra="forbid"``, see ``pack_kernel/plugins.py``), so injecting an
    undeclared field would make every successful run of a third-party pack
    fail response validation with HTTP 500.
    """
    if hasattr(output_model, "from_analysis_report"):
        return output_model.from_analysis_report(result, cost_usd=cost_usd)
    if hasattr(output_model, "from_research_result"):
        return output_model.from_research_result(result, cost_usd=cost_usd)
    if hasattr(output_model, "from_summary_result"):
        return output_model.from_summary_result(result, cost_usd=cost_usd)
    if hasattr(result, "model_dump"):
        data = result.model_dump()
        declares_cost = "cost_usd" in getattr(output_model, "model_fields", {})
        if cost_usd is not None and declares_cost:
            data["cost_usd"] = cost_usd
        return data
    return result


async def run_in_executor(fn: Any, *args: Any) -> Any:
    """Execute a blocking callable in the application thread pool."""
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


async def save_run_best_effort(
    *,
    run_id: str,
    query: str,
    result: dict[str, Any],
    metadata: dict[str, Any],
    session_id: str | None = None,
) -> None:
    """Persist run history without ever failing the request."""
    if state.shared_memory is None:
        return
    save_fn = functools.partial(
        state.shared_memory.save_run,
        run_id=run_id,
        query=query,
        result=result,
        metadata=metadata,
    )
    try:
        await asyncio.wait_for(
            run_in_executor(save_fn), timeout=SAVE_RUN_TIMEOUT_SECONDS
        )
    except TimeoutError:
        logger.warning(
            "save_run — timed out, run history not persisted",
            extra={"run_id": run_id, "session_id": session_id},
        )
    except Exception as exc:
        logger.warning(
            "save_run — failed, run history not persisted",
            extra={"run_id": run_id, "session_id": session_id, "error": str(exc)},
        )


def pack_requires_human_review(pack_id: str) -> bool:
    """True when the pack's policy mandates a human review of every output."""
    from control_plane import PolicyRegistry

    policy = PolicyRegistry.get(pack_id)
    return policy is not None and policy.human_review_required


async def create_review_best_effort(
    *,
    run_id: str,
    pack_id: str,
    session_id: str | None,
    result_payload: dict[str, Any],
) -> None:
    """Queue a pending human review for a regulated run, without ever failing it."""
    if state.review_store is None or not pack_requires_human_review(pack_id):
        return
    from core.review_store import summarize_output

    create_fn = functools.partial(
        state.review_store.create,
        run_id=run_id,
        pack_id=pack_id,
        session_id=session_id,
        output_summary=summarize_output(result_payload),
    )
    try:
        await asyncio.wait_for(
            run_in_executor(create_fn), timeout=SAVE_RUN_TIMEOUT_SECONDS
        )
    except TimeoutError:
        logger.warning(
            "review create — timed out, pending review not queued",
            extra={"run_id": run_id, "pack_id": pack_id},
        )
    except Exception as exc:
        logger.warning(
            "review create — failed, pending review not queued",
            extra={"run_id": run_id, "pack_id": pack_id, "error": str(exc)},
        )


async def wait_for_completed_idempotency_record(
    store: IdempotencyStore,
    key: str,
    timeout_seconds: float,
) -> PackRunResult | None:
    """Wait until an in-flight idempotent request completes."""

    deadline = monotonic() + timeout_seconds

    while monotonic() < deadline:
        record = store.get(key)

        if record is None:
            return None

        if record.status is IdempotencyStatus.COMPLETED:
            if isinstance(record.response, PackRunResult):
                return record.response

            if isinstance(record.response, Mapping):
                try:
                    return PackRunResult(
                        serialized=record.response["serialized"],
                        used_version=record.response["used_version"],
                        run_id=record.response["run_id"],
                    )
                except (KeyError, TypeError):
                    pass

            raise RuntimeError("Completed idempotency record has an invalid response.")

        await asyncio.sleep(IDEMPOTENCY_POLL_INTERVAL_SECONDS)

    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=(
            "Another request with the same Idempotency-Key is still being processed."
        ),
    )


async def execute_typed_pack_run(
    pack_id: str,
    body: Any,
    *,
    affinity_key: str | None = None,
    requested_version: str | None = None,
    idempotency_key: str | None = None,
) -> PackRunResult:
    """Run a registered pack with the same kernel path as ``POST /packs/{id}/run``.

    Raises:
        HTTPException: Mapped status codes (402 budget, 403 compliance, 409
            session conflict, 422 validation, 5xx agent failures, …).
    """
    if state.shutting_down.is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is shutting down.",
        )

    run_id = str(uuid.uuid4())
    session_id = getattr(body, "session_id", None) or None
    set_span_attributes({"pack_id": pack_id, "run_id": run_id})

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
            affinity_key=affinity_key,
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

    # Prefer the real registry class schema (not a MagicMock from tests that
    # patch PackRegistry.get). Match by identity when possible.
    output_model = next(
        (
            pv.pack_cls.output_schema
            for pv in PackRegistry._get_versions(pack_id)
            if pv.pack_cls is pack_cls_to_use
        ),
        PackRegistry._get_versions(pack_id)[0].pack_cls.output_schema,
    )

    validate_pack_body_fields(pack_id, body)
    query = validate_pack_query(pack_id, pack_primary_text(body))

    body_hash = hashlib.sha256(body.model_dump_json().encode("utf-8")).hexdigest()

    idempotency_store = state.get_shared_idempotency_store()

    settings = get_settings()

    if idempotency_key is not None:
        while True:
            try:
                reserved = idempotency_store.reserve(
                    idempotency_key,
                    body_hash,
                )
            except IdempotencyConflictError as exc:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=str(exc),
                ) from exc

            if reserved:
                break

            result = await wait_for_completed_idempotency_record(
                idempotency_store,
                idempotency_key,
                settings.stream_timeout_seconds,
            )

            if result is not None:
                return result

    stored = False
    session_acquired = False

    try:
        from domain_packs.common.compliance import assert_regulated_pack_runtime_enabled

        try:
            assert_regulated_pack_runtime_enabled(
                pack_id, regulated_packs_enabled=settings.regulated_packs_enabled
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)
            ) from exc

        def _execute() -> tuple[Any, dict[str, Any]]:
            with pack_cls_to_use(
                run_id=run_id,
                llm=state.get_shared_llm(),
                checkpointer=state.get_shared_checkpointer(),
                **pack_runtime_kwargs(pack_cls_to_use),
            ) as pipeline:
                result = invoke_pack_run(pack_cls_to_use, pipeline, body)
                cost_usd = getattr(pipeline, "cost_usd", None)
                result_payload = (
                    result.to_dict()
                    if hasattr(result, "to_dict")
                    else result.model_dump()
                    if hasattr(result, "model_dump")
                    else {}
                )
                serialized = serialize_pack_result(result, output_model, cost_usd)
                return serialized, result_payload

        if session_id:
            if not state.try_acquire_session(session_id):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=SESSION_IN_FLIGHT_DETAIL,
                )

            session_acquired = True

        try:
            serialized, result_payload = await run_in_executor(_execute)
        except AgentBudgetExceededError as exc:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED, detail=str(exc)
            ) from exc
        except AgentTimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)
            ) from exc
        except AgentAuthenticationError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
            ) from exc
        except (AgentExecutionError, AgentValidationError) as exc:
            auth_cause = find_auth_cause(exc)
            if auth_cause is not None:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=str(
                        make_auth_error(pack_id, settings.llm_provider, auth_cause)
                    ),
                ) from exc
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc

        await save_run_best_effort(
            run_id=run_id,
            query=query,
            result=result_payload,
            metadata={
                "pack_id": pack_id,
                "pack_version": used_version,
                **({"session_id": session_id} if session_id else {}),
            },
            session_id=session_id,
        )
        await create_review_best_effort(
            run_id=run_id,
            pack_id=pack_id,
            session_id=session_id,
            result_payload=result_payload,
        )
        result = PackRunResult(
            serialized=serialized,
            used_version=used_version,
            run_id=run_id,
        )

        if idempotency_key is not None:
            idempotency_store.store_result(
                idempotency_key,
                result,
            )
            stored = True

        return result

    finally:
        if idempotency_key is not None and not stored:
            idempotency_store.release(idempotency_key)

        if session_acquired and session_id is not None:
            state.release_session(session_id)
