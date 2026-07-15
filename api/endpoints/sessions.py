"""api/endpoints/sessions.py — Session run history."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter
from fastapi import Path as FastAPIPath

import api.state as state
from api.models import HistoryEntry, HistoryResponse

router = APIRouter(tags=["Sessions"])
logger = logging.getLogger(__name__)


@router.get(
    "/sessions/{session_id}/history",
    response_model=HistoryResponse,
    status_code=200,
    summary="Retrieve run history for a session",
    response_description="Ordered list of run records associated with the given session ID.",
)
async def get_session_history(
    session_id: Annotated[
        str,
        FastAPIPath(
            min_length=1,
            max_length=200,
            pattern=r"^[a-zA-Z0-9_-]+$",
            description="Session identifier",
        ),
    ],
) -> HistoryResponse:
    """Return all run records associated with session_id, ordered newest-first."""
    mem = state.shared_memory
    if mem is None:
        return HistoryResponse(session_id=session_id, entries=[], total=0)

    from api.pack_execution import run_in_executor

    runs = await run_in_executor(mem.list_runs_by_session, session_id)
    entries = [
        HistoryEntry(
            run_id=r["run_id"],
            query=r["query"],
            result_summary=str(r.get("result", {}) or "")[:200],
            created_at=r.get("created_at", ""),
            metadata=r.get("metadata", {}),
        )
        for r in runs
    ]
    logger.info(
        "GET /sessions/%s/history — %d entries returned", session_id, len(entries)
    )
    return HistoryResponse(session_id=session_id, entries=entries, total=len(entries))
