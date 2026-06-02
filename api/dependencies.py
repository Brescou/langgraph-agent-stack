"""api/dependencies.py — FastAPI dependency functions and request helpers."""

from __future__ import annotations

import hmac
import inspect
import logging
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status

import api.state as state
from control_plane.enforce import effective_budget_usd
from core.config import Settings, get_settings
from core.security import rate_limit_client_key, resolve_client_ip

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request helpers (IP / rate-limit key)
# ---------------------------------------------------------------------------


def _request_peer_host(request: Request) -> str | None:
    return request.client.host if request.client else None


def _request_client_ip(request: Request) -> str:
    settings = get_settings()
    return resolve_client_ip(
        _request_peer_host(request),
        request.headers,
        trust_proxy=settings.trust_proxy_headers,
        forwarded_allow_ips=settings.forwarded_allow_ips,
    )


def _rate_limit_key(request: Request) -> str:
    settings = get_settings()
    return rate_limit_client_key(
        _request_peer_host(request),
        request.headers,
        trust_proxy=settings.trust_proxy_headers,
        forwarded_allow_ips=settings.forwarded_allow_ips,
        api_key=settings.api_key,
    )


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def verify_api_key(request: Request) -> None:
    """Validate the shared Bearer secret when API_KEY is set.

    Also enforced globally via auth_middleware; this dependency documents
    the contract in OpenAPI for pack routes.
    """
    api_key = get_settings().api_key
    if api_key is None:
        return
    auth_header = request.headers.get("Authorization", "")
    token = (
        auth_header.removeprefix("Bearer ").strip()
        if auth_header.startswith("Bearer ")
        else ""
    )
    if not token or not hmac.compare_digest(token, api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Pack helpers
# ---------------------------------------------------------------------------


def get_legacy_pack_cls(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> type[Any]:
    """FastAPI dependency: pack class for legacy POST /run and POST /run/stream."""
    from pack_kernel.registry import PackRegistry

    try:
        return PackRegistry.get(
            settings.default_pack_id,
            affinity_key=_rate_limit_key(request),
        )
    except KeyError:
        from domain_packs.research.research_analysis.pack import ResearchAnalysisPack

        return state.active_pack_cls or ResearchAnalysisPack


def pack_runtime_kwargs(pack_cls: type) -> dict[str, Any]:
    """Extra constructor kwargs: policy budget and optional connector."""
    kwargs: dict[str, Any] = {}
    pack_id = getattr(pack_cls, "pack_id", None)
    if pack_id:
        budget = effective_budget_usd(pack_id, get_settings())
        if budget is not None:
            kwargs["budget_usd"] = budget
    if (
        state.shared_connector is not None
        and "connector" in inspect.signature(pack_cls.__init__).parameters
    ):
        kwargs["connector"] = state.shared_connector
    return kwargs


def pack_primary_text(body: Any) -> str:
    """Extract the main free-text field from a typed pack request body."""
    for field in (
        "query",
        "text",
        "company",
        "topic",
        "ticket_subject",
        "question",
        "role_title",
    ):
        if hasattr(body, field):
            value = getattr(body, field)
            if value:
                return str(value)
    return str(body)[:500]


def validate_pack_body_fields(pack_id: str, body: Any) -> None:
    """Content-safety scan on all string fields of a typed pack body."""
    from control_plane.enforce import validate_pack_body

    try:
        validate_pack_body(body, pack_id, state.input_validator)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


def validate_pack_query(pack_id: str, raw_query: str) -> str:
    """Validate query text using pack policy constraints and global sanitizer."""
    from control_plane.enforce import validate_query_for_pack

    try:
        return validate_query_for_pack(raw_query, pack_id, state.input_validator)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
