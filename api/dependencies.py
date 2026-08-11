"""api/dependencies.py — FastAPI dependency functions and request helpers."""

from __future__ import annotations

import inspect
import logging
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status

import api.state as state
from control_plane.enforce import effective_budget_usd
from core.config import Settings, get_settings
from core.security import (
    rate_limit_client_key,
    resolve_client_ip,
    verify_bearer_token,
)

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
    """Rate-limit bucket key: per-token when multi-tenant, per-IP otherwise.

    With 2+ configured Bearer secrets (multi-tenant), each token gets its own
    bucket (``token:<sha256[:16]>``). With a single shared key, token hashing
    would collapse every caller into one global bucket, so the limit stays
    scoped per client IP (see ``rate_limit_client_key`` docstring).
    """
    settings = get_settings()
    api_keys = settings.resolved_api_keys
    return rate_limit_client_key(
        _request_peer_host(request),
        request.headers,
        trust_proxy=settings.trust_proxy_headers,
        forwarded_allow_ips=settings.forwarded_allow_ips,
        # rate_limit_client_key only uses api_key as an "auth is configured"
        # gate (it hashes the presented token, not this value) — pass the
        # first resolved key so API_KEYS-only deployments are covered too.
        api_key=api_keys[0] if api_keys else None,
        rate_limit_per_token=len(api_keys) > 1,
    )


def _rate_limit_route_path(request: Request) -> str:
    """Low-cardinality endpoint path used to scope the rate-limit key.

    Prefers the route template (e.g. ``/sessions/{session_id}/history``) when
    routing has already resolved the route; falls back to the raw request
    path, which is acceptable given the strict patterns on path parameters.
    """
    route = request.scope.get("route")
    path_format = getattr(route, "path_format", None)
    if path_format:
        return str(path_format)
    return request.url.path


def _rate_limit_endpoint_key(request: Request) -> str:
    """Rate-limit bucket key scoped per client AND per endpoint path."""
    return f"{_rate_limit_key(request)}:{_rate_limit_route_path(request)}"


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def verify_api_key(request: Request) -> None:
    """Validate the Bearer secret when API_KEY and/or API_KEYS is set.

    The token must match ANY configured key (multi-tenant: each tenant has
    its own key; rotation: old + new keys coexist during the transition).
    Also enforced globally via auth_middleware; this dependency documents
    the contract in OpenAPI for pack routes.
    """
    api_keys = get_settings().resolved_api_keys
    if not api_keys:
        return
    auth_header = request.headers.get("Authorization", "")
    token = (
        auth_header.removeprefix("Bearer ").strip()
        if auth_header.startswith("Bearer ")
        else ""
    )
    if not verify_bearer_token(token, api_keys):
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


def pack_runtime_kwargs(
    pack_cls: type,
    *,
    pack_id: str | None = None,
    pack_version: str | None = None,
) -> dict[str, Any]:
    """Extra constructor kwargs: policy budget, optional connector, pack identity."""
    kwargs: dict[str, Any] = {}
    params = inspect.signature(pack_cls.__init__).parameters
    resolved_pack_id = pack_id or getattr(pack_cls, "pack_id", None)
    if resolved_pack_id:
        budget = effective_budget_usd(resolved_pack_id, get_settings())
        if budget is not None:
            kwargs["budget_usd"] = budget
    if pack_id is not None and "pack_id" in params:
        kwargs["pack_id"] = pack_id
    if pack_version is not None and "pack_version" in params:
        kwargs["pack_version"] = pack_version
    if state.shared_connector is not None and "connector" in params:
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
