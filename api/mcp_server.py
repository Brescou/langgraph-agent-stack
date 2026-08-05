"""api/mcp_server.py — Opt-in MCP server exposing one tool per domain pack.

Enabled only when ``MCP_SERVER_ENABLED=true``. Tools are generated from
``PackRegistry`` (default version only) and execute via
:func:`api.pack_execution.execute_typed_pack_run` — the same kernel path as
typed REST routes.

Follow-up (#92): per-call pack version selection (``X-Pack-Version`` equivalent)
is intentionally out of scope for this transport.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, cast

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined

from core.config import get_settings
from domain_packs.common.compliance import REGULATED_PACK_IDS
from pack_kernel.registry import PackRegistry

logger = logging.getLogger(__name__)

_MCP_IMPORT_HINT = "Install with: uv sync --extra mcp"


def _ensure_mcp_installed() -> None:
    """Raise a clear ImportError when the optional ``mcp`` extra is missing."""
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"MCP server support requires the mcp package. {_MCP_IMPORT_HINT}"
        ) from exc


def _http_detail(detail: Any) -> str:
    """Normalize FastAPI HTTPException.detail to a plain string."""
    if isinstance(detail, str):
        return detail
    return str(detail)


def _map_http_exception(exc: HTTPException) -> None:
    """Re-raise an HTTPException as the matching MCP protocol/tool error."""
    from mcp.server.fastmcp.exceptions import ToolError
    from mcp.shared.exceptions import McpError
    from mcp.types import INVALID_PARAMS, ErrorData

    detail = _http_detail(exc.detail)
    if exc.status_code == 422:
        raise McpError(ErrorData(code=INVALID_PARAMS, message=detail)) from exc
    # 402 budget / 403 compliance / other kernel errors → tool error with same text
    raise ToolError(detail) from exc


def _make_tool_callable(pack_id: str, input_model: type[BaseModel]) -> Any:
    """Build an async tool function with a flat signature from ``input_model``."""
    from mcp.shared.exceptions import McpError
    from mcp.types import INVALID_PARAMS, ErrorData

    params: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}
    for name, field in input_model.model_fields.items():
        annotations[name] = field.annotation
        if field.is_required():
            default: Any = inspect.Parameter.empty
        elif field.default is not PydanticUndefined:
            default = field.default
        else:
            default = None
        params.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=field.annotation,
            )
        )

    async def _impl(**kwargs: Any) -> Any:
        from api.pack_execution import execute_typed_pack_run

        try:
            body = input_model.model_validate(kwargs)
        except ValidationError as exc:
            raise McpError(ErrorData(code=INVALID_PARAMS, message=str(exc))) from exc

        # Default registry version only — see module docstring / issue #92.
        try:
            outcome = await execute_typed_pack_run(
                pack_id,
                body,
                affinity_key=None,
                requested_version=None,
                idempotency_key=None,
            )
        except HTTPException as exc:
            _map_http_exception(exc)
            raise  # pragma: no cover — _map_http_exception always raises

        serialized = outcome.serialized
        if hasattr(serialized, "model_dump"):
            return serialized.model_dump()
        return serialized

    _impl.__name__ = pack_id
    _impl.__doc__ = f"Run the {pack_id} domain pack."
    # FastMCP derives the call-time arg model from __signature__; assign via
    # Any because FunctionType does not declare that attribute for pyright.
    impl: Any = _impl
    impl.__signature__ = inspect.Signature(params, return_annotation=dict)
    impl.__annotations__ = {**annotations, "return": dict}
    return cast(Any, impl)


def _iter_mcp_pack_ids(*, regulated_packs_enabled: bool) -> list[str]:
    """Pack IDs exposed as MCP tools (omit gated regulated packs when disabled)."""
    pack_ids: list[str] = []
    for pack_id in PackRegistry.list_packs():
        if pack_id in REGULATED_PACK_IDS and not regulated_packs_enabled:
            continue
        pack_ids.append(pack_id)
    return pack_ids


def build_mcp_server() -> Any:
    """Create a ``FastMCP`` instance with one tool per eligible pack.

    Returns:
        Configured ``FastMCP`` server (caller owns lifespan / mounting).
    """
    _ensure_mcp_installed()
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.tools.base import Tool
    from mcp.server.transport_security import TransportSecuritySettings

    settings = get_settings()
    # DNS rebinding protection defaults to localhost-only hosts; that breaks
    # any real deployment Host header. Disable deliberately: this transport is
    # mounted on the FastAPI app, so Bearer API_KEY (+ existing middleware) is
    # the auth guardrail — not Host allowlisting.
    mcp_server = FastMCP(
        "langgraph-agent-stack",
        stateless_http=True,
        streamable_http_path="/",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )

    for pack_id in _iter_mcp_pack_ids(
        regulated_packs_enabled=settings.regulated_packs_enabled
    ):
        pack_cls = PackRegistry.get(pack_id)
        input_model, _output_model = PackRegistry.get_schemas(pack_id)
        description = getattr(pack_cls, "description", None) or pack_id
        fn = _make_tool_callable(pack_id, input_model)
        tool = Tool.from_function(fn, name=pack_id, description=description)
        # Override listed schema with the pack's Pydantic JSON Schema so Field
        # constraints (minLength, ge, …) match REST. FastMCP has no public
        # "register Tool with custom parameters" API in v1, so we write through
        # _tool_manager._tools (private). Risk is bounded by pyproject pin
        # mcp>=1.0,<2 — revisit if upgrading past that major.
        tool.parameters = input_model.model_json_schema()
        mcp_server._tool_manager._tools[tool.name] = tool
        logger.info("MCP tool registered", extra={"pack_id": pack_id})

    return mcp_server


def mount_mcp_server(app: FastAPI) -> Any:
    """Build tools and mount streamable HTTP at ``/mcp`` on ``app``.

    Returns:
        The ``FastMCP`` instance (its ``session_manager.run()`` must be entered
        during application lifespan).
    """
    mcp_server = build_mcp_server()
    app.mount("/mcp", mcp_server.streamable_http_app())
    logger.info("MCP streamable HTTP mounted at /mcp")
    return mcp_server
