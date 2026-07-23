"""api/lifespan.py — FastAPI application startup and shutdown lifecycle."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

import api.state as state
from core.config import Settings, get_settings
from core.memory import cleanup_checkpointer_async, create_run_history
from core.observability import init_tracing, server_shutting_down
from core.review_store import create_review_store
from core.security import create_rate_limiter, create_session_registry, create_idempotency_store
from pack_kernel.builtin_packs import register_builtin_packs
from pack_kernel.registry import PackRegistry

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

register_builtin_packs()


async def _init_llm_and_checkpointer(settings: Settings) -> None:
    """Create the shared LLM and async checkpointer at startup.

    On failure the globals are set to None and a warning is logged.
    """
    from core.llm import get_llm
    from core.memory import init_checkpointer

    try:
        state.shared_llm = get_llm(settings.llm_config)
        state.shared_checkpointer = await init_checkpointer(settings)
        logger.info("LLM provider '%s' configured successfully", settings.llm_provider)
    except (ImportError, ValueError) as exc:
        logger.warning("LLM configuration warning: %s", exc)
        state.shared_llm = None
        state.shared_checkpointer = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown resources.

    Startup:
        * Records the process start time for uptime reporting.
        * Pre-warms a ThreadPoolExecutor used by all blocking agent calls.
        * Initialises LLM, checkpointer, rate limiter, and pack routers.

    Shutdown:
        * Gracefully shuts down the thread pool, waiting for in-flight tasks.
    """
    state.start_time = time.monotonic()
    settings = get_settings()

    if state.rate_limiter is None:
        state.rate_limiter = create_rate_limiter(
            backend=settings.rate_limit_backend,
            redis_url=settings.redis_url,
        )

    if state.session_registry is None:
        state.session_registry = create_session_registry(
            backend=settings.session_registry_backend,
            redis_url=settings.redis_url,
            ttl_seconds=settings.session_lock_ttl_seconds,
        )

    if state.idempotency_store is None:
        state.idempotency_store = create_idempotency_store(
            backend=settings.idempotency_backend,
            redis_url=settings.redis_url,
            ttl_seconds=settings.idempotency_ttl_seconds,
        )

    if state.review_store is None:
        state.review_store = create_review_store(
            backend=settings.review_store_backend,
            sqlite_path=settings.review_store_path,
        )

    if settings.memory_backend.value == "postgres" and not settings.postgres_url:
        raise RuntimeError(
            "POSTGRES_URL is required when MEMORY_BACKEND=postgres. "
            "Set the POSTGRES_URL environment variable."
        )
    if settings.memory_backend.value == "redis" and not settings.redis_url:
        raise RuntimeError(
            "REDIS_URL is required when MEMORY_BACKEND=redis. "
            "Set the REDIS_URL environment variable."
        )

    state.executor = ThreadPoolExecutor(
        max_workers=settings.thread_pool_max_workers,
        thread_name_prefix="agent-worker",
    )

    init_tracing()
    await _init_llm_and_checkpointer(settings)

    try:
        state.active_pack_cls = PackRegistry.get(settings.default_pack_id)
        logger.info(
            "Active domain pack resolved",
            extra={"pack_id": settings.default_pack_id},
        )
    except KeyError as exc:
        raise RuntimeError(
            f"DEFAULT_PACK_ID '{settings.default_pack_id}' is not registered. "
            "Check pack_kernel/builtin_packs.py."
        ) from exc

    from pack_kernel.plugins import register_plugin_packs

    plugin_pack_ids = register_plugin_packs(
        enabled=settings.pack_plugins_enabled,
        allowlist=settings.resolved_pack_plugins_allowlist,
    )
    if plugin_pack_ids:
        logger.info(
            "Plugin packs loaded",
            extra={"pack_ids": plugin_pack_ids},
        )

    from connectors.resolver import resolve_connector

    state.shared_connector = resolve_connector(settings)
    if state.shared_connector is not None:
        logger.info(
            "Retrieval connector enabled",
            extra={"connector_id": settings.connector_id},
        )

    # Wire per-pack routers — guard against duplicate registration on test reuse
    from api.router_factory import build_pack_router

    _existing_prefixes = {
        getattr(r, "path", "").split("/{")[0] for r in app.routes if hasattr(r, "path")
    }
    for pack_id in PackRegistry.list_packs():
        expected_prefix = f"/packs/{pack_id}/run"
        if expected_prefix in _existing_prefixes:
            logger.debug(
                "Pack router already registered — skipping",
                extra={"pack_id": pack_id},
            )
            continue
        pack_cls = PackRegistry.get(pack_id)
        in_schema, out_schema = PackRegistry.get_schemas(pack_id)
        app.include_router(build_pack_router(pack_id, pack_cls, in_schema, out_schema))
        logger.info("Pack router registered", extra={"pack_id": pack_id})

    state.shared_memory = create_run_history(settings)

    mcp_server = None
    if settings.mcp_server_enabled:
        from api.mcp_server import mount_mcp_server

        mcp_server = mount_mcp_server(app)

    logger.info(
        "API server starting up",
        extra={
            "version": state.APP_VERSION,
            "environment": settings.environment,
            "host": settings.api_host,
            "port": settings.api_port,
            "llm_provider": settings.llm_provider,
            "memory_backend": settings.memory_backend.value,
            "mcp_server_enabled": settings.mcp_server_enabled,
        },
    )

    state.shutting_down.clear()
    if server_shutting_down is not None:
        server_shutting_down.set(0)

    if mcp_server is not None:
        async with mcp_server.session_manager.run():
            yield  # Application is live here (MCP session manager active)
    else:
        yield  # Application is live here

    logger.info("API server shutting down — draining in-flight requests")
    state.shutting_down.set()
    if server_shutting_down is not None:
        server_shutting_down.set(1)
    if state.executor is not None:
        state.executor.shutdown(wait=True, cancel_futures=False)
    await cleanup_checkpointer_async()
    if state.shared_memory is not None:
        state.shared_memory.close()
    if state.review_store is not None:
        state.review_store.close()
    logger.info("Shutdown complete")
