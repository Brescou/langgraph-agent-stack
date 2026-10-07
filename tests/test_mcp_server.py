"""tests/test_mcp_server.py — Issue #92: domain packs as MCP tools."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("mcp")

from domain_packs.common.compliance import REGULATED_PACK_IDS  # noqa: E402
from pack_kernel.registry import PackRegistry  # noqa: E402


def _route_paths(app: Any) -> set[str]:
    """Collect mounted / route paths for snapshot-style assertions."""
    paths: set[str] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        if path:
            paths.add(path)
        # Mounted sub-apps (Starlette Mount)
        name = type(route).__name__
        if name == "Mount" and path:
            paths.add(path)
    return paths


@pytest.fixture()
def mcp_disabled_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    """App with MCP_SERVER_ENABLED=false (default)."""
    from core.config import get_settings

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("MCP_SERVER_ENABLED", "false")
    monkeypatch.setenv("REGULATED_PACKS_ENABLED", "false")
    get_settings.cache_clear()

    import api.state as api_state

    api_state.shared_llm = None
    api_state.shared_checkpointer = None

    from api.main import app

    with TestClient(app) as client:
        yield client

    api_state.shared_llm = None
    api_state.shared_checkpointer = None
    get_settings.cache_clear()


@pytest.fixture()
def mcp_enabled_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    """App with MCP_SERVER_ENABLED=true and mock LLM."""
    from core.config import get_settings

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("MCP_SERVER_ENABLED", "true")
    monkeypatch.setenv("REGULATED_PACKS_ENABLED", "false")
    get_settings.cache_clear()

    import api.state as api_state

    api_state.shared_llm = None
    api_state.shared_checkpointer = None

    from api.main import app

    with TestClient(app) as client:
        yield client

    api_state.shared_llm = None
    api_state.shared_checkpointer = None
    get_settings.cache_clear()


@asynccontextmanager
async def _mcp_session_with_app_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    regulated_packs_enabled: bool = False,
    extra_env: dict[str, str] | None = None,
) -> AsyncIterator[tuple[Any, Any]]:
    """Start FastAPI lifespan (LLM/executor) and an in-process MCP client session."""
    from core.config import get_settings

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("MCP_SERVER_ENABLED", "true")
    monkeypatch.setenv(
        "REGULATED_PACKS_ENABLED", "true" if regulated_packs_enabled else "false"
    )
    if extra_env:
        for key, value in extra_env.items():
            monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    from core.cost import _reset_effective_table

    _reset_effective_table()

    import api.state as api_state

    api_state.shared_llm = None
    api_state.shared_checkpointer = None

    # mcp 2.x dropped create_connected_server_and_client_session; Client now
    # takes an MCPServer directly and wires the in-memory streams itself.
    from mcp import Client

    from api.main import app
    from api.mcp_server import build_mcp_server

    with TestClient(app):
        mcp = build_mcp_server()
        async with Client(mcp) as session:
            yield session, mcp

    api_state.shared_llm = None
    api_state.shared_checkpointer = None
    get_settings.cache_clear()


class TestMcpFlagOff:
    def test_mcp_not_mounted_when_disabled(
        self, mcp_disabled_client: TestClient
    ) -> None:
        from api.main import app

        paths = _route_paths(app)
        assert "/mcp" not in paths
        # Sanity: core routes still served. Checked over HTTP because
        # fastapi >= 0.141 keeps included routers out of app.routes.
        assert mcp_disabled_client.get("/health").status_code == 200


class TestMcpFlagOnMount:
    def test_mcp_mounted_when_enabled(self, mcp_enabled_client: TestClient) -> None:
        from api.main import app

        paths = _route_paths(app)
        assert "/mcp" in paths

    def test_restart_drops_the_mount_when_mcp_is_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lifespan with the flag off must not keep a /mcp mount from an earlier one.

        Regression for #169: Starlette never unmounts, and the suite reuses one
        app, so ``test_mcp_not_mounted_when_disabled`` failed whenever a
        flag-on startup had run first.
        """
        from core.config import get_settings

        monkeypatch.setenv("LLM_PROVIDER", "mock")
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.setenv("REGULATED_PACKS_ENABLED", "false")
        monkeypatch.setenv("MCP_SERVER_ENABLED", "true")
        get_settings.cache_clear()

        from api.main import app

        try:
            with TestClient(app):
                assert "/mcp" in _route_paths(app)

            monkeypatch.setenv("MCP_SERVER_ENABLED", "false")
            get_settings.cache_clear()
            with TestClient(app) as client:
                assert "/mcp" not in _route_paths(app)
                assert "/metrics" in _route_paths(app)
                assert client.get("/health").status_code == 200
            assert "/mcp" not in _route_paths(app)
        finally:
            get_settings.cache_clear()

    def test_restart_with_mcp_enabled_does_not_stack_mounts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from core.config import get_settings

        monkeypatch.setenv("LLM_PROVIDER", "mock")
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.setenv("REGULATED_PACKS_ENABLED", "false")
        monkeypatch.setenv("MCP_SERVER_ENABLED", "true")
        get_settings.cache_clear()

        from api.main import app

        def _mcp_mounts() -> list[Any]:
            return [
                route for route in app.routes if getattr(route, "path", None) == "/mcp"
            ]

        try:
            with TestClient(app):
                pass
            with TestClient(app):
                assert len(_mcp_mounts()) == 1
        finally:
            get_settings.cache_clear()


class TestMcpAuth:
    """`/mcp` is not auth-exempt — same Bearer gate as REST when API_KEY is set."""

    def test_mcp_requires_bearer_when_api_key_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from core.config import get_settings

        monkeypatch.setenv("LLM_PROVIDER", "mock")
        monkeypatch.setenv("MCP_SERVER_ENABLED", "true")
        monkeypatch.setenv("REGULATED_PACKS_ENABLED", "false")
        monkeypatch.setenv("API_KEY", "mcp-secret-token")
        get_settings.cache_clear()

        import api.state as api_state

        api_state.shared_llm = None
        api_state.shared_checkpointer = None

        from api.main import app

        try:
            with TestClient(app, raise_server_exceptions=False) as client:
                missing = client.post("/mcp", json={})
                assert missing.status_code == 401
                assert "Bearer" in missing.json()["detail"]

                wrong = client.post(
                    "/mcp",
                    json={},
                    headers={"Authorization": "Bearer wrong-token"},
                )
                assert wrong.status_code == 401

                # Auth passes; MCP may still reject for Accept headers (406) —
                # anything other than 401 proves the mount is behind the gate.
                ok = client.post(
                    "/mcp",
                    json={},
                    headers={
                        "Authorization": "Bearer mcp-secret-token",
                        "Accept": "application/json, text/event-stream",
                    },
                )
                assert ok.status_code != 401
        finally:
            api_state.shared_llm = None
            api_state.shared_checkpointer = None
            get_settings.cache_clear()


class TestMcpTools:
    @pytest.mark.asyncio
    async def test_list_tools_matches_non_gated_packs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async with _mcp_session_with_app_runtime(monkeypatch) as (session, _mcp):
            listed = await session.list_tools()
            names = sorted(t.name for t in listed.tools)
            expected = sorted(
                pid
                for pid in PackRegistry.list_packs()
                if pid not in REGULATED_PACK_IDS
            )
            assert names == expected

    @pytest.mark.asyncio
    async def test_regulated_pack_absent_when_gated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async with _mcp_session_with_app_runtime(
            monkeypatch, regulated_packs_enabled=False
        ) as (session, _mcp):
            listed = await session.list_tools()
            names = {t.name for t in listed.tools}
            for pack_id in REGULATED_PACK_IDS:
                assert pack_id not in names

    @pytest.mark.asyncio
    async def test_summariser_end_to_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async with _mcp_session_with_app_runtime(monkeypatch) as (session, _mcp):
            result = await session.call_tool(
                "summariser",
                {"text": "Hello world. This is a short paragraph for summarising."},
            )
            assert result.is_error is False, result.content
            # Structured or text JSON payload
            payload: dict[str, Any]
            if result.structured_content:
                payload = dict(result.structured_content)
            else:
                text = result.content[0].text  # type: ignore[index]
                payload = json.loads(text)
            assert "bullets" in payload
            assert "original_length" in payload
            # Schema-valid against pack output
            _input_model, output_model = PackRegistry.get_schemas("summariser")
            output_model.model_validate(payload)

    @pytest.mark.asyncio
    async def test_tool_input_schema_matches_pack(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async with _mcp_session_with_app_runtime(monkeypatch) as (session, _mcp):
            listed = await session.list_tools()
            tool = next(t for t in listed.tools if t.name == "summariser")
            input_model, _ = PackRegistry.get_schemas("summariser")
            expected = input_model.model_json_schema()
            actual = tool.input_schema
            assert actual.get("required") == expected.get("required")
            assert set(actual.get("properties", {})) == set(
                expected.get("properties", {})
            )
            # Spot-check Field constraint preserved from Pydantic schema
            text_schema = actual["properties"]["text"]
            assert text_schema.get("type") == "string"
            if "minLength" in expected["properties"]["text"]:
                assert (
                    text_schema.get("minLength")
                    == expected["properties"]["text"]["minLength"]
                )

    @pytest.mark.asyncio
    async def test_field_constraint_violation_raises_invalid_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Field constraint the tool signature cannot express must raise INVALID_PARAMS.

        ``_make_tool_callable`` copies only annotations into ``__signature__``,
        so MCPServer's own argument validation sees ``bullet_count: int`` and
        lets 99 through. The pack's own ``model_validate`` then rejects it
        against ``maximum: 10``, and that is the branch under test: it is the
        only place our ``MCPError`` is constructed.

        Worth a test of its own because the branch was broken across the mcp
        2.x migration and nothing caught it. ``MCPError`` stopped accepting an
        ``ErrorData`` and now takes ``code``/``message``, so the old call raised
        ``TypeError``, which MCPServer masked as a generic
        ``Error executing tool summariser`` with ``is_error=True``. A missing
        required field does not reach here: MCPServer rejects that first.
        """
        from mcp.shared.exceptions import MCPError
        from mcp.types import INVALID_PARAMS

        async with _mcp_session_with_app_runtime(monkeypatch) as (session, _mcp):
            input_model, _ = PackRegistry.get_schemas("summariser")
            schema = input_model.model_json_schema()
            assert schema["properties"]["bullet_count"]["maximum"] == 10, (
                "test assumes summariser caps bullet_count at 10"
            )
            with pytest.raises(MCPError) as excinfo:
                await session.call_tool(
                    "summariser",
                    {"text": "Some valid text here.", "bullet_count": 99},
                )
            assert excinfo.value.code == INVALID_PARAMS
            assert "bullet_count" in excinfo.value.message

    @pytest.mark.asyncio
    async def test_budget_exceeded_maps_to_tool_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from core.config import get_settings
        from core.cost import _reset_effective_table
        from core.mock_llm import MOCK_MODEL_ID

        cost_file = tmp_path / "mock_cost.json"
        cost_file.write_text(
            json.dumps({MOCK_MODEL_ID: [10.0, 10.0]}),
            encoding="utf-8",
        )
        try:
            async with _mcp_session_with_app_runtime(
                monkeypatch,
                extra_env={
                    "PACK_DEFAULT_BUDGET_USD": "0.001",
                    "LLM_COST_TABLE_PATH": str(cost_file),
                },
            ) as (session, _mcp):
                # meeting_prep uses CostTracker (same pattern as test_mock_cost_usd).
                result = await session.call_tool(
                    "meeting_prep",
                    {
                        "company": "Acme",
                        "person": "Jane",
                        "meeting_goal": "discovery",
                    },
                )
                assert result.is_error is True, result.content
                text = result.content[0].text  # type: ignore[index]
                assert "budget" in text.lower() or "0.001" in text or "USD" in text
        finally:
            get_settings.cache_clear()
            _reset_effective_table()
