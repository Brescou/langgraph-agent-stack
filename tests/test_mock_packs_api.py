"""tests/test_mock_packs_api.py — Parametrised API sweep for all built-in packs in mock mode."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from domain_packs.common.compliance import REGULATED_PACK_IDS
from pack_kernel.builtin_packs import all_builtin_pack_classes
from pack_kernel.base_pack import BaseDomainPack

_BUILTIN_PACK_CLASSES = all_builtin_pack_classes()
_BUILTIN_PACK_IDS = [pack_cls.pack_id for pack_cls in _BUILTIN_PACK_CLASSES]


@pytest.fixture()
def mock_packs_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    """Real app with LLM_PROVIDER=mock and no patched checkpointer."""
    from core.config import get_settings

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("API_KEY", raising=False)
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


def _pack_request_body(pack_cls: type[BaseDomainPack]) -> dict[str, Any]:
    from core.mock_llm import minimal_valid_input

    return minimal_valid_input(pack_cls.input_schema)


@pytest.mark.parametrize("pack_cls", _BUILTIN_PACK_CLASSES, ids=_BUILTIN_PACK_IDS)
def test_pack_run_via_api_mock(
    mock_packs_client: TestClient, pack_cls: type[BaseDomainPack]
) -> None:
    """Each built-in pack returns 200 in mock mode, or 403 when regulated."""
    pack_id = pack_cls.pack_id
    body = _pack_request_body(pack_cls)
    response = mock_packs_client.post(f"/packs/{pack_id}/run", json=body)

    if pack_id in REGULATED_PACK_IDS:
        assert response.status_code == 403
        assert "REGULATED_PACKS_ENABLED" in response.json()["detail"]
        return

    assert response.status_code == 200, response.text
    data = response.json()
    assert isinstance(data, dict)
    assert data
