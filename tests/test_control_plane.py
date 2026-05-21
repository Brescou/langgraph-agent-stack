"""tests/test_control_plane.py — Policy registry and enforcement."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from control_plane import PolicyRegistry, effective_budget_usd, validate_query_for_pack
from control_plane.policies import ExecutionConstraints, PackPolicy
from core.config import Settings, get_settings
from core.security import InputValidator


def test_default_policies_registered() -> None:
    policies = PolicyRegistry.list_policies()
    assert "research_analysis" in policies
    assert "research_only" in policies
    assert "summariser" in policies
    assert "analysis_only" in policies


def test_effective_budget_prefers_global_setting() -> None:
    with patch.dict(
        os.environ,
        {"LLM_PROVIDER": "mock", "PACK_DEFAULT_BUDGET_USD": "1.5"},
        clear=False,
    ):
        get_settings.cache_clear()
        settings = Settings()
    assert effective_budget_usd("research_analysis", settings) == 1.5


def test_validate_query_respects_pack_max_chars() -> None:
    policy = PackPolicy(
        pack_id="research_analysis",
        constraints=ExecutionConstraints(max_query_chars=5),
    )
    with patch.object(PolicyRegistry, "get", return_value=policy):
        with pytest.raises(ValueError, match="maximum length"):
            validate_query_for_pack(
                "1234567890",
                "research_analysis",
                InputValidator(max_length=2000),
            )
