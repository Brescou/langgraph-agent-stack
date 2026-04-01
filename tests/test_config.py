"""
tests/test_config.py — Unit tests for core/config.py Settings and get_settings.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError


class TestGetSettings:
    def test_get_settings_returns_settings_instance(self) -> None:
        from core.config import Settings, get_settings

        get_settings.cache_clear()
        try:
            s = get_settings()
            assert isinstance(s, Settings)
        finally:
            get_settings.cache_clear()

    def test_get_settings_is_cached(self) -> None:
        from core.config import get_settings

        get_settings.cache_clear()
        try:
            s1 = get_settings()
            s2 = get_settings()
            assert s1 is s2
        finally:
            get_settings.cache_clear()

    def test_cache_clear_returns_fresh_instance(self) -> None:
        from core.config import get_settings

        get_settings.cache_clear()
        try:
            s1 = get_settings()
            get_settings.cache_clear()
            s2 = get_settings()
            assert s1 is not s2
        finally:
            get_settings.cache_clear()


class TestSettingsLlmConfig:
    def test_llm_config_property_returns_llm_config(self) -> None:
        from core.config import Settings
        from core.llm import LLMConfig

        settings = Settings(
            llm_provider="anthropic",
            anthropic_api_key="sk-ant-test123456789012345",
        )
        config = settings.llm_config
        assert isinstance(config, LLMConfig)
        assert config.provider == "anthropic"
        assert config.anthropic_api_key == "sk-ant-test123456789012345"


class TestSettingsValidators:
    def test_postgres_backend_requires_postgres_url(self) -> None:
        from core.config import get_settings

        env = {
            "MEMORY_BACKEND": "postgres",
            "LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "sk-ant-test123456789012345",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("POSTGRES_URL", None)
            get_settings.cache_clear()
            try:
                from core.config import Settings

                with pytest.raises(ValidationError, match="POSTGRES_URL"):
                    Settings()
            finally:
                get_settings.cache_clear()

    def test_postgres_backend_with_url_succeeds(self) -> None:
        from core.config import Settings

        env = {"POSTGRES_URL": "postgresql://user:pass@localhost/db"}
        with patch.dict(os.environ, env, clear=False):
            s = Settings(
                llm_provider="anthropic",
                anthropic_api_key="sk-ant-test123456789012345",
                memory_backend="postgres",
            )
        assert s.memory_backend.value == "postgres"
