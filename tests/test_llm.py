"""tests/test_llm.py — Unit tests for the LLM factory (core/llm.py)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from core.llm import LLMConfig, get_llm


class TestGetLlmAnthropic:
    def test_returns_chat_anthropic(self):
        mock_model = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "langchain_anthropic": MagicMock(
                    ChatAnthropic=MagicMock(return_value=mock_model)
                )
            },
        ):
            config = LLMConfig(provider="anthropic", anthropic_api_key="sk-ant-test123")
            result = get_llm(config)
        assert result is mock_model

    def test_raises_without_api_key(self):
        with patch.dict(
            "sys.modules",
            {"langchain_anthropic": MagicMock(ChatAnthropic=MagicMock())},
        ):
            config = LLMConfig(provider="anthropic", anthropic_api_key=None)
            with pytest.raises(ValueError, match="anthropic_api_key"):
                get_llm(config)

    def test_raises_on_import_error(self):
        original = sys.modules.pop("langchain_anthropic", None)
        try:
            config = LLMConfig(provider="anthropic", anthropic_api_key="sk-ant-test123")
            with pytest.raises(ImportError):
                get_llm(config)
        finally:
            if original is not None:
                sys.modules["langchain_anthropic"] = original


class TestGetLlmOpenAI:
    def test_returns_chat_openai(self):
        mock_model = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "langchain_openai": MagicMock(
                    ChatOpenAI=MagicMock(return_value=mock_model)
                )
            },
        ):
            config = LLMConfig(provider="openai", openai_api_key="sk-openai-test")
            result = get_llm(config)
        assert result is mock_model

    def test_raises_without_api_key(self):
        with patch.dict(
            "sys.modules",
            {"langchain_openai": MagicMock(ChatOpenAI=MagicMock())},
        ):
            config = LLMConfig(provider="openai", openai_api_key=None)
            with pytest.raises(ValueError, match="openai_api_key"):
                get_llm(config)


class TestGetLlmOllama:
    def test_returns_chat_ollama(self):
        mock_model = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "langchain_ollama": MagicMock(
                    ChatOllama=MagicMock(return_value=mock_model)
                )
            },
        ):
            config = LLMConfig(provider="ollama")
            result = get_llm(config)
        assert result is mock_model


class TestGetLlmUnknown:
    def test_raises_on_unknown_provider(self):
        # Force an unknown provider via direct config construction
        config = LLMConfig.__new__(LLMConfig)
        object.__setattr__(config, "provider", "unknown_provider")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm(config)


class TestGetLlmGoogle:
    def test_returns_chat_google(self) -> None:
        mock_model = MagicMock()
        mock_module = MagicMock()
        mock_module.ChatGoogleGenerativeAI.return_value = mock_model
        with patch.dict("sys.modules", {"langchain_google_genai": mock_module}):
            config = LLMConfig(provider="google", google_api_key="gai-test-key")
            result = get_llm(config)
        assert result is mock_model

    def test_raises_without_api_key(self) -> None:
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"langchain_google_genai": mock_module}):
            config = LLMConfig(provider="google", google_api_key=None)
            with pytest.raises(ValueError, match="google_api_key"):
                get_llm(config)


class TestGetLlmBedrock:
    def test_returns_chat_bedrock(self) -> None:
        mock_model = MagicMock()
        mock_module = MagicMock()
        mock_module.ChatBedrock.return_value = mock_model
        with patch.dict("sys.modules", {"langchain_aws": mock_module}):
            config = LLMConfig(
                provider="bedrock",
                aws_access_key_id="AKIATEST",
                aws_secret_access_key="secret",
            )
            result = get_llm(config)
        assert result is mock_model

    def test_raises_without_access_key(self) -> None:
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"langchain_aws": mock_module}):
            config = LLMConfig(provider="bedrock", aws_access_key_id=None)
            with pytest.raises(ValueError, match="aws_access_key_id"):
                get_llm(config)

    def test_raises_without_secret_key(self) -> None:
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"langchain_aws": mock_module}):
            config = LLMConfig(
                provider="bedrock",
                aws_access_key_id="AKIATEST",
                aws_secret_access_key=None,
            )
            with pytest.raises(ValueError, match="aws_secret_access_key"):
                get_llm(config)


class TestGetLlmAzure:
    def test_returns_azure_chat_openai(self) -> None:
        mock_model = MagicMock()
        mock_module = MagicMock()
        mock_module.AzureChatOpenAI.return_value = mock_model
        with patch.dict("sys.modules", {"langchain_openai": mock_module}):
            config = LLMConfig(
                provider="azure",
                azure_openai_api_key="azure-key",
                azure_openai_endpoint="https://my-resource.openai.azure.com/",
            )
            result = get_llm(config)
        assert result is mock_model

    def test_raises_without_api_key(self) -> None:
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"langchain_openai": mock_module}):
            config = LLMConfig(provider="azure", azure_openai_api_key=None)
            with pytest.raises(ValueError, match="azure_openai_api_key"):
                get_llm(config)

    def test_raises_without_endpoint(self) -> None:
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"langchain_openai": mock_module}):
            config = LLMConfig(
                provider="azure",
                azure_openai_api_key="azure-key",
                azure_openai_endpoint=None,
            )
            with pytest.raises(ValueError, match="azure_openai_endpoint"):
                get_llm(config)
