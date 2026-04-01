"""Tests for core/observability.py — structured logging and OTel tracing."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest


class TestConfigureLogging:
    """Tests for ``configure_logging``."""

    def test_sets_root_logger_level(self) -> None:
        from core.observability import configure_logging

        configure_logging(level="WARNING")
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_adds_handler_to_root_logger(self) -> None:
        from core.observability import configure_logging

        configure_logging(level="INFO")
        root = logging.getLogger()
        assert len(root.handlers) >= 1
        assert isinstance(root.handlers[-1], logging.StreamHandler)

    def test_fallback_formatter_when_json_logger_missing(self) -> None:
        from core.observability import configure_logging

        with patch.dict(
            "sys.modules", {"pythonjsonlogger": None, "pythonjsonlogger.json": None}
        ):
            configure_logging(level="DEBUG")
        root = logging.getLogger()
        handler = root.handlers[-1]
        assert isinstance(handler.formatter, logging.Formatter)


class TestInitTracing:
    """Tests for ``init_tracing``."""

    def test_noop_when_otel_not_available(self) -> None:
        from core.observability import init_tracing

        with patch("core.observability._OTEL_AVAILABLE", False):
            init_tracing()

    def test_noop_when_otel_disabled(self) -> None:
        from core.observability import init_tracing

        with (
            patch("core.observability._OTEL_AVAILABLE", True),
            patch.dict("os.environ", {"OTEL_ENABLED": "false"}),
        ):
            init_tracing()


class TestGetTracer:
    """Tests for ``get_tracer``."""

    def test_returns_noop_tracer_when_otel_unavailable(self) -> None:
        from core.observability import get_tracer

        with (
            patch("core.observability._tracer", None),
            patch("core.observability._OTEL_AVAILABLE", False),
        ):
            tracer = get_tracer()
            assert tracer is not None
            assert hasattr(tracer, "start_as_current_span")

    def test_returns_module_tracer_when_set(self) -> None:
        from core.observability import get_tracer

        sentinel = MagicMock()
        with patch("core.observability._tracer", sentinel):
            assert get_tracer() is sentinel


class TestTraceSpan:
    """Tests for ``trace_span``."""

    def test_context_manager_works_without_otel(self) -> None:
        from core.observability import trace_span

        with (
            patch("core.observability._tracer", None),
            patch("core.observability._OTEL_AVAILABLE", False),
        ):
            with trace_span("test-span", attributes={"key": "value"}) as span:
                assert span is None

    def test_context_manager_propagates_exceptions(self) -> None:
        from core.observability import trace_span

        with (
            patch("core.observability._tracer", None),
            patch("core.observability._OTEL_AVAILABLE", False),
            pytest.raises(ValueError, match="boom"),
        ):
            with trace_span("failing-span"):
                raise ValueError("boom")
