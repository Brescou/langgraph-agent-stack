"""
core/observability.py — Structured logging and optional OpenTelemetry tracing.

This module provides two capabilities that are essential for running the
agent pipeline in production:

1. **Structured JSON logging** via ``python-json-logger``.  When enabled,
   every log record is emitted as a single JSON line containing timestamp,
   level, logger name, message, and all ``extra`` fields — ready for
   ingestion by ELK, Datadog, or any structured-log sink.

2. **OpenTelemetry tracing** (optional).  When the ``opentelemetry-sdk``
   package is installed and ``OTEL_ENABLED=true``, a tracer provider is
   configured with an OTLP exporter.  The module exposes a thin ``Tracer``
   wrapper so callers can create spans without importing OTel directly.

Both features degrade gracefully: if the optional packages are not
installed, standard ``logging`` and no-op tracing are used instead.
"""

from __future__ import annotations

import contextvars
import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

# ---------------------------------------------------------------------------
# Request ID context variable — propagates request_id through async call chains
# ---------------------------------------------------------------------------

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def set_request_id(request_id: str) -> None:
    """Set the current request ID in the context variable."""
    _request_id_var.set(request_id)


def get_request_id() -> str:
    """Return the current request ID, or empty string if not set."""
    return _request_id_var.get()


class RequestIdFilter(logging.Filter):
    """Inject request_id into every log record from the context variable."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()  # type: ignore[attr-defined]
        return True


class SanitizingFilter(logging.Filter):
    """Redact sensitive keys in log record extras before serialisation.

    Delegates to ``core.security.sanitize_log_data`` which recursively
    strips passwords, tokens, API keys, and URLs with embedded credentials.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        from core.security import sanitize_log_data

        if hasattr(record, "__dict__"):
            for key in list(record.__dict__):
                if key.startswith("_") or key in {
                    "name",
                    "msg",
                    "args",
                    "created",
                    "relativeCreated",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "lineno",
                    "funcName",
                    "pathname",
                    "filename",
                    "module",
                    "levelno",
                    "levelname",
                    "message",
                    "msecs",
                    "process",
                    "processName",
                    "thread",
                    "threadName",
                    "request_id",
                    "taskName",
                }:
                    continue
                value = getattr(record, key, None)
                if isinstance(value, dict):
                    setattr(record, key, sanitize_log_data(value))
        return True


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger with structured JSON output if available.

    Falls back to the standard ``%(asctime)s`` text format when
    ``python-json-logger`` is not installed.

    Args:
        level: Python log level name (``DEBUG``, ``INFO``, …).
    """
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(level)

    try:
        from pythonjsonlogger.json import JsonFormatter

        formatter = JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    except ImportError:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())
    handler.addFilter(SanitizingFilter())
    root.addHandler(handler)


# ---------------------------------------------------------------------------
# OpenTelemetry tracing (optional)
# ---------------------------------------------------------------------------

_tracer: Any | None = None
_OTEL_AVAILABLE = False

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _OTEL_AVAILABLE = True
except ImportError:
    pass


def init_tracing(service_name: str = "langgraph-agent-stack") -> None:
    """Initialise the OpenTelemetry tracer provider.

    This is a no-op when ``opentelemetry-sdk`` is not installed or
    ``OTEL_ENABLED`` is not set to a truthy value.

    Args:
        service_name: The ``service.name`` resource attribute.
    """
    global _tracer

    if not _OTEL_AVAILABLE:
        logging.getLogger(__name__).debug(
            "OpenTelemetry SDK not installed — tracing disabled"
        )
        return

    otel_enabled = os.getenv("OTEL_ENABLED", "false").lower() in ("1", "true", "yes")
    if not otel_enabled:
        logging.getLogger(__name__).debug("OTEL_ENABLED is not set — tracing disabled")
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    otel_insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "").lower() in (
        "1",
        "true",
        "yes",
    )
    is_local = "localhost" in otlp_endpoint or "127.0.0.1" in otlp_endpoint
    if otel_insecure or is_local:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    else:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)

    logging.getLogger(__name__).info(
        "OpenTelemetry tracing enabled",
        extra={"endpoint": otlp_endpoint, "service": service_name},
    )


def get_tracer() -> Any:
    """Return the active OTel tracer, or a no-op stub."""
    if _tracer is not None:
        return _tracer
    if _OTEL_AVAILABLE:
        return trace.get_tracer("langgraph-agent-stack")

    class _NoOpTracer:
        """Minimal stub so callers can use ``tracer.start_as_current_span``."""

        @contextmanager
        def start_as_current_span(
            self, name: str, **kwargs: Any
        ) -> Generator[None, None, None]:
            yield

    return _NoOpTracer()


@contextmanager
def trace_span(
    name: str, attributes: dict[str, Any] | None = None
) -> Generator[Any, None, None]:
    """Context manager that wraps a block in an OTel span.

    Attributes are attached if the span is real.  When OTel is disabled the
    block executes with zero overhead.

    Args:
        name: Span name (e.g. ``"research_node"``).
        attributes: Optional key-value pairs attached to the span.

    Yields:
        The active span, or ``None`` when tracing is disabled.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes and hasattr(span, "set_attribute"):
            for k, v in attributes.items():
                span.set_attribute(k, v)
        yield span


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        make_asgi_app,
    )

    http_requests_total = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "path", "status_code"],
    )
    http_request_duration_seconds = Histogram(
        "http_request_duration_seconds",
        "HTTP request duration in seconds",
        ["path"],
        buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    )
    llm_requests_total = Counter(
        "llm_requests_total",
        "Total LLM API calls",
        ["provider", "status"],
    )
    active_pipelines = Gauge(
        "active_pipelines",
        "Currently running agent pipelines",
    )

    def create_metrics_app() -> Any:
        """Return ASGI app for /metrics endpoint."""
        return make_asgi_app()

    _PROMETHEUS_AVAILABLE = True

except ImportError:
    http_requests_total = None  # type: ignore[assignment]
    http_request_duration_seconds = None  # type: ignore[assignment]
    llm_requests_total = None  # type: ignore[assignment]
    active_pipelines = None  # type: ignore[assignment]
    _PROMETHEUS_AVAILABLE = False

    def create_metrics_app() -> Any:
        """No-op when prometheus-client is not installed."""
        return None
