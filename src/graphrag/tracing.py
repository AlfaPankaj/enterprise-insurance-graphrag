"""OpenTelemetry tracing (v2 — WS-D observability, optional dependency).

Zero cost when disabled: ``TRACING_ENABLED=false`` (default) or the
``opentelemetry`` packages absent → every entry point returns a no-op context
manager. When enabled, spans follow the pipeline stages so a distributed
trace shows retrieve → rerank → prune → answer per query, and every HTTP
request carries its own span (request id, caller, tenant, outcome).

Export: OTLP HTTP (``TRACING_OTLP_ENDPOINT``, or the standard
``OTEL_EXPORTER_OTLP_ENDPOINT`` env var) — Jaeger/Tempo/Datadog/New Relic all
speak it. ``configure()`` is called from the API lifespan; the response
carries ``X-Trace-ID`` so support can correlate a user report to a trace.

Optional deps (not in the base requirements — see ``requirements-otel.txt``):
opentelemetry-api / opentelemetry-sdk / opentelemetry-exporter-otlp-proto-http
"""

from __future__ import annotations

import importlib.util
import logging
from contextlib import contextmanager

from graphrag.config import settings

logger = logging.getLogger("graphrag.tracing")

_TRACER = None
_PROVIDER = None


def otel_installed() -> bool:
    """True when the opentelemetry packages are importable (no import here)."""
    return importlib.util.find_spec("opentelemetry") is not None


def tracing_enabled() -> bool:
    """Span recording is active: setting on AND packages installed."""
    return bool(settings.TRACING_ENABLED) and otel_installed()


def _clear_provider() -> None:
    """Reset OTel's set-once guard so the provider can be (re)configured."""
    try:
        from opentelemetry import trace as trace_api
        once = getattr(trace_api, "_TRACER_PROVIDER_SET_ONCE", None)
        if once is not None and hasattr(once, "_done"):
            once._done = False  # noqa: SLF001 - config/test only
        trace_api._TRACER_PROVIDER = None  # noqa: SLF001
    except Exception:  # noqa: BLE001 - best effort against API drift
        pass


def configure(exporter=None) -> bool:
    """Set up the global tracer provider. Returns True when tracing is live.

    ``exporter`` (tests) overrides the exporter; production builds an
    OTLP HTTP exporter from ``TRACING_OTLP_ENDPOINT`` /
    ``OTEL_EXPORTER_OTLP_ENDPOINT``. Called at API startup; safe to call
    again (existing provider is replaced).
    """
    global _TRACER, _PROVIDER
    if not tracing_enabled():
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (BatchSpanProcessor,
                                                    SimpleSpanProcessor)

        _clear_provider()
        endpoint = settings.TRACING_OTLP_ENDPOINT or ""
        if exporter is None and not endpoint:
            logger.info("tracing enabled but no OTLP endpoint configured — "
                        "spans are no-op exported")
            _PROVIDER = TracerProvider(resource=Resource.create({SERVICE_NAME: "graphrag-api"}))
        elif exporter is None:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import \
                OTLPSpanExporter
            _PROVIDER = TracerProvider(resource=Resource.create({SERVICE_NAME: "graphrag-api"}))
            _PROVIDER.add_span_processor(BatchSpanProcessor(
                OTLPSpanExporter(endpoint=endpoint)))
        else:
            _PROVIDER = TracerProvider(resource=Resource.create({SERVICE_NAME: "graphrag-api"}))
            _PROVIDER.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(_PROVIDER)
        _TRACER = trace.get_tracer("graphrag")
        return True
    except Exception as exc:  # noqa: BLE001 - observability must never break the app
        logger.warning("tracing setup failed: %s", exc)
        return False


def reset() -> None:
    """Unset the global provider (tests)."""
    global _TRACER, _PROVIDER
    _TRACER = None
    _PROVIDER = None
    _clear_provider()


def get_tracer():
    """The module tracer (None when tracing is not configured)."""
    global _TRACER
    if _TRACER is not None:
        return _TRACER
    if tracing_enabled():
        configure()
    return _TRACER


@contextmanager
def start_span(name: str, attributes: dict | None = None):
    """Record a span (no-op context manager when tracing is off)."""
    tracer = get_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)
        yield span


def current_trace_id() -> str | None:
    """Hex trace id of the active span (for X-Trace-ID correlation)."""
    tracer = get_tracer()
    if tracer is None:
        return None
    try:
        from opentelemetry import trace as trace_api
        span = trace_api.get_current_span()
        ctx = span.get_span_context()
        return ctx.trace_id.to_bytes(16, "big").hex() if ctx.is_valid else None
    except Exception:  # noqa: BLE001
        return None
