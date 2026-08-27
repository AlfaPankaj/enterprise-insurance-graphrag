"""v2 OpenTelemetry tracing tests (optional-dependency paths)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest
from fastapi.testclient import TestClient

import graphrag.tracing as tracing
from graphrag.config import settings


@pytest.fixture(autouse=True)
def _reset_tracing():
    yield
    settings.TRACING_ENABLED = False
    tracing.reset()


@pytest.fixture()
def in_memory_exporter():
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import \
        InMemorySpanExporter
    return InMemorySpanExporter()


def test_disabled_is_noop():
    settings.TRACING_ENABLED = False
    assert tracing.tracing_enabled() is False
    assert tracing.get_tracer() is None
    with tracing.start_span("nope", {"k": "v"}) as span:
        assert span is None
    assert tracing.current_trace_id() is None


def test_configure_with_in_memory_exporter(in_memory_exporter):
    settings.TRACING_ENABLED = True
    assert tracing.configure(exporter=in_memory_exporter) is True
    with tracing.start_span("unit.span", {"answer": 42}):
        pass
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "unit.span"
    assert spans[0].attributes["answer"] == 42


def test_nested_spans_have_parent(in_memory_exporter):
    settings.TRACING_ENABLED = True
    tracing.configure(exporter=in_memory_exporter)
    with tracing.start_span("outer"):
        with tracing.start_span("inner"):
            pass
    spans = in_memory_exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}
    assert set(by_name) == {"outer", "inner"}
    assert by_name["inner"].parent is not None
    assert by_name["inner"].parent.span_id == by_name["outer"].context.span_id


def test_configure_without_endpoint_still_records(in_memory_exporter):
    settings.TRACING_ENABLED = True
    settings.TRACING_OTLP_ENDPOINT = ""
    assert tracing.configure(exporter=in_memory_exporter) is True


# ---------------------------------------------------------------------------
# pipeline spans
# ---------------------------------------------------------------------------

NODES = {"CLM-0003": ("Claim", {"id": "CLM-0003", "status": "IN_REVIEW",
                                "amount": 5000.0})}
EDGES = []


class _Node:
    def __init__(self, props):
        self._props = props

    def __getitem__(self, k):
        return self._props[k]

    def items(self):
        return self._props.items()

    def keys(self):
        return self._props.keys()


class _Cursor:
    def __init__(self, query, kwargs):
        self.query, self.kwargs = query, kwargs

    def data(self):
        return []

    def single(self):
        q = self.query
        if "MATCH (d:Dataset)" in q:
            return None
        if "labels(n) AS labels, n.id AS id" in q:
            eid = self.kwargs.get("id")
            return {"labels": [NODES[eid][0]], "id": eid} if eid in NODES else None
        return None

    def __iter__(self):
        q = self.query
        if "labels(n) AS labels, n" in q:
            ids = self.kwargs.get("ids") or []
            for i in ids:
                if i in NODES:
                    yield {"labels": [NODES[i][0]], "n": _Node(NODES[i][1])}
        return
        yield  # pragma: no cover


class _Session:
    def run(self, query, **kwargs):
        return _Cursor(query, kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Driver:
    def session(self, **kw):
        return _Session()


def test_pipeline_emits_stage_spans(in_memory_exporter, monkeypatch):
    from graphrag.query_pipeline import run_query

    monkeypatch.setattr(settings, "AUDIT_ENABLED", False)  # don't touch the repo trail
    settings.TRACING_ENABLED = True
    tracing.configure(exporter=in_memory_exporter)
    result = run_query(_Driver(), "Does claim CLM-0003 have a fraud flag?",
                       answer_mode="extractive")
    assert "CLM-0003" in result["answer"]
    names = {s.name for s in in_memory_exporter.get_finished_spans()}
    assert {"graphrag.retrieve", "graphrag.rerank", "graphrag.prune",
            "graphrag.answer"} <= names


# ---------------------------------------------------------------------------
# HTTP middleware: X-Trace-ID correlation
# ---------------------------------------------------------------------------

def test_http_span_and_trace_header(in_memory_exporter, monkeypatch):
    import graphrag.api_server as api

    settings.TRACING_ENABLED = True
    tracing.configure(exporter=in_memory_exporter)
    # the lifespan reconfigures tracing from settings — pin the test provider
    monkeypatch.setattr(api, "configure_tracing", lambda: None)
    monkeypatch.setattr(settings, "AUTH_MODE", "none")
    monkeypatch.setattr(settings, "API_KEY", "")
    with TestClient(api.app) as client:
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.headers.get("X-Trace-ID")          # correlation id emitted
        assert len(resp.headers["X-Trace-ID"]) == 32   # 16-byte hex
    spans = in_memory_exporter.get_finished_spans()
    http_spans = [s for s in spans if s.name == "http.request"]
    assert http_spans
    assert http_spans[0].attributes["http.status_code"] == 200
    assert http_spans[0].attributes["http.url.path"] == "/health/live"
