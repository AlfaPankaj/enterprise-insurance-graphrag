"""v2 zero-dependency Prometheus registry + /metrics endpoint tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import jwt as pyjwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import graphrag.prometheus as prom
from graphrag.config import settings


@pytest.fixture(autouse=True)
def _reset_metrics():
    prom.reset()
    yield
    prom.reset()


def test_counter_inc_and_render():
    prom.requests_total.inc(kind="query")
    prom.requests_total.inc(kind="query")
    prom.requests_total.inc(kind="upload")
    text = prom.render()
    assert '# HELP graphrag_requests_total' in text
    assert '# TYPE graphrag_requests_total counter' in text
    assert 'graphrag_requests_total{kind="query"} 2' in text
    assert 'graphrag_requests_total{kind="upload"} 1' in text


def test_histogram_buckets_and_sum():
    prom.query_latency.observe(0.3)
    prom.query_latency.observe(1.2)
    text = prom.render()
    assert 'graphrag_query_latency_seconds_bucket{le="0.5"} 1' in text
    assert 'graphrag_query_latency_seconds_bucket{le="2.5"} 2' in text
    assert 'graphrag_query_latency_seconds_bucket{le="+Inf"} 2' in text
    assert "graphrag_query_latency_seconds_sum 1.5" in text
    assert "graphrag_query_latency_seconds_count 2" in text


def test_unobserved_metrics_not_rendered():
    text = prom.render()
    assert "graphrag_requests_total" not in text  # nothing observed yet


def test_float_counter_precision():
    prom.llm_cost_total.inc(0.000027)
    text = prom.render()
    assert "2.7e-05" in text or "0.000027" in text


# ---------------------------------------------------------------------------
# /metrics endpoint (on the real app; auth gating)
# ---------------------------------------------------------------------------

def test_metrics_endpoint_dev_mode(monkeypatch):
    import graphrag.api_server as api

    monkeypatch.setattr(settings, "AUTH_MODE", "none")
    monkeypatch.setattr(settings, "API_KEY", "")
    prom.requests_total.inc(kind="query")
    with TestClient(api.app) as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        assert 'graphrag_requests_total{kind="query"} 1' in resp.text


def test_metrics_endpoint_role_gated(monkeypatch):
    import graphrag.api_server as api

    monkeypatch.setattr(settings, "AUTH_MODE", "jwt")
    monkeypatch.setattr(settings, "JWT_SECRET", "x" * 40)
    monkeypatch.setattr(settings, "JWT_ISSUER", "")
    monkeypatch.setattr(settings, "JWT_AUDIENCE", "")

    def token(roles):
        return pyjwt.encode({"sub": "u", "roles": roles}, "x" * 40,
                            algorithm="HS256")

    with TestClient(api.app) as client:
        # analyst may NOT scrape metrics
        r = client.get("/metrics",
                       headers={"Authorization": f"Bearer {token(['analyst'])}"})
        assert r.status_code == 403
        # auditor/admin may
        r = client.get("/metrics",
                       headers={"Authorization": f"Bearer {token(['auditor'])}"})
        assert r.status_code == 200
        r = client.get("/metrics",
                       headers={"Authorization": f"Bearer {token(['admin'])}"})
        assert r.status_code == 200


def test_rate_limited_counter_increments_on_429():
    from graphrag.rate_limiter import _WINDOWS, rate_limit

    _WINDOWS.clear()
    app = FastAPI()

    @app.get("/limited")
    def limited(_rl=Depends(rate_limit(limit=2, window_s=60))):
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/limited").status_code == 200
    assert client.get("/limited").status_code == 200
    assert client.get("/limited").status_code == 429
    assert "graphrag_rate_limited_total 1" in prom.render()
    _WINDOWS.clear()


def test_audit_records_counter_increments(tmp_path):
    from graphrag.traversal_logger import AuditStore

    store = AuditStore(tmp_path / "audit.jsonl")
    store.append({"audit_id": "a1", "query": "q", "answer": "a"})
    store.append({"audit_id": "a2", "query": "q", "answer": "a"})
    assert "graphrag_audit_records_total 2" in prom.render()
