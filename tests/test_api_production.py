"""End-to-end API hardening tests (FastAPI TestClient against the real app).

These exercise the production wiring — liveness, auth enforcement, the
validation handler, security headers, and the metrics endpoint — without
touching Neo4j (the heavy query/upload paths are covered elsewhere).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from graphrag.api_server import app
from graphrag.config import settings


def _client():
    return TestClient(app)


def test_liveness_no_dependencies():
    with _client() as client:
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json() == {"status": "alive"}


def test_metrics_without_auth_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")
    with _client() as client:
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "summary" in body and "metrics" in body


def test_auth_enforced_when_key_set(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "prod-key-123")
    with _client() as client:
        assert client.get("/api/v1/metrics").status_code == 403
        ok = client.get("/api/v1/metrics", headers={"X-API-Key": "prod-key-123"})
        assert ok.status_code == 200


def test_security_headers_on_api_responses(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")
    with _client() as client:
        resp = client.get("/health/live")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"


def test_query_validation_422(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")
    with _client() as client:
        resp = client.post("/api/v1/query", json={"query": ""})
        assert resp.status_code == 422
        assert resp.json()["error"] == "Validation Error"


def test_upload_rejects_non_pdf(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")
    with _client() as client:
        resp = client.post(
            "/api/v1/upload",
            files={"file": ("claims.csv", b"a,b\n1,2\n", "text/csv")},
        )
        assert resp.status_code == 400
        assert "PDF" in resp.json()["detail"]


def test_upload_rejects_fake_pdf_magic(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")
    with _client() as client:
        resp = client.post(
            "/api/v1/upload",
            files={"file": ("fake.pdf", b"not really a pdf", "application/pdf")},
        )
        assert resp.status_code == 400
        assert "header" in resp.json()["detail"]
