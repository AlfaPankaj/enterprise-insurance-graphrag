"""Tests for Phase 5 API security (src/graphrag/security.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from graphrag.config import settings
from graphrag.security import require_api_key, setup_security


def _tiny_app() -> FastAPI:
    app = FastAPI()

    @app.get("/protected")
    def protected(_auth=Depends(require_api_key)):
        return {"ok": True}

    return app


def test_auth_disabled_when_no_key(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")
    client = TestClient(_tiny_app())
    assert client.get("/protected").status_code == 200


def test_auth_rejects_missing_and_wrong_key(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "s3cret")
    client = TestClient(_tiny_app())
    assert client.get("/protected").status_code == 403
    assert client.get("/protected", headers={"X-API-Key": "wrong"}).status_code == 403


def test_auth_accepts_correct_key(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "s3cret")
    client = TestClient(_tiny_app())
    resp = client.get("/protected", headers={"X-API-Key": "s3cret"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_security_headers_present(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")
    monkeypatch.setattr(settings, "CORS_ORIGINS", "http://localhost:8501")
    app = _tiny_app()
    setup_security(app)
    client = TestClient(app)
    resp = client.get("/protected")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["X-XSS-Protection"] == "1; mode=block"
    assert resp.headers["Referrer-Policy"] == "no-referrer"
    # CORS preflight allowed from a configured origin
    pre = client.options("/protected", headers={
        "Origin": "http://localhost:8501",
        "Access-Control-Request-Method": "GET",
    })
    assert pre.headers.get("access-control-allow-origin") == "http://localhost:8501"
