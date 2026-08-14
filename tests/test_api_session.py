"""API session endpoints (Phase 6) — GET/POST /api/v1/session.

These exercise the wiring (auth, validation, response shape, pass-through to
``switch_session``) without running a real ingest — ``switch_session`` is
monkeypatched, and ``current_session_id`` is stubbed so no test depends on the
state of Neo4j.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

import graphrag.api_server as api
from graphrag.api_server import app
from graphrag.config import settings


def _client():
    return TestClient(app)


def test_get_session_lists_four_sessions(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")
    monkeypatch.setattr(api, "current_session_id", lambda d: "pdf_demo")
    with _client() as client:
        resp = client.get("/api/v1/session")
        assert resp.status_code == 200
        body = resp.json()
        assert body["current_session"] == "pdf_demo"
        ids = [s["id"] for s in body["sessions"]]
        assert ids == ["fraud_oracle", "insurance_claims",
                       "insurance_dataset", "pdf_demo"]


def test_post_session_auth_enforced(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "prod-key")
    monkeypatch.setattr(api, "switch_session",
                        lambda d, s, force=False, timeout=900:
                        {"status": "already_loaded", "session": s, "output": ""})
    monkeypatch.setattr(api, "current_session_id", lambda d: "pdf_demo")
    with _client() as client:
        resp = client.post("/api/v1/session", json={"session_id": "pdf_demo"})
        assert resp.status_code == 403
        ok = client.post("/api/v1/session", json={"session_id": "pdf_demo"},
                         headers={"X-API-Key": "prod-key"})
        assert ok.status_code == 200
        assert ok.json()["seeded"] is False  # already loaded -> no-op


def test_post_session_unknown_id_400(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")
    with _client() as client:
        resp = client.post("/api/v1/session", json={"session_id": "nope"})
        assert resp.status_code == 400
        assert "unknown session" in resp.json()["detail"]


def test_post_session_missing_field_422(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")
    with _client() as client:
        assert client.post("/api/v1/session", json={}).status_code == 422


def test_post_session_switches_via_backend(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")
    calls = []

    def fake_switch(driver, session_id, force=False, timeout=900):
        calls.append((session_id, force))
        return {"status": "seeded", "session": session_id, "output": "ok"}

    monkeypatch.setattr(api, "switch_session", fake_switch)
    monkeypatch.setattr(api, "current_session_id", lambda d: "fraud_oracle")
    with _client() as client:
        resp = client.post("/api/v1/session",
                           json={"session_id": "fraud_oracle", "force": True})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["seeded"] is True
        assert body["session"] == "fraud_oracle"
        assert body["current_session"] == "fraud_oracle"
        assert "elapsed_ms" in body
    assert calls == [("fraud_oracle", True)]  # force passed through


def test_post_session_seeding_failure_500(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")

    def fake_switch(driver, session_id, force=False, timeout=900):
        raise RuntimeError("ingest exploded")

    monkeypatch.setattr(api, "switch_session", fake_switch)
    with _client() as client:
        resp = client.post("/api/v1/session", json={"session_id": "pdf_demo"})
        assert resp.status_code == 500
        assert "ingest exploded" in resp.json()["detail"]
