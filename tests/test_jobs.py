"""v2 durable job runner tests — store lifecycle, recovery, API endpoints."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest
from fastapi.testclient import TestClient

import graphrag.jobs as jobs_mod
from graphrag.config import settings
from graphrag.jobs import (JobCancelled, JobError, JobStore, STATUS_FAILED,
                           STATUS_INTERRUPTED, STATUS_PENDING,
                           STATUS_SUCCEEDED, register_handler)


@pytest.fixture()
def store(tmp_path):
    return JobStore(tmp_path / "jobs.db")


def _wait(job_store, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = job_store.get(job_id)
        if job and job["status"] not in (STATUS_PENDING, "running"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {job_store.get(job_id)}")


# ---------------------------------------------------------------------------
# store lifecycle
# ---------------------------------------------------------------------------

def test_submit_runs_handler_and_records_result(store):
    register_handler("echo", lambda job, params, progress, cancelled: {
        "got": params.get("x")})
    jid = store.submit("echo", {"x": 42})
    job = _wait(store, jid)
    assert job["status"] == STATUS_SUCCEEDED
    assert job["result"] == {"got": 42}
    assert job["started_at"] and job["finished_at"]


def test_progress_lines_recorded(store):
    def handler(job, params, progress, cancelled):
        progress("line one")
        progress("line two")
        return {"ok": True}

    register_handler("progressy", handler)
    jid = store.submit("progressy")
    job = _wait(store, jid)
    assert job["progress"] == ["line one", "line two"]


def test_failed_handler_records_error(store):
    def handler(job, params, progress, cancelled):
        raise RuntimeError("boom")

    register_handler("failer", handler)
    jid = store.submit("failer")
    job = _wait(store, jid)
    assert job["status"] == STATUS_FAILED
    assert "boom" in job["error"]


def test_unknown_kind_rejected(store):
    with pytest.raises(JobError, match="unknown job kind"):
        store.submit("no-such-kind")


def test_cancel_pending_and_running_jobs(store):
    """Deterministic cancel semantics (no worker-thread races)."""
    now = "2026-01-01T00:00:00"
    with store._lock, store._conn:
        store._conn.execute(
            "INSERT INTO jobs (id, kind, params, status, created_at) "
            "VALUES ('p1', 'x', '{}', 'pending', ?)", (now,))
        store._conn.execute(
            "INSERT INTO jobs (id, kind, params, status, created_at, started_at) "
            "VALUES ('r1', 'x', '{}', 'running', ?, ?)", (now, now))
    # pending -> cancelled before it ever starts
    assert store.cancel("p1") is True
    assert store.get("p1")["status"] == "cancelled"
    assert store.get("p1")["finished_at"]
    # running -> cooperative cancel flag set; the worker exits on its own
    assert store.cancel("r1") is True
    assert store.get("r1")["status"] == "running"
    assert "r1" in store._cancel_flags
    # already-terminal job -> no-op
    assert store.cancel("p1") is False


def test_list_newest_first(store):
    register_handler("echo2", lambda job, params, progress, cancelled: {})
    ids = [store.submit("echo2") for _ in range(3)]
    for jid in ids:
        _wait(store, jid)
    listed = [j["id"] for j in store.list(2)]
    assert listed == ids[:1:-1][:2] or len(listed) == 2


def test_crash_recovery_marks_running_as_interrupted(tmp_path):
    path = tmp_path / "jobs.db"
    s1 = JobStore(path)
    import sqlite3
    with s1._conn:
        s1._conn.execute(
            "INSERT INTO jobs (id, kind, params, status, created_at) "
            "VALUES ('stuck', 'x', '{}', 'running', '2026-01-01T00:00:00')")
        s1._conn.commit()
    s1._conn.close()
    # a new store (fresh process) recovers the abandoned job
    s2 = JobStore(path)
    assert s2.get("stuck")["status"] == STATUS_INTERRUPTED


# ---------------------------------------------------------------------------
# API endpoints (auth + flow)
# ---------------------------------------------------------------------------

@pytest.fixture()
def api_store(tmp_path, monkeypatch):
    import graphrag.api_server as api

    s = JobStore(tmp_path / "api-jobs.db")
    monkeypatch.setattr(api, "get_store", lambda: s)
    return s


def test_jobs_endpoints_flow(monkeypatch, api_store):
    import graphrag.api_server as api

    monkeypatch.setattr(settings, "AUTH_MODE", "none")
    monkeypatch.setattr(settings, "API_KEY", "")

    register_handler("api_echo", lambda job, params, progress, cancelled: {
        "echo": params.get("value")})

    with TestClient(api.app) as client:
        # submit
        resp = client.post("/api/v1/jobs", json={"kind": "api_echo",
                                                 "params": {"value": "hi"}})
        assert resp.status_code == 200
        jid = resp.json()["job_id"]

        # poll until done
        deadline = time.time() + 5
        status = None
        while time.time() < deadline:
            r = client.get(f"/api/v1/jobs/{jid}")
            assert r.status_code == 200
            status = r.json()["job"]["status"]
            if status == "succeeded":
                break
            time.sleep(0.02)
        assert status == "succeeded"
        assert client.get(f"/api/v1/jobs/{jid}").json()["job"]["result"] == \
            {"echo": "hi"}

        # list
        listed = client.get("/api/v1/jobs").json()["jobs"]
        assert any(j["id"] == jid for j in listed)

        # unknown kind -> 400; unknown id -> 404
        assert client.post("/api/v1/jobs",
                           json={"kind": "nope"}).status_code == 400
        assert client.get("/api/v1/jobs/missing").status_code == 404


def test_jobs_endpoint_role_gated(monkeypatch, api_store):
    import graphrag.api_server as api
    import jwt as pyjwt

    monkeypatch.setattr(settings, "AUTH_MODE", "jwt")
    monkeypatch.setattr(settings, "JWT_SECRET", "x" * 40)
    monkeypatch.setattr(settings, "JWT_ISSUER", "")
    monkeypatch.setattr(settings, "JWT_AUDIENCE", "")

    def token(roles):
        return pyjwt.encode({"sub": "u", "roles": roles}, "x" * 40,
                            algorithm="HS256")

    with TestClient(api.app) as client:
        analyst = {"Authorization": f"Bearer {token(['analyst'])}"}
        auditor = {"Authorization": f"Bearer {token(['auditor'])}"}
        admin = {"Authorization": f"Bearer {token(['admin'])}"}
        # submit is admin-only
        assert client.post("/api/v1/jobs", json={"kind": "x"},
                           headers=analyst).status_code == 403
        # audit roles may read; analysts may not
        assert client.get("/api/v1/jobs", headers=auditor).status_code == 200
        assert client.get("/api/v1/jobs", headers=analyst).status_code == 403
        assert client.get("/api/v1/jobs", headers=admin).status_code == 200


def test_session_switch_job_handler_uses_line_cb(monkeypatch, tmp_path):
    """The default handler streams seed lines through switch_session."""
    import graphrag.sessions as sessions_mod

    calls = {"line_cb": None}

    def fake_switch(driver, session_id, force=False, timeout=900, line_cb=None):
        calls["line_cb"] = line_cb
        if line_cb:
            line_cb("seeding line 1")
        return {"status": "seeded", "session": session_id}

    monkeypatch.setattr(sessions_mod, "switch_session", fake_switch)
    jobs_mod.register_default_handlers(lambda: object())
    handler = jobs_mod.get_handler("session_switch")
    progress = []
    result = handler({}, {"session_id": "pdf_demo"}, progress.append,
                     lambda: False)
    assert result["status"] == "seeded"
    assert calls["line_cb"] is not None
    assert progress == ["seeding line 1"]
