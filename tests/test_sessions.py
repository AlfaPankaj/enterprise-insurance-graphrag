"""Tests for the session switcher (Phase 6)."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import graphrag.sessions as sessions_mod
from graphrag.sessions import (SESSIONS, SESSION_BY_ID, _run_blocking,
                               _seed_command, active_switches, clear_progress,
                               session_for_marker, start_switch,
                               switch_in_progress, switch_progress,
                               switch_session)


@pytest.fixture(autouse=True)
def _clear_progress_store():
    """Never leak background-switch handles between tests."""
    yield
    sessions_mod._PROGRESS.clear()


def _fake_proc(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return type("P", (), {"returncode": returncode, "stdout": stdout,
                          "stderr": stderr})()


def test_sessions_registry_has_all_sessions():
    assert [s["id"] for s in SESSIONS] == [
        "fraud_oracle", "insurance_claims", "insurance_dataset", "pdf_demo",
        "banking_demo",
    ]
    assert {s["kind"] for s in SESSIONS} == {"excel", "pdf", "banking"}
    assert all(s["id"] in SESSION_BY_ID for s in SESSIONS)
    # each excel session names a real CSV-backed dataset
    for s in SESSIONS:
        if s["kind"] == "excel":
            assert (Path(__file__).resolve().parents[1] / "data" / "Real_datasets"
                    / f"{s['dataset']}.csv").exists()


def test_session_for_marker_mapping():
    assert session_for_marker("fraud_oracle") == "fraud_oracle"
    assert session_for_marker("insurance_claims") == "insurance_claims"
    assert session_for_marker("insurance_dataset") == "insurance_dataset"
    # data_synthetic is the same Kaggle source -> insurance_dataset session
    assert session_for_marker("data_synthetic") == "insurance_dataset"
    assert session_for_marker("synthetic") == "pdf_demo"
    assert session_for_marker(None) == "pdf_demo"  # graph predates the marker
    assert session_for_marker("unknown_thing") == "pdf_demo"


def _has(cmd: list[str], fragment: str) -> bool:
    return any(fragment in c for c in cmd)


def test_seed_command_excel_vs_pdf():
    cmd = _seed_command(SESSION_BY_ID["fraud_oracle"])
    assert _has(cmd, "ingest_real_dataset.py")
    assert "fraud_oracle" in cmd and "--reset" in cmd
    cmd = _seed_command(SESSION_BY_ID["insurance_dataset"])
    assert "insurance_dataset" in cmd
    cmd = _seed_command(SESSION_BY_ID["pdf_demo"])
    assert _has(cmd, "seed_graph.py") and "--reset" in cmd and "--apply-schema" in cmd


def test_switch_noop_when_already_loaded(monkeypatch):
    """Selecting the currently-loaded session must NOT re-run the ingest."""
    monkeypatch.setattr("graphrag.sessions.current_session_id",
                        lambda d: "fraud_oracle")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return _fake_proc()

    monkeypatch.setattr("graphrag.sessions.subprocess.run", fake_run)
    out = switch_session(None, "fraud_oracle")
    assert out["status"] == "already_loaded"
    assert calls == []


def test_switch_runs_ingest_when_different(monkeypatch):
    monkeypatch.setattr("graphrag.sessions.current_session_id", lambda d: "pdf_demo")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return _fake_proc(stdout="seeded ok")

    monkeypatch.setattr("graphrag.sessions.subprocess.run", fake_run)
    out = switch_session(None, "insurance_claims")
    assert out["status"] == "seeded"
    assert calls and _has(calls[0], "ingest_real_dataset.py")
    assert "insurance_claims" in calls[0]


def test_switch_force_reruns_even_if_loaded(monkeypatch):
    monkeypatch.setattr("graphrag.sessions.current_session_id",
                        lambda d: "fraud_oracle")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return _fake_proc()

    monkeypatch.setattr("graphrag.sessions.subprocess.run", fake_run)
    out = switch_session(None, "fraud_oracle", force=True)
    assert out["status"] == "seeded"
    assert len(calls) == 1


def test_switch_failure_raises(monkeypatch):
    monkeypatch.setattr("graphrag.sessions.current_session_id", lambda d: "pdf_demo")

    def fake_run(cmd, **kw):
        return _fake_proc(returncode=3, stdout="boom", stderr="err")

    monkeypatch.setattr("graphrag.sessions.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="insurance_claims"):
        switch_session(None, "insurance_claims")


def test_unknown_session_raises():
    with pytest.raises(ValueError):
        switch_session(None, "nope")


def _wait_done(handle: dict, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while not handle["done"] and time.time() < deadline:
        time.sleep(0.01)
    return handle


# ---------------------------------------------------------------------------
# background streaming runner (web UI path)
# ---------------------------------------------------------------------------

def test_run_blocking_streams_lines(monkeypatch):
    monkeypatch.setattr("graphrag.sessions.current_session_id",
                        lambda d: "pdf_demo")

    class FakeProc:
        def __init__(self, lines, rc=0):
            self.stdout = iter(lines)
            self.returncode = rc

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr("graphrag.sessions.subprocess.Popen",
                        lambda *a, **k: FakeProc(["line one", "line two"]))
    got: list[str] = []
    out = _run_blocking(None, "insurance_claims", force=True, timeout=60,
                        line_cb=got.append)
    assert got == ["line one", "line two"]
    assert out["status"] == "seeded"
    assert "line one" in out["output"]


def test_start_switch_background_success(monkeypatch):
    def fake_blocking(driver, session_id, force, timeout, line_cb=None):
        line_cb("parsed 2,000 nodes")
        return {"status": "seeded", "session": session_id, "output": "ok"}

    monkeypatch.setattr("graphrag.sessions._run_blocking", fake_blocking)
    handle = start_switch(None, "insurance_claims")
    _wait_done(handle)
    assert handle["done"] and handle["ok"]
    assert handle["result"]["status"] == "seeded"
    assert handle["lines"] == ["parsed 2,000 nodes"]
    assert switch_progress("insurance_claims") is handle
    clear_progress("insurance_claims")
    assert switch_progress("insurance_claims") is None


def test_start_switch_background_failure(monkeypatch):
    def fake_blocking(driver, session_id, force, timeout, line_cb=None):
        raise RuntimeError("ingest exploded")

    monkeypatch.setattr("graphrag.sessions._run_blocking", fake_blocking)
    handle = start_switch(None, "fraud_oracle", force=True)
    _wait_done(handle)
    assert handle["done"] and not handle["ok"]
    assert "ingest exploded" in handle["error"]


def test_switch_in_progress_tracks_running(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def fake_blocking(driver, session_id, force, timeout, line_cb=None):
        started.set()
        release.wait(5)
        return {"status": "seeded", "session": session_id, "output": ""}

    monkeypatch.setattr("graphrag.sessions._run_blocking", fake_blocking)
    handle = start_switch(None, "insurance_claims", force=True)
    assert started.wait(5)
    assert switch_in_progress()
    assert [h["session"] for h in active_switches()] == ["insurance_claims"]
    release.set()
    _wait_done(handle)
    assert not switch_in_progress()  # finished handles are not "active"
