"""Unit tests for the auto-pipeline (scripts/auto_pipeline.py)."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import httpx
import pytest

from scripts.auto_pipeline import (API_SESSION, _load_session,
                                   changed_datasets, fingerprint,
                                   stamp_report, switch_session_via_api)

import scripts.auto_pipeline as auto_pipeline


def _args(**over):
    base = dict(no_api=False, api_url="http://localhost:8000", api_key=None)
    base.update(over)
    return argparse.Namespace(**base)


def _dir_with(paths: dict[str, str]) -> Path:
    d = Path(tempfile.mkdtemp())
    for name, content in paths.items():
        (d / name).write_text(content, encoding="utf-8")
    return d


def test_fingerprint_detects_content_change():
    d = _dir_with({"a.csv": "x,1\ny,2\n"})
    fp1 = fingerprint(d / "a.csv")
    (d / "a.csv").write_text("x,1\ny,2\nz,3\n", encoding="utf-8")
    fp2 = fingerprint(d / "a.csv")
    assert fp2 != fp1
    assert fp2["size"] > fp1["size"]
    assert fp2["sha256"] != fp1["sha256"]


def test_changed_datasets_unchanged_returns_empty():
    d = _dir_with({"a.csv": "x,1\n", "b.csv": "y,2\n"})
    state = {"files": {name: fingerprint(d / f"{name}.csv") for name in ("a", "b")}}
    assert changed_datasets(d, state, force=False) == []


def test_changed_datasets_new_and_modified():
    d = _dir_with({"a.csv": "x,1\n"})
    state = {"files": {"a": fingerprint(d / "a.csv")}}
    (d / "a.csv").write_text("x,1\nchanged\n", encoding="utf-8")  # modified
    assert changed_datasets(d, state, force=False) == ["a"]
    state["files"]["a"] = fingerprint(d / "a.csv")
    (d / "b.csv").write_text("new\n", encoding="utf-8")  # new file
    assert changed_datasets(d, state, force=False) == ["b"]


def test_changed_datasets_force_returns_all():
    d = _dir_with({"a.csv": "x\n", "b.csv": "y\n"})
    assert changed_datasets(d, {"files": {}}, force=True) == ["a", "b"]


# ---------------------------------------------------------------------------
# session switching via the API
# ---------------------------------------------------------------------------

def test_api_session_map_excludes_data_synthetic():
    assert API_SESSION == {
        "fraud_oracle": "fraud_oracle",
        "insurance_claims": "insurance_claims",
        "insurance_dataset": "insurance_dataset",
    }


def test_switch_session_via_api_ok(monkeypatch):
    class FakeResp:
        status_code = 200
        text = ""

        def json(self):
            return {"seeded": True, "session": "fraud_oracle"}

    monkeypatch.setattr(auto_pipeline.httpx, "post", lambda *a, **k: FakeResp())
    assert switch_session_via_api("fraud_oracle", "http://localhost:8000", None) == "ok"


def test_switch_session_via_api_unreachable(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(auto_pipeline.httpx, "post", boom)
    out = switch_session_via_api("fraud_oracle", "http://localhost:8000", None)
    assert out == "unreachable"


def test_switch_session_via_api_error_raises(monkeypatch):
    class FakeResp:
        status_code = 400
        text = "unknown session 'nope'"

        def json(self):
            return {}

    monkeypatch.setattr(auto_pipeline.httpx, "post", lambda *a, **k: FakeResp())
    with pytest.raises(RuntimeError, match="400"):
        switch_session_via_api("nope", "http://localhost:8000", None)


def test_load_session_uses_api_when_available(monkeypatch):
    calls = []
    monkeypatch.setattr(auto_pipeline, "switch_session_via_api",
                        lambda sid, url, key, timeout=900: "ok")
    monkeypatch.setattr(auto_pipeline, "run",
                        lambda cmd: calls.append(cmd) or 0)
    assert _load_session("fraud_oracle", _args()) is True
    assert calls == []  # API handled it — no direct ingest ran


def test_load_session_falls_back_to_direct_when_api_down(monkeypatch):
    calls = []
    monkeypatch.setattr(auto_pipeline, "switch_session_via_api",
                        lambda sid, url, key, timeout=900: "unreachable")
    monkeypatch.setattr(auto_pipeline, "run",
                        lambda cmd: calls.append(cmd) or 0)
    assert _load_session("insurance_claims", _args()) is True
    assert len(calls) == 1 and "insurance_claims" in calls[0]


def test_load_session_direct_for_data_synthetic(monkeypatch):
    api_calls = []
    ingest_calls = []

    def fake_api(sid, url, key, timeout=900):
        api_calls.append(sid)
        return "ok"

    monkeypatch.setattr(auto_pipeline, "switch_session_via_api", fake_api)
    monkeypatch.setattr(auto_pipeline, "run",
                        lambda cmd: ingest_calls.append(cmd) or 0)
    assert _load_session("data_synthetic", _args()) is True
    assert api_calls == []  # no API session for the synthetic variant
    assert len(ingest_calls) == 1 and "data_synthetic" in ingest_calls[0]


def test_load_session_no_api_flag_forces_direct(monkeypatch):
    calls = []
    monkeypatch.setattr(auto_pipeline, "switch_session_via_api",
                        lambda sid, url, key, timeout=900: "ok")
    monkeypatch.setattr(auto_pipeline, "run",
                        lambda cmd: calls.append(cmd) or 0)
    assert _load_session("fraud_oracle", _args(no_api=True)) is True
    assert len(calls) == 1


def test_load_session_api_error_returns_false(monkeypatch):
    def boom(sid, url, key, timeout=900):
        raise RuntimeError("API session switch failed (500): boom")

    monkeypatch.setattr(auto_pipeline, "switch_session_via_api", boom)
    assert _load_session("fraud_oracle", _args()) is False


def test_stamp_report_appends_and_replaces_stamp(tmp_path):
    report = tmp_path / "real_dataset_results.md"
    report.write_text("# Real-Dataset Validation\n\nBody text.\n", encoding="utf-8")
    orig = auto_pipeline.REPORT
    auto_pipeline.REPORT = report
    try:
        stamp_report(["fraud_oracle"], {"timestamp": "2026-08-13 12:00:00",
                                        "elapsed_s": 10.0})
        text = report.read_text(encoding="utf-8")
        assert "Body text." in text
        # assert on stable substrings (the em-dash renders as '?' in the
        # cp1252 console, so never match the full line)
        assert "Auto-pipeline run 2026-08-13 12:00:00 (10.0s): fraud_oracle" in text
        assert "results refreshed in `data/benchmarks/`" in text

        stamp_report(["insurance_claims"], {"timestamp": "2026-08-13 12:05:00",
                                            "elapsed_s": 5.0})
        text = report.read_text(encoding="utf-8")
        # only the newest stamp survives; body is untouched
        assert "Auto-pipeline run 2026-08-13 12:00:00" not in text
        assert "Auto-pipeline run 2026-08-13 12:05:00 (5.0s): insurance_claims" in text
        assert "Body text." in text
    finally:
        auto_pipeline.REPORT = orig
