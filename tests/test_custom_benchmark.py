"""Tests for auto-benchmarking of custom (user-uploaded) sessions.

Covers the generic ground-truth query builder (scripts/benchmark_real_dataset.py
--custom-session) and the sessions.py trigger that runs it in the background
after a custom CSV session seeds, so the dashboard's Pipeline Validation row
fills in without any manual benchmark step.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import graphrag.sessions as sessions_mod
from graphrag.sessions import (_maybe_auto_benchmark, benchmark_running,
                               ensure_pdf_demo_fraud_benchmark)
from scripts.benchmark_fraud_detection import custom_ground_truth
from scripts.benchmark_real_dataset import (_expand_to_target,
                                           _queries_custom_session,
                                           _queries_synthetic)


# ---------------------------------------------------------------------------
# 100-query expansion (real datasets) + synthetic demo builder
# ---------------------------------------------------------------------------

def test_expand_to_target_reaches_100_unique():
    import scripts.benchmark_real_dataset as b

    for ds in ["fraud_oracle", "insurance_claims", "insurance_dataset",
               "data_synthetic"]:
        rows = b._rows(ds)
        qs = _expand_to_target(ds, rows, 100)
        assert len(qs) == 100, f"{ds}: {len(qs)} queries"
        assert len({q["query"] for q in qs}) == 100, f"{ds}: dupes"
        assert all(q["expected"] for q in qs), f"{ds}: empty expected"
        assert any(q["category"] == "id-lookup" for q in qs)
        # boundary coverage: first and last row ids must appear
        ids = {e for q in qs for e in q["expected"]}
        assert _id_for(ds, rows, 1) in ids and _id_for(ds, rows, len(rows)) in ids


def _id_for(ds: str, rows: list[dict], i: int) -> str:
    if ds == "data_synthetic":
        from scripts.ingest_real_dataset import policy_id

        return policy_id(rows[i - 1].get("Customer ID", ""), i, pad=5)
    return f"CLM-{i:05d}"


def test_queries_synthetic_builds_100():
    import json as _json

    raw = _json.loads(
        (PROJECT_ROOT / "data" / "samples" / "claims.json").read_text(
            encoding="utf-8"))
    items = raw if isinstance(raw, list) else raw.get("claims", [])
    qs = _queries_synthetic(items)
    assert len(qs) == 100
    cats = {q["category"] for q in qs}
    assert "id-lookup" in cats and "fraud-list" in cats
    # every expected id must reference a real claim in the demo data
    known = {(c.get("claim") or {}).get("id") for c in items}
    known |= {c.get("doc_id") for c in items}
    for q in qs:
        assert all(e in known for e in q["expected"]), q["query"]


# ---------------------------------------------------------------------------
# generic query builder (pure — no Neo4j)
# ---------------------------------------------------------------------------

def _rows(*rows: dict) -> list[dict]:
    return list(rows)


def test_custom_queries_claims_csv_with_fraud():
    rows = _rows(
        {"claim_id": "CLM-9001", "amount": "12000", "fraud": "1"},
        {"claim_id": "CLM-9002", "amount": "300", "fraud": "0"},
        {"claim_id": "CLM-9003", "amount": "8000", "fraud": "Yes"},
    )
    queries = _queries_custom_session(rows)
    cats = [q["category"] for q in queries]
    assert "id-lookup" in cats and "fraud-list" in cats and "amount-threshold" in cats
    idq = next(q for q in queries if q["category"] == "id-lookup")
    assert idq["expected"] == ["CLM-9001"]
    assert "CLM-9001" in idq["query"]
    fraudq = next(q for q in queries if q["category"] == "fraud-list")
    assert fraudq["expected"] == ["CLM-9001", "CLM-9003"]
    thrq = next(q for q in queries if q["category"] == "amount-threshold")
    assert thrq["mode"] == "top-k"
    # threshold = midpoint between 8000 and 12000 (10000) -> only CLM-9001
    # exceeds; the retriever treats "over" as >=, so the threshold must never
    # coincide with a data point
    assert thrq["expected"] == ["CLM-9001"]
    assert "amount" in thrq["query"]
    assert "10000" in thrq["query"]


def test_custom_queries_generic_records_csv():
    rows = _rows(
        {"name": "alice", "age": "30", "city": "NYC"},
        {"name": "bob", "age": "25", "city": "LA"},
        {"name": "carol", "age": "40", "city": "SF"},
        {"name": "dave", "age": "35", "city": "NYC"},
    )
    queries = _queries_custom_session(rows)
    cats = [q["category"] for q in queries]
    assert "id-lookup" in cats and "amount-threshold" in cats
    assert "fraud-list" not in cats
    idq = next(q for q in queries if q["category"] == "id-lookup")
    # no id column -> generated REC-00001 ids (same as the ingest adapter)
    assert idq["expected"] == ["REC-00001"]
    assert "record" in idq["query"].lower()


def test_custom_queries_no_variance_skips_threshold():
    rows = _rows(
        {"claim_id": "CLM-1", "status": "open"},
        {"claim_id": "CLM-2", "status": "closed"},
        {"claim_id": "CLM-3", "status": "open"},
    )
    queries = _queries_custom_session(rows)
    assert [q["category"] for q in queries] == ["id-lookup"]


def test_custom_queries_empty_rows():
    assert _queries_custom_session([]) == []


# ---------------------------------------------------------------------------
# auto-benchmark trigger (sessions.py) — no subprocess in tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_bench_state():
    yield
    sessions_mod._BENCH_RUNNING.clear()


def _custom_meta(session_id: str) -> dict:
    return {"id": session_id, "kind": "custom", "dataset": session_id,
            "label": f"{session_id} — custom (CSV)", "desc": "Custom CSV"}


def test_maybe_auto_benchmark_custom_csv(monkeypatch, tmp_path):
    monkeypatch.setattr("graphrag.sessions.get_session_meta",
                        lambda sid: _custom_meta(sid))
    monkeypatch.setattr("graphrag.sessions.ROOT", PROJECT_ROOT)
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return type("P", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("graphrag.sessions.subprocess.run", fake_run)
    _maybe_auto_benchmark("my_claims")
    # background thread — give it a moment to run
    deadline = time.time() + 5
    while len(calls) < 2 and time.time() < deadline:
        time.sleep(0.01)
    assert calls, "auto-benchmark command never ran"
    # retrieval benchmark first, then the fraud benchmark (custom CSV ground truth)
    assert any("benchmark_real_dataset.py" in c for c in calls[0])
    assert "--custom-session" in calls[0] and "my_claims" in calls[0]
    assert any("benchmark_fraud_detection.py" in c for c in calls[1])
    assert "--custom-session" in calls[1] and "my_claims" in calls[1]


def test_maybe_auto_benchmark_skips_builtin(monkeypatch):
    monkeypatch.setattr("graphrag.sessions.get_session_meta",
                        lambda sid: {"id": sid, "kind": "excel",
                                     "dataset": sid, "label": sid, "desc": ""})
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("graphrag.sessions.subprocess.run", fake_run)
    _maybe_auto_benchmark("fraud_oracle")
    time.sleep(0.2)
    assert calls == []


def test_maybe_auto_benchmark_skips_fraud_when_results_exist(monkeypatch, tmp_path):
    """Fraud benchmark should be skipped when fraud_detection_<name>.json exists
    (e.g. the CSV has no fraud column -> the fraud step failed last time)."""
    monkeypatch.setattr("graphrag.sessions.get_session_meta",
                        lambda sid: _custom_meta(sid))
    (tmp_path / "data" / "benchmarks").mkdir(parents=True)
    (tmp_path / "data" / "benchmarks" / "fraud_detection_my_claims.json").write_text("{}")
    monkeypatch.setattr("graphrag.sessions.ROOT", tmp_path)
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("graphrag.sessions.subprocess.run", fake_run)
    _maybe_auto_benchmark("my_claims")
    deadline = time.time() + 5
    while not calls and time.time() < deadline:
        time.sleep(0.01)
    assert calls, "retrieval benchmark should still run"
    assert not any("benchmark_fraud_detection.py" in c for c in calls)


def test_maybe_auto_benchmark_skips_when_results_exist(monkeypatch, tmp_path):
    monkeypatch.setattr("graphrag.sessions.get_session_meta",
                        lambda sid: _custom_meta(sid))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "benchmarks").mkdir()
    (tmp_path / "data" / "benchmarks" / "real_my_claims.json").write_text("{}")
    monkeypatch.setattr("graphrag.sessions.ROOT", tmp_path)
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("graphrag.sessions.subprocess.run", fake_run)
    _maybe_auto_benchmark("my_claims")
    time.sleep(0.2)
    assert calls == []


def test_benchmark_running_flag(monkeypatch):
    monkeypatch.setattr("graphrag.sessions.get_session_meta",
                        lambda sid: _custom_meta(sid))
    monkeypatch.setattr("graphrag.sessions.ROOT", PROJECT_ROOT)

    release = threading.Event()

    def fake_run(cmd, **kw):
        release.wait(5)
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("graphrag.sessions.subprocess.run", fake_run)
    _maybe_auto_benchmark("my_claims")
    deadline = time.time() + 5
    while not benchmark_running("my_claims") and time.time() < deadline:
        time.sleep(0.01)
    assert benchmark_running("my_claims")
    release.set()
    deadline = time.time() + 5
    while benchmark_running("my_claims") and time.time() < deadline:
        time.sleep(0.01)
    assert not benchmark_running("my_claims")


# ---------------------------------------------------------------------------
# custom fraud ground truth + pdf_demo fraud benchmark
# ---------------------------------------------------------------------------

def test_custom_ground_truth_claims_csv(tmp_path):
    p = tmp_path / "claims.csv"
    p.write_text("claim_id,amount,fraud\nCLM-1,1000,1\nCLM-2,500,0\nCLM-3,200,Yes\n",
                 encoding="utf-8")
    gt = custom_ground_truth(p)
    assert gt == {"CLM-1": True, "CLM-2": False, "CLM-3": True}


def test_custom_ground_truth_no_fraud_column(tmp_path):
    p = tmp_path / "people.csv"
    p.write_text("name,age\nalice,30\n", encoding="utf-8")
    assert custom_ground_truth(p) == {}


def test_ensure_pdf_demo_fraud_benchmark(monkeypatch, tmp_path):
    (tmp_path / "data" / "benchmarks").mkdir(parents=True)
    monkeypatch.setattr("graphrag.sessions.ROOT", tmp_path)
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("graphrag.sessions.subprocess.run", fake_run)
    ensure_pdf_demo_fraud_benchmark()
    deadline = time.time() + 5
    while not calls and time.time() < deadline:
        time.sleep(0.01)
    assert calls
    cmd = calls[0]
    assert any("benchmark_fraud_detection.py" in c for c in cmd)
    assert "--dataset" in cmd and "synthetic" in cmd


def test_ensure_pdf_demo_fraud_benchmark_skips_when_exists(monkeypatch, tmp_path):
    (tmp_path / "data" / "benchmarks").mkdir(parents=True)
    (tmp_path / "data" / "benchmarks" / "fraud_detection_synthetic.json").write_text("{}")
    monkeypatch.setattr("graphrag.sessions.ROOT", tmp_path)
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("graphrag.sessions.subprocess.run", fake_run)
    ensure_pdf_demo_fraud_benchmark()
    time.sleep(0.2)
    assert calls == []
