"""v2 tamper-evident audit trail tests (hash chain, rotation, verification)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag.traversal_logger import AuditStore


def _record(audit_id: str) -> dict:
    return {"audit_id": audit_id, "query": "q", "answer": "a"}


def _lines(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def test_append_chains_records(tmp_path):
    store = AuditStore(tmp_path / "a.jsonl")
    store.append(_record("r1"))
    store.append(_record("r2"))
    recs = _lines(store.path)
    assert recs[0]["record_hash"] and recs[1]["record_hash"]
    assert recs[1]["prev_hash"] == recs[0]["record_hash"]   # linked
    assert store.verify()["valid"]


def test_verify_detects_tampered_record(tmp_path):
    store = AuditStore(tmp_path / "a.jsonl")
    store.append(_record("r1"))
    store.append(_record("r2"))
    lines = store.path.read_text().splitlines()
    # tamper: change the answer inside record 2 without recomputing the hash
    tampered = json.loads(lines[1])
    tampered["answer"] = "SOMETHING ELSE"
    lines[1] = json.dumps(tampered)
    store.path.write_text("\n".join(lines) + "\n")
    result = store.verify()
    assert not result["valid"]
    assert result["broken_at"]


def test_verify_detects_deleted_record(tmp_path):
    store = AuditStore(tmp_path / "a.jsonl")
    store.append(_record("r1"))
    store.append(_record("r2"))
    store.append(_record("r3"))
    lines = store.path.read_text().splitlines()
    store.path.write_text("\n".join([lines[0], lines[2]]) + "\n")  # r2 removed
    assert not store.verify()["valid"]


def test_chain_survives_rotation_across_segments(tmp_path):
    store = AuditStore(tmp_path / "a.jsonl", max_records=3)
    for i in range(7):
        store.append(_record(f"r{i}"))
    segs = store.segments()
    assert len(segs) == 2                       # archive segment + active file
    assert len(_lines(segs[-1])) == 3           # active holds the newest 3
    assert store.verify()["valid"]              # chain unbroken across segments


def test_verify_detects_tampering_in_archive_segment(tmp_path):
    store = AuditStore(tmp_path / "a.jsonl", max_records=3)
    for i in range(7):
        store.append(_record(f"r{i}"))
    archive = store.segments()[0]
    lines = archive.read_text().splitlines()
    tampered = json.loads(lines[2])
    tampered["answer"] = "HACKED"
    lines[2] = json.dumps(tampered)
    archive.write_text("\n".join(lines) + "\n")
    assert not store.verify()["valid"]


def test_legacy_records_anchor_the_chain(tmp_path):
    # a v1-style file (no hashes) then a new store appends a hashed record
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps(_record("legacy")) + "\n")
    store = AuditStore(path)
    store.append(_record("new"))
    result = store.verify()
    assert result["valid"]
    assert result["legacy_records"] == 1
    assert result["records_checked"] == 1
    # the new record's prev_hash binds to the legacy line's raw bytes
    new_rec = _lines(path)[1]
    assert new_rec["prev_hash"]


def test_get_searches_archive_segments(tmp_path):
    store = AuditStore(tmp_path / "a.jsonl", max_records=2)
    for i in range(5):
        store.append(_record(f"r{i}"))
    assert store.get("r0") is not None          # archived record still found
    assert store.get("r4") is not None
    assert store.get("missing") is None
