"""Unit tests for the audit trail store and record builder (Phase 4)."""

import json

import pytest

from graphrag.path_extractor import build_cypher
from graphrag.traversal_logger import AuditStore, build_audit_record


@pytest.fixture
def store(tmp_path):
    return AuditStore(tmp_path / "audit_trail.jsonl")


def _record(audit_id="abc123"):
    return {
        "audit_id": audit_id,
        "timestamp": "2026-08-12T12:00:00",
        "query": "Does claim CLM-0003 have a fraud flag?",
        "answer": "Yes",
        "answer_mode": "extractive",
        "reranker": "lexical",
        "max_hops": 2,
        "retrieval": {"seeds": ["CLM-0003"], "seed_kinds": ["id"],
                      "node_count": 3, "edge_count": 2},
        "traversal": {"nodes_visited": ["CLM-0003", "FRD-CLM-0003"],
                      "edges_traversed": [["CLM-0003", "FRAUD_DETECTED", "FRD-CLM-0003"]],
                      "paths": [{"nodes": ["CLM-0003", "FRD-CLM-0003"],
                                 "edges": [["CLM-0003", "FRAUD_DETECTED", "FRD-CLM-0003"]]}]},
        "cypher": build_cypher({"query": "q", "seeds": [], "nodes": [], "edges": []}),
        "ranking": [{"id": "CLM-0003", "label": "Claim", "score": 5.0}],
        "pruned": {"kept": ["CLM-0003"], "dropped": [], "kept_count": 1,
                   "dropped_count": 0, "budget": 1280},
        "tokens": {"before": 100, "after": 40, "savings_percent": 60.0},
        "timings_ms": {"retrieval_ms": 1.0, "rerank_ms": 2.0,
                       "prune_ms": 3.0, "total_ms": 6.0},
    }


def test_append_and_recent(store):
    store.append(_record())
    recs = store.recent()
    assert len(recs) == 1
    assert recs[0]["audit_id"] == "abc123"


def test_recent_newest_first(store):
    store.append(_record("first"))
    store.append(_record("second"))
    assert [r["audit_id"] for r in store.recent()] == ["second", "first"]


def test_recent_limit(store):
    for i in range(10):
        store.append(_record(f"r{i:03d}"))
    recs = store.recent(3)
    assert len(recs) == 3
    assert recs[0]["audit_id"] == "r009"


def test_recent_survives_restart(store, tmp_path):
    store.append(_record("persisted"))
    # a brand-new store over the same file sees the record (cross-process)
    store2 = AuditStore(tmp_path / "audit_trail.jsonl")
    assert store2.recent()[0]["audit_id"] == "persisted"


def test_recent_tolerates_torn_line(store):
    store.append(_record("good"))
    with store.path.open("a", encoding="utf-8") as fh:
        fh.write('{"audit_id": "torn"')  # invalid JSON tail
    recs = store.recent()
    assert recs[0]["audit_id"] == "good"


def test_get_by_id(store):
    store.append(_record("target"))
    assert store.get("target") is not None
    assert store.get("missing") is None


def test_trim_keeps_newest_records(tmp_path):
    small = AuditStore(tmp_path / "trim.jsonl", max_records=5)
    for i in range(12):
        small.append(_record(f"r{i:03d}"))
    recs = small.recent()
    assert len(recs) == 5
    assert recs[0]["audit_id"] == "r011"  # newest retained
    assert "r000" not in {r["audit_id"] for r in recs}  # oldest trimmed
    # file itself stays bounded
    assert small._count <= 5


def test_clear(store):
    store.append(_record())
    assert store.clear() == 1
    assert store.recent() == []


def test_build_audit_record_shape():
    sub = {"query": "q", "seeds": [{"id": "S1", "label": "Claim", "kind": "id"}],
           "nodes": [{"id": "S1", "label": "Claim", "props": {}},
                     {"id": "T1", "label": "FraudFlag", "props": {}}],
           "edges": [{"source": "S1", "type": "FRAUD_DETECTED", "target": "T1"}]}
    rec = build_audit_record(
        query="q", subgraph=sub,
        ranked=[({"id": "T1", "label": "FraudFlag"}, 0.9),
                ({"id": "S1", "label": "Claim"}, 0.1)],
        pruned={"kept": ["T1"], "dropped": [], "node_count": 1, "dropped_count": 0,
                "budget": 1280, "tokens": 10},
        tokens={"before": 20, "after": 10, "savings_percent": 50.0},
        answer="a", answer_mode="llm", answer_model="llama3.1",
        reranker="lexical", max_hops=1,
        t0=0.0, t1=0.1, t2=0.2, t3=0.3, t4=0.4,
    )
    assert rec["audit_id"]
    assert rec["tokens"]["savings_percent"] == 50.0
    assert rec["traversal"]["nodes_visited"] == ["S1", "T1"]
    assert rec["ranking"][0]["id"] == "T1"
    assert rec["answer_mode"] == "llm"
    assert rec["answer_model"] == "llama3.1"
    assert rec["timings_ms"] == {"retrieval_ms": 100.0, "rerank_ms": 100.0,
                                 "prune_ms": 100.0, "answer_ms": 100.0,
                                 "total_ms": 400.0}
    # record must be JSON-serializable end to end
    json.dumps(rec)
