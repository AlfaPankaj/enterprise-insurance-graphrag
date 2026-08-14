"""Unit tests for audit report exports — JSON / HTML / PDF (Phase 4)."""

import json

from graphrag.audit_reporter import (render_html, render_json, render_pdf,
                                     save_exports)


def _record(audit_id="abc123"):
    """Minimal but complete audit record (mirrors the real store shape)."""
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
        "cypher": "-- seed lookup --\nMATCH (s0:Claim {id: \"CLM-0003\"})",
        "ranking": [{"id": "CLM-0003", "label": "Claim", "score": 5.0},
                     {"id": "FRD-CLM-0003", "label": "FraudFlag", "score": 1.0}],
        "pruned": {"kept": ["CLM-0003"], "dropped": [], "kept_count": 1,
                   "dropped_count": 0, "budget": 1280},
        "tokens": {"before": 100, "after": 40, "savings_percent": 60.0},
        "timings_ms": {"retrieval_ms": 1.0, "rerank_ms": 2.0,
                       "prune_ms": 3.0, "total_ms": 6.0},
    }


def test_render_json_round_trips():
    data = json.loads(render_json(_record()))
    assert data["audit_id"] == "abc123"
    assert data["query"] == "Does claim CLM-0003 have a fraud flag?"


def test_render_html_contains_key_sections():
    html = render_html(_record())
    for needle in ("GraphRAG Audit Trail Report", "Does claim CLM-0003 have a fraud flag?",
                   "Cypher used", "FRAUD_DETECTED", "abc123", "kept / ", "60.0%"):
        assert needle in html, f"missing {needle!r} in report"


def test_render_pdf_valid_header():
    pdf = render_pdf(_record())
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 1000


def test_save_exports_writes_all_kinds(tmp_path, monkeypatch):
    from graphrag import audit_reporter
    # tmp_path is outside the repo ROOT — point both the export dir and the
    # ROOT anchor at it so relative_to() in save_exports succeeds
    monkeypatch.setattr(audit_reporter, "_export_dir", lambda: tmp_path)
    monkeypatch.setattr(audit_reporter, "ROOT", tmp_path)
    out = save_exports(_record())
    assert set(out) == {"json", "html", "pdf"}
    for kind, rel in out.items():
        path = tmp_path / rel.split("/")[-1]
        assert path.exists()
        assert path.stat().st_size > 0
