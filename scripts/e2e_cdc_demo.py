"""Phase 2 E2E demo — the full CDC upload pipeline against live Neo4j (Shot 1).

Drives the FastAPI ``POST /api/v1/upload`` endpoint end to end:

  1. idempotent re-upload of an unchanged PDF   -> 0 changes
  2. upload of a *modified* policy PDF          -> 1 modified, 2 deleted
       - dropped coverage    -> hard-deleted (single-doc ownership)
       - dropped endorsement -> SKIPPED (still referenced by its own PDF's
                               snapshot — reference counting)
  3. verify the graph state, then re-upload the original PDF to restore it.

Usage:
    .venv/Scripts/python.exe scripts/e2e_cdc_demo.py

Requires: Neo4j running (docker start graphrag-neo4j) and a seeded baseline
(scripts/seed_graph.py --reset --apply-schema --snapshots).
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fastapi.testclient import TestClient  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402

import data_pipeline as dp  # reuse the exact PDF renderer  # noqa: E402
from graphrag.api_server import app  # noqa: E402

POLICY_ID = "POL-0009"  # 2 coverages + END-0007 (whose own PDF/snapshot exists)
NEW_PREMIUM = 56_000.0
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "graphrag-demo")


def _upload(client: TestClient, pdf_bytes: bytes, filename: str) -> dict:
    resp = client.post(
        "/api/v1/upload",
        files={"file": (filename, pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 200, f"upload failed: {resp.text}"
    return resp.json()


def _graph_state(driver) -> dict:
    with driver.session() as s:
        premium = s.run(
            "MATCH (p:Policy {id: $id}) RETURN p.premium AS v", id=POLICY_ID
        ).single()["v"]
        dropped_cov = s.run(
            "MATCH (p:Policy {id: $id})-[:COVERS]->(c:Coverage) "
            "RETURN count(c) AS c",
            id=POLICY_ID,
        ).single()["c"]
        end = s.run(
            "MATCH (e:Endorsement {id: 'END-0007'}) RETURN count(*) AS c"
        ).single()["c"]
        edge = s.run(
            f"MATCH (p:Policy {{id: '{POLICY_ID}'}})-[:ENDORSED_BY]->(e:Endorsement {{id: 'END-0007'}}) "
            "RETURN count(*) AS c"
        ).single()["c"]
        return {"premium": premium, "policy_coverages": dropped_cov, "endorsement_nodes": end, "endorsed_by_edges": edge}


def main() -> int:
    samples = json.loads((PROJECT_ROOT / "data" / "samples" / "policies.json").read_text(encoding="utf-8"))
    original = next(p for p in samples if p["id"] == POLICY_ID)

    driver = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with TestClient(app) as client:
            # ---- 1) idempotent re-upload of the unchanged PDF ----
            original_pdf = (PROJECT_ROOT / "data" / "pdfs" / f"policy_{POLICY_ID}.pdf").read_bytes()
            r1 = _upload(client, original_pdf, f"policy_{POLICY_ID}.pdf")
            assert r1["update_stats"]["entities_added"] == 0
            assert r1["update_stats"]["entities_updated"] == 0
            assert r1["update_stats"]["entities_deleted"] == 0
            print("1) idempotent re-upload            -> 0 changes "
                  f"({r1['update_stats']['update_time_ms']:.1f}ms)  OK")

            # ---- 2) modified PDF: bump premium, drop a coverage, drop END-0007 ----
            modified = copy.deepcopy(original)
            modified["premium"] = NEW_PREMIUM
            second = modified["coverages"][-1]
            dropped_coverage = second["id"]
            modified["coverages"] = [c for c in modified["coverages"] if c["id"] != dropped_coverage]
            modified["endorsements"] = []

            with tempfile.TemporaryDirectory() as tmp:
                dp._pdf_policy(modified, Path(tmp))
                mod_pdf = (Path(tmp) / f"policy_{POLICY_ID}.pdf").read_bytes()

            r2 = _upload(client, mod_pdf, f"policy_{POLICY_ID}.pdf")
            s2 = r2["update_stats"]
            assert s2["entities_updated"] == 1, s2          # Policy premium
            assert s2["entities_deleted"] == 2, s2          # dropped coverage + END-0007
            assert s2["deleted_skipped"] == 1, s2           # END-0007 kept via reference counting
            print("2) modified PDF upload              -> 1 modified, 2 deleted, "
                  f"{s2['deleted_skipped']} skipped (ref-count) "
                  f"({s2['update_time_ms']:.1f}ms)  OK")

            state = _graph_state(driver)
            assert state["premium"] == NEW_PREMIUM, state
            assert state["policy_coverages"] == 1, state        # COV dropped for real
            assert state["endorsement_nodes"] == 1, state       # END-0007 survived
            assert state["endorsed_by_edges"] == 1, state       # edge survives (own PDF re-derives it)
            print("3) graph state                     -> premium updated, dropped coverage gone, "
                  "END-0007 + edge kept  OK")

            # ---- 3) restore: re-upload the original ----
            r3 = _upload(client, original_pdf, f"policy_{POLICY_ID}.pdf")
            assert r3["update_stats"]["entities_added"] == 2, r3["update_stats"]
            state = _graph_state(driver)
            assert state["premium"] == original["premium"], state
            assert state["policy_coverages"] == 2, state
            print("4) restore with original PDF       -> 2 re-added, graph back to baseline  OK")

        print("\nE2E CDC demo: ALL CHECKS PASSED")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
