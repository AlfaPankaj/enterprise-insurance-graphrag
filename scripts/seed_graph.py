#!/usr/bin/env python3
"""
GraphRAG Insurance Claims System — Seed the Neo4j graph from the Phase 1 samples.

Reads data/samples/*.json and creates the knowledge graph per docs/graph_schema.md:

    (Policyholder)-[:HAS_POLICY]->(Policy)-[:COVERS]->(Coverage)
    (Policy)-[:HAS_CLAIM]->(Claim)-[:FRAUD_DETECTED]->(FraudFlag)
    (Policy)-[:ENDORSED_BY]->(Endorsement)
    (Investigator)-[:INVESTIGATES_CLAIM]->(Claim)

Idempotent: every entity is MERGE'd on its stable id, so re-running never
duplicates nodes or edges.

Usage:
    .venv/Scripts/python.exe scripts/seed_graph.py --apply-schema
    .venv/Scripts/python.exe scripts/seed_graph.py --reset --apply-schema --snapshots --verify-constraints
    NEO4J_PASSWORD=secret .venv/Scripts/python.exe scripts/seed_graph.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from neo4j import GraphDatabase
from neo4j.exceptions import ConstraintError, Neo4jError

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BATCH_SIZE = 500


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed the Neo4j graph from data/samples/*.json (idempotent).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Bolt URI")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", "graphrag-demo"),
                        help="Neo4j password (or set NEO4J_PASSWORD env var)")
    parser.add_argument("--reset", action="store_true",
                        help="DETACH DELETE all nodes before seeding (clean slate)")
    parser.add_argument("--samples-dir", type=Path, default=PROJECT_ROOT / "data" / "samples")
    parser.add_argument("--schema-file", type=Path, default=PROJECT_ROOT / "docs" / "schema.cypher")
    parser.add_argument("--apply-schema", action="store_true",
                        help="Apply constraints/indexes from docs/schema.cypher before seeding")
    parser.add_argument("--verify-constraints", action="store_true",
                        help="Assert that duplicate entity ids are rejected by the constraints")
    parser.add_argument("--snapshots", action="store_true",
                        help="Create per-document CDC snapshots by extracting data/pdfs/*.pdf")
    return parser.parse_args(argv)


def split_cypher(text: str) -> list[str]:
    """Split a .cypher file into executable statements (strip block/line comments)."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    statements: list[str] = []
    for stmt in text.split(";"):
        cleaned = "\n".join(
            line for line in stmt.splitlines() if not line.strip().startswith("//")
        ).strip()
        if cleaned:
            statements.append(cleaned)
    return statements


def load_json(samples_dir: Path, name: str) -> list:
    path = samples_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {name} in {samples_dir} — run scripts/data_pipeline.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def node_props(data: dict, exclude: frozenset[str] = frozenset()) -> dict:
    """Keep only scalar properties (drop nested structures and join fields)."""
    return {k: v for k, v in data.items() if k not in exclude and not isinstance(v, (dict, list))}


def batches(rows: Iterable[Any], size: int = BATCH_SIZE) -> Iterable[list]:
    buffer = list(rows)
    for start in range(0, len(buffer), size):
        yield buffer[start : start + size]


def clear_graph_batched(session) -> None:
    """Delete the whole graph in per-label, LIMIT-chunked batches.

    A single ``MATCH (n) DETACH DELETE n`` on a 120k-node graph (e.g. right
    after a full data_synthetic ingest) can exceed the server's per-transaction
    memory limit. Deleting label-by-label with a small LIMIT keeps each
    transaction tiny and idempotent (re-running is safe).
    """
    labels = [r["label"] for r in session.run(
        "MATCH (n) WITH labels(n) AS l UNWIND l AS label "
        "RETURN DISTINCT label ORDER BY label").data()]
    for label in labels:
        while True:
            res = session.run(
                f"MATCH (n:{label}) WITH n LIMIT 2000 DETACH DELETE n "
                "RETURN count(*) AS c").single()
            if not res or res["c"] == 0:
                break
    # anything without a label (defensive)
    while True:
        res = session.run(
            "MATCH (n) WHERE size(labels(n)) = 0 WITH n LIMIT 2000 "
            "DETACH DELETE n RETURN count(*) AS c").single()
        if not res or res["c"] == 0:
            break


def apply_schema(session, schema_path: Path) -> None:
    statements = split_cypher(schema_path.read_text(encoding="utf-8"))
    print(f"  applying {len(statements)} statements from {schema_path.name} ...")
    for stmt in statements:
        session.run(stmt)
    print("  schema (constraints + indexes) applied")


def load_nodes(runner, nodes_by_label: dict[str, list[dict]]) -> None:
    """runner is a Session or an explicit Transaction (both expose .run)."""
    for label, rows in nodes_by_label.items():
        if not rows:
            continue
        for batch in batches(rows):
            runner.run(
                f"UNWIND $rows AS r MERGE (n:{label} {{id: r.id}}) SET n += r.props",
                rows=batch,
            )


def load_relationships(runner, rels: list[tuple[str, str, str, str, str]]) -> None:
    """rels: (a_label, b_label, a_id, b_id, relationship_type)"""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for a_label, b_label, a_id, b_id, rtype in rels:
        groups[(a_label, b_label, rtype)].append({"a": a_id, "b": b_id})
    for (a_label, b_label, rtype), rows in groups.items():
        if not rows:
            continue
        for batch in batches(rows):
            runner.run(
                f"UNWIND $rows AS r MATCH (a:{a_label} {{id: r.a}}), (b:{b_label} {{id: r.b}}) "
                f"MERGE (a)-[:{rtype}]->(b)",
                rows=batch,
            )


def seed(session, samples_dir: Path) -> None:
    policies = load_json(samples_dir, "policies.json")
    claims = load_json(samples_dir, "claims.json")
    endorsements = load_json(samples_dir, "endorsements.json")

    nodes: dict[str, list[dict]] = defaultdict(list)
    rels: list[tuple[str, str, str, str, str]] = []
    counter: Counter = Counter()

    for pol in policies:
        ph = pol["policyholder"]
        p = node_props(pol)
        nodes["Policyholder"].append({"id": ph["id"], "props": ph})
        nodes["Policy"].append({"id": p["id"], "props": p})
        rels.append(("Policyholder", "Policy", ph["id"], p["id"], "HAS_POLICY"))
        counter["policies"] += 1
        for cov in pol["coverages"]:
            nodes["Coverage"].append({"id": cov["id"], "props": cov})
            rels.append(("Policy", "Coverage", p["id"], cov["id"], "COVERS"))
        for end in pol["endorsements"]:
            nodes["Endorsement"].append(
                {"id": end["id"], "props": node_props(end, {"policy_id"})}
            )
            rels.append(("Policy", "Endorsement", p["id"], end["id"], "ENDORSED_BY"))

    for record in claims:
        claim = record["claim"]
        nodes["Claim"].append(
            {"id": claim["id"], "props": node_props(claim, {"policy_id", "doc_id"})}
        )
        rels.append(("Policy", "Claim", claim["policy_id"], claim["id"], "HAS_CLAIM"))
        counter["claims"] += 1
        flag = record.get("fraud_flag")
        if flag:
            nodes["FraudFlag"].append({"id": flag["id"], "props": flag})
            rels.append(("Claim", "FraudFlag", claim["id"], flag["id"], "FRAUD_DETECTED"))
            counter["fraud_flags"] += 1
        investigator = record.get("investigator")
        if investigator:
            nodes["Investigator"].append({"id": investigator["id"], "props": investigator})
            rels.append(
                ("Investigator", "Claim", investigator["id"], claim["id"], "INVESTIGATES_CLAIM")
            )

    for end in endorsements:
        # MERGE by id dedupes against the endorsements nested in policies.json
        nodes["Endorsement"].append(
            {"id": end["id"], "props": node_props(end, {"policy_id"})}
        )
        rels.append(("Policy", "Endorsement", end["policy_id"], end["id"], "ENDORSED_BY"))

    # Whole load in one transaction: a failure rolls back everything (idempotent
    # MERGEs make re-running the recovery path either way).
    with session.begin_transaction() as tx:
        load_nodes(tx, nodes)
        load_relationships(tx, rels)

    # Queued counts are pre-dedupe (endorsements appear in policies.json AND
    # endorsements.json; investigators repeat across claims) — the graph counts
    # reported afterwards are the authoritative numbers.
    print(f"  nodes queued: {sum(len(v) for v in nodes.values())} "
          f"({', '.join(f'{k}={len(v)}' for k, v in sorted(nodes.items()))})")
    print(f"  relationships queued: {len(rels)}")
    print(f"  claims: {counter['claims']} | policies: {counter['policies']} | "
          f"fraud flags: {counter['fraud_flags']}")


def create_snapshots(driver, pdfs_dir: Path, samples_dir: Path) -> None:
    """Baseline the CDC snapshots by running the extractor over the generated PDFs.

    The snapshot for a doc is exactly what the extractor produces for its PDF, so
    re-uploading an unchanged PDF yields zero CDC changes.
    """
    import sys

    src = PROJECT_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from graphrag.entity_extractor import extract_entities
    from graphrag.graph_store import save_existing_entities
    from graphrag.pdf_processor import extract_text_from_pdf

    pdfs = sorted(pdfs_dir.glob("*.pdf"))
    if not pdfs:
        print("  no PDFs found; skipping snapshots")
        return
    with driver.session() as session:
        with session.begin_transaction() as tx:
            for pdf in pdfs:
                text = extract_text_from_pdf(pdf.read_bytes())
                result = extract_entities(text, doc_id_hint=pdf.name)
                if not result["entities"]:
                    print(f"  WARN: no entities extracted from {pdf.name}")
                    continue
                save_existing_entities(tx, result["doc_id"] or pdf.stem, result["entities"])
    print(f"  CDC snapshots created for {len(pdfs)} PDFs (mode={pdfs and 'heuristic'})")


def report_counts(session) -> None:
    node_counts = session.run(
        "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY label"
    ).data()
    print("  graph nodes:", ", ".join(f"{r['label']}={r['c']}" for r in node_counts))
    rel_counts = session.run(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS c ORDER BY type"
    ).data()
    print("  graph relationships:", ", ".join(f"{r['type']}={r['c']}" for r in rel_counts))


def verify_constraints(session) -> None:
    """Insert duplicate ids; a raised constraint error proves the constraint works."""
    print("  verifying uniqueness constraints (expected failures) ...")
    checks = [
        ("Policy", {"id": "POL-0001", "policy_number": "DUP", "type": "COMMERCIAL_GENERAL_LIABILITY"}),
        ("Claim", {"id": "CLM-0001", "claim_number": "DUP", "amount": 1.0}),
        ("Policyholder", {"id": "PH-0001", "name": "DUP"}),
    ]
    for label, props in checks:
        try:
            session.run(f"CREATE (n:{label} $props)", props=props)
        except ConstraintError:
            print(f"  OK: duplicate {label} id rejected (ConstraintError)")
            continue
        except Neo4jError as e:
            print(f"  WARN: {label} probe raised non-constraint error: {type(e).__name__}: {e}")
            continue
        # CREATE succeeded -> the id did not exist, so the constraint did not
        # fire; remove the probe node so no junk is left behind.
        session.run(f"MATCH (n:{label} $props) DETACH DELETE n", props=props)
        print(f"  FAIL: duplicate {label} id was NOT rejected (probe node cleaned up)")

    print("  registered constraints:")
    for r in session.run("SHOW CONSTRAINTS YIELD name, type, labelsOrTypes").data():
        print(f"    - {r['name']} ({r['type']}) on {r['labelsOrTypes']}")
    print(f"  registered indexes: {len(session.run('SHOW INDEXES YIELD name').data())}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session() as session:
            session.run("RETURN 1")  # connectivity check
            print(f"connected to {args.uri}")
            if args.apply_schema:
                apply_schema(session, args.schema_file)
            if args.reset:
                clear_graph_batched(session)
                print("  graph cleared (--reset)")
            seed(session, args.samples_dir)
            # stamp the demo dataset so the dashboard picks the right fraud
            # ground-truth table (samples/claims.json labels) — but never
            # overwrite a real-dataset marker (e.g. when adding snapshots to a
            # graph that was seeded from a real CSV without --reset)
            if args.reset or not session.run(
                "MATCH (d:Dataset) RETURN d LIMIT 1"
            ).single():
                session.run("MERGE (d:Dataset {name: 'synthetic'})")
            report_counts(session)
            if args.snapshots:
                create_snapshots(driver, PROJECT_ROOT / "data" / "pdfs", args.samples_dir)
            if args.verify_constraints:
                verify_constraints(session)
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
