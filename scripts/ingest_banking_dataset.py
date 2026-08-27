#!/usr/bin/env python3
"""Ingest the banking demo dataset (v2 — WS-E, G18).

Loads ``data/samples/banking.json`` into Neo4j with the banking ontology:

    (Customer)-[:HOLDS]->(Account)-[:POSTED]->(Transaction)
    (Account)-[:HAS_DISPUTE]->(Dispute)-[:ABOUT]->(Transaction)
    (Account)-[:HAS_ALERT]->(AMLAlert)

Same machinery as the other sessions: tenant stamping + PII at-rest
encryption flow through ``load_nodes``; the ``(:Dataset {name: 'banking'})``
marker (with a revision bump) is stamped for session detection and cache
invalidation. This is the seed command the session switcher runs for the
``banking_demo`` session.

Usage:
    python scripts/ingest_banking_dataset.py --reset
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))          # scripts.* imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # graphrag.* imports

from graphrag.config import settings  # noqa: E402
from scripts.seed_graph import load_nodes, load_relationships  # noqa: E402

SAMPLE = PROJECT_ROOT / "data" / "samples" / "banking.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reset", action="store_true", help="clear the graph first")
    p.add_argument("--uri", default=settings.NEO4J_URI)
    p.add_argument("--user", default=settings.NEO4J_USER)
    p.add_argument("--password", default=settings.NEO4J_PASSWORD)
    return p.parse_args(argv)


def adapt_banking(data: dict) -> tuple[dict[str, list[dict]], list[tuple]]:
    """banking.json -> ({label: [{"id", "props"}]}, rel tuples)."""
    nodes: dict[str, list[dict]] = {}
    rels: list[tuple] = []
    counter: Counter = Counter()

    for rec in data["customers"]:
        nodes.setdefault("Customer", []).append({"id": rec["id"], "props": rec})
        counter["customers"] += 1

    for rec in data["accounts"]:
        props = {k: v for k, v in rec.items() if k != "customer_id"}
        nodes.setdefault("Account", []).append({"id": rec["id"], "props": props})
        rels.append(("Customer", "Account", rec["customer_id"], rec["id"], "HOLDS"))
        counter["accounts"] += 1

    for rec in data["transactions"]:
        props = {k: v for k, v in rec.items() if k != "account_id"}
        nodes.setdefault("Transaction", []).append({"id": rec["id"], "props": props})
        rels.append(("Account", "Transaction", rec["account_id"], rec["id"], "POSTED"))
        counter["transactions"] += 1

    for rec in data["disputes"]:
        props = {k: v for k, v in rec.items()
                 if k not in ("account_id", "transaction_id")}
        nodes.setdefault("Dispute", []).append({"id": rec["id"], "props": props})
        rels.append(("Account", "Dispute", rec["account_id"], rec["id"], "HAS_DISPUTE"))
        rels.append(("Dispute", "Transaction", rec["id"], rec["transaction_id"], "ABOUT"))
        counter["disputes"] += 1

    for rec in data["aml_alerts"]:
        props = {k: v for k, v in rec.items()
                 if k not in ("account_id", "customer_id")}
        nodes.setdefault("AMLAlert", []).append({"id": rec["id"], "props": props})
        rels.append(("Account", "AMLAlert", rec["account_id"], rec["id"], "HAS_ALERT"))
        counter["aml_alerts"] += 1

    return nodes, rels


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not SAMPLE.exists():
        print(f"ERROR: {SAMPLE} missing — run scripts/data_pipeline_banking.py first")
        return 1
    data = json.loads(SAMPLE.read_text(encoding="utf-8"))

    nodes, rels = adapt_banking(data)
    tenant = settings.DEFAULT_TENANT if settings.TENANT_MODE == "column" else None

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session() as session:
            if args.reset:
                from scripts.seed_graph import clear_graph_batched
                clear_graph_batched(session)
                print("  graph cleared (--reset)")
            if tenant:
                print(f"  stamping tenant_id={tenant} (TENANT_MODE=column)")
            load_nodes(session, nodes, tenant_id=tenant)
            load_relationships(session, rels)
            session.run(
                "MERGE (d:Dataset {name: 'banking'}) "
                "ON CREATE SET d.rev = 0 "
                "ON MATCH SET d.rev = coalesce(d.rev, 0) + 1"
            )
            counts = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY label"
            ).data()
            rel_counts = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t"
            ).data()
        print("  graph nodes:", ", ".join(f"{r['label']}={r['c']}" for r in counts))
        print("  relationships:", ", ".join(f"{r['t']}={r['c']}" for r in rel_counts))
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
