#!/usr/bin/env python3
"""Migrate the graph from Neo4j into PostgreSQL/Apache AGE (Option A).

One-shot, idempotent-by-MERGE copy of every node and edge from the running
Neo4j graph into an AGE graph, with count verification at the end. Run the
standard seeding first (Neo4j stays the source of truth), then:

    python scripts/migrate_to_age.py --dsn postgresql://graphrag:graphrag@localhost:5432/graphrag

See docs/DEPLOYMENT_SCALE_OPTIONS.md for the full Option-A walkthrough.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphrag.config import settings  # noqa: E402

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# labels whose unique key is not ``id`` (must mirror the write paths)
_KEY_PROPS = {"Dataset": "name", "DocSnapshot": "doc_id"}


def _safe_label(label: str) -> str:
    if not _IDENT.match(label or ""):
        raise ValueError(f"unsafe label from source graph: {label!r}")
    return label


def _count(session, cypher: str) -> int:
    row = session.run(cypher).single()
    return int(row["c"]) if row else 0


def migrate(neo4j_uri, user, password, dsn, graph, batch, reset) -> int:
    from neo4j import GraphDatabase
    from graphrag.postgres_backend import AgeDriver, ensure_graph, graph_exists

    src = GraphDatabase.driver(neo4j_uri, auth=(user, password))
    dst = AgeDriver(dsn=dsn, graph=graph)
    try:
        with dst.session() as out:
            conn = out._conn
            if reset and graph_exists(conn, graph):
                with conn.cursor() as cur:
                    cur.execute("SELECT ag_catalog.drop_graph(%s, true)", (graph,))
                conn.commit()
                print(f"dropped existing AGE graph {graph!r}")
            ensure_graph(conn, graph)

            # ---- nodes (bucketed by label; MERGE on the label's key) -----
            with src.session() as s:
                total_nodes = _count(s, "MATCH (n) RETURN count(*) AS c")
            buckets: dict[str, list] = {}
            skipped = 0
            skip = 0
            while True:
                with src.session() as s:
                    page = s.run(
                        "MATCH (n) RETURN labels(n)[0] AS label, properties(n) AS props "
                        "ORDER BY label SKIP $skip LIMIT $lim",
                        skip=skip, lim=batch,
                    ).data()
                if not page:
                    break
                skip += len(page)
                for r in page:
                    props = r["props"] or {}
                    key = _KEY_PROPS.get(r["label"], "id")
                    if props.get(key) is not None:
                        buckets.setdefault(r["label"], []).append(
                            {"k": str(props[key]), "p": props})
                    else:
                        skipped += 1
            migrated = 0
            for label, rows in buckets.items():
                key = _KEY_PROPS.get(label, "id")
                for i in range(0, len(rows), batch):
                    chunk = rows[i:i + batch]
                    out.run(
                        f"UNWIND $rows AS row MERGE (n:{_safe_label(label)} "
                        f"{{{key}: row.k}}) SET n = row.p",
                        rows=chunk,
                    )
                    migrated += len(chunk)
            print(f"nodes: migrated {migrated}, skipped {skipped} "
                  f"(of {total_nodes} source nodes)")

            # ---- edges (bucketed by type — Cypher types are static) ------
            with src.session() as s:
                total_edges = _count(s, "MATCH ()-[r]->() RETURN count(*) AS c")
            edge_buckets: dict[str, list] = {}
            skip = 0
            while True:
                with src.session() as s:
                    page = s.run(
                        "MATCH (a)-[r]->(b) WHERE a.id IS NOT NULL AND b.id IS NOT NULL "
                        "RETURN labels(a)[0] AS la, a.id AS ia, labels(b)[0] AS lb, "
                        "b.id AS ib, type(r) AS rel, properties(r) AS props "
                        "ORDER BY rel SKIP $skip LIMIT $lim",
                        skip=skip, lim=batch,
                    ).data()
                if not page:
                    break
                skip += len(page)
                for p in page:
                    edge_buckets.setdefault(p["rel"], []).append(
                        {"ia": str(p["ia"]), "ib": str(p["ib"]),
                         "props": p["props"] or {}})
            migrated_e = 0
            for rel, rows in edge_buckets.items():
                for i in range(0, len(rows), batch):
                    chunk = rows[i:i + batch]
                    out.run(
                        f"UNWIND $rows AS row "
                        "MATCH (a {id: row.ia}) MATCH (b {id: row.ib}) "
                        f"MERGE (a)-[r:{_safe_label(rel)}]->(b) SET r = row.props",
                        rows=chunk,
                    )
                    migrated_e += len(chunk)
            print(f"edges: migrated {migrated_e} (of {total_edges} source edges)")

            # ---- verify --------------------------------------------------
            n_dst = _count(out, "MATCH (n) RETURN count(*) AS c")
            e_dst = _count(out, "MATCH ()-[r]->() RETURN count(*) AS c")
            print(f"verify: AGE graph holds {n_dst} nodes / {e_dst} edges "
                  f"(source: {total_nodes} nodes / {total_edges} edges)")
            if n_dst + skipped < total_nodes:
                print("WARNING: node count mismatch — inspect skipped nodes")
                return 1
    finally:
        src.close()
        dst.close()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uri", default=settings.NEO4J_URI)
    p.add_argument("--user", default=settings.NEO4J_USER)
    p.add_argument("--password", default=settings.NEO4J_PASSWORD)
    p.add_argument("--dsn", default=settings.POSTGRES_DSN,
                   help="target PostgreSQL DSN (POSTGRES_DSN by default)")
    p.add_argument("--graph", default=settings.AGE_GRAPH_NAME)
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--reset", action="store_true",
                   help="drop the target AGE graph first")
    args = p.parse_args(argv)
    if not args.dsn:
        p.error("--dsn / POSTGRES_DSN is required (see .env.example)")
    return migrate(args.uri, args.user, args.password, args.dsn,
                   args.graph, args.batch, args.reset)


if __name__ == "__main__":
    raise SystemExit(main())
