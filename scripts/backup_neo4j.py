"""Neo4j backup (Phase 5): portable graph export to JSON.

Two modes:

* **cypher (default)** — streams every node + relationship via the Python
  driver into a timestamped JSON file. Works against any Neo4j (including the
  docker-compose container) with no shell access. Restore = re-seed from the
  JSON's node properties (the export carries labels + full property maps; a
  dedicated restore script is a follow-up). This is the primary mode.
* **admin** — wraps ``neo4j-admin database dump`` for deployments where the
  operator has host-level access to the database (true byte-level backup).

Usage:
    .venv/Scripts/python.exe scripts/backup_neo4j.py
    .venv/Scripts/python.exe scripts/backup_neo4j.py --out data/backups
    .venv/Scripts/python.exe scripts/backup_neo4j.py --mode admin --neo4j-home /var/lib/neo4j
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graphrag.config import settings  # noqa: E402


def export_to_json(driver, out_dir: Path) -> Path:
    """Dump all nodes + relationships to a timestamped JSON backup file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"graph_backup_{stamp}.json"

    nodes, rels = [], []
    with driver.session() as session:
        for row in session.run(
            "MATCH (n) RETURN labels(n) AS labels, properties(n) AS props, "
            "elementId(n) AS el"
        ):
            nodes.append({"labels": row["labels"], "props": row["props"]})
        for row in session.run(
            "MATCH (a)-[r]->(b) RETURN elementId(a) AS src, elementId(b) AS dst, "
            "type(r) AS type, properties(r) AS props"
        ):
            rels.append({"source": row["src"], "target": row["dst"],
                         "type": row["type"], "props": row["props"]})

    payload = {
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "node_count": len(nodes),
        "relationship_count": len(rels),
        "nodes": nodes,
        "relationships": rels,
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def admin_dump(neo4j_home: Path, out_dir: Path, database: str = "neo4j") -> Path:
    """Byte-level backup via ``neo4j-admin database dump`` (host access needed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = out_dir / f"graph_backup_{stamp}.dump"
    cmd = [str(neo4j_home / "bin" / "neo4j-admin"), "database", "dump",
           database, "--to-path", str(out_dir), "--overwrite-destination"]
    subprocess.run(cmd, check=True, capture_output=True)
    # neo4j-admin names the file itself; find the newest .dump in the dir
    newest = max(out_dir.glob("*.dump"), key=lambda p: p.stat().st_mtime)
    return newest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "backups")
    p.add_argument("--mode", choices=["cypher", "admin"], default="cypher")
    p.add_argument("--neo4j-home", type=Path, default=Path("/var/lib/neo4j"))
    p.add_argument("--uri", default=settings.NEO4J_URI)
    p.add_argument("--user", default=settings.NEO4J_USER)
    p.add_argument("--password", default=settings.NEO4J_PASSWORD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()
    if args.mode == "admin":
        out = admin_dump(args.neo4j_home, args.out)
        print(f"admin dump: {out} ({(out.stat().st_size / 1024 / 1024):.1f} MB)")
        return 0

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        out = export_to_json(driver, args.out)
    finally:
        driver.close()
    elapsed = time.perf_counter() - t0
    print(f"cypher backup: {out} ({out.stat().st_size / 1024:.0f} KB, "
          f"{elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
