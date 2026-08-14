#!/usr/bin/env python3
"""Seed the graph for a custom (user-uploaded) session.

Reads the custom-session registry (``data/custom_sessions.json``), re-processes
the session's stored source files (CSV -> generic adapter, PDFs -> the standard
extraction pipeline) and stamps the ``(:Dataset {name})`` marker. This is the
"seed command" the session switcher runs for ``kind == "custom"`` sessions, so
custom datasets support the same switch / re-seed / live-progress flow as the
built-in ones.

Usage:
    .venv/Scripts/python.exe scripts/ingest_custom_dataset.py my_claims --reset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))          # for scripts.* imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # for graphrag.* imports

from neo4j import GraphDatabase  # noqa: E402

from graphrag.config import settings  # noqa: E402
from graphrag.custom_sessions import (  # noqa: E402
    CUSTOM_DIR, PROJECT_ROOT as CS_ROOT, build_from_csv, build_from_pdfs,
    get_custom_session,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed a custom session's graph.")
    p.add_argument("name", help="custom session name (from data/custom_sessions.json)")
    p.add_argument("--reset", action="store_true", help="clear the whole graph first")
    p.add_argument("--uri", default=settings.NEO4J_URI)
    p.add_argument("--user", default=settings.NEO4J_USER)
    p.add_argument("--password", default=settings.NEO4J_PASSWORD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = get_custom_session(args.name)
    if not record:
        print(f"ERROR: no custom session named '{args.name}' "
              f"(see data/custom_sessions.json)")
        return 1

    sources = [CS_ROOT / src for src in record["sources"]]
    missing = [str(s) for s in sources if not s.exists()]
    if missing:
        print(f"ERROR: source file(s) missing: {', '.join(missing)}")
        return 1

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        print(f"seeding custom session '{args.name}' ({record['kind']}) ...")
        if record["kind"] == "csv":
            build_from_csv(driver, sources[0], args.name, reset=args.reset,
                           line_cb=print)
        elif record["kind"] == "pdf":
            build_from_pdfs(driver, sources, args.name, reset=args.reset,
                            line_cb=print)
        else:
            print(f"ERROR: unknown custom kind '{record['kind']}'")
            return 1
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
