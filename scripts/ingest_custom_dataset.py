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

from neo4j import GraphDatabase

from graphrag.config import settings
from graphrag.custom_sessions import (
    CUSTOM_DIR,
    build_from_pdfs,
    get_custom_session,
    verify_session_manifest,
)
from graphrag.custom_sessions import (
    PROJECT_ROOT as CS_ROOT,
)
from upload import (
    UploadLimits,
    UploadValidationError,
    revalidate_persisted_uploads,
    safe_record_upload_audit,
    scanner_from_settings,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed a custom session's graph.")
    p.add_argument("name", help="custom session name (from data/custom_sessions.json)")
    p.add_argument("--reset", action="store_true", help="clear only this tenant's graph scope")
    p.add_argument("--tenant", default=settings.DEFAULT_TENANT,
                   help="tenant owner recorded in the custom-session registry")
    p.add_argument("--uri", default=settings.NEO4J_URI)
    p.add_argument("--user", default=settings.NEO4J_USER)
    p.add_argument("--password", default=settings.NEO4J_PASSWORD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = get_custom_session(args.name, args.tenant)
    if not record:
        print(f"ERROR: no custom session named '{args.name}' "
              f"(see data/custom_sessions.json)")
        return 1

    try:
        verify_session_manifest(record)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    sources = [CS_ROOT / src for src in record["sources"]]
    missing = [str(s) for s in sources if not s.exists()]
    if missing:
        print(f"ERROR: source file(s) missing: {', '.join(missing)}")
        return 1

    # Re-validate persisted sources before every ingest. This catches manual
    # registry edits, post-upload tampering, corrupt PDFs, and oversized CSVs
    # before any graph mutation occurs.
    limits = UploadLimits.from_settings(settings)
    audit_path = Path(settings.UPLOAD_AUDIT_PATH)
    if not audit_path.is_absolute():
        audit_path = CS_ROOT / audit_path
    validated = []
    try:
        validated = revalidate_persisted_uploads(
            sources, allowed_root=CUSTOM_DIR,
            recorded_metadata=record.get("uploads") or [], limits=limits,
            malware_scanner=scanner_from_settings(settings),
        )
    except (OSError, UploadValidationError) as exc:
        safe_record_upload_audit(
            audit_path=audit_path, outcome="rejected", actor="system",
            tenant_id=args.tenant, entry_point="custom_ingest",
            uploads=validated, session_name=args.name,
            reason=getattr(exc, "code", "stored_file_error"),
            extra={"message": str(exc)},
        )
        print(f"ERROR: upload validation failed: {exc}")
        return 1

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        print(f"seeding custom session '{args.name}' ({record['kind']}) ...")
        if record["kind"] == "csv":
            mapping = record.get("schema_mapping") or {}
            if mapping.get("status") != "approved":
                print("ERROR: CSV schema mapping has not been human-approved")
                return 1
            from graphrag.relational_ingest import ingest_csv_bundle
            result = ingest_csv_bundle(
                driver, sources, mapping, args.name, reset=args.reset,
                tenant_id=args.tenant, line_cb=print,
            )
        elif record["kind"] == "pdf":
            result = build_from_pdfs(
                driver, sources, args.name, reset=args.reset,
                line_cb=print, tenant_id=args.tenant,
            )
        else:
            print(f"ERROR: unknown custom kind '{record['kind']}'")
            return 1
        safe_record_upload_audit(
            audit_path=audit_path, outcome="ingested", actor="system",
            tenant_id=args.tenant, entry_point="custom_ingest",
            uploads=validated, session_name=args.name, extra={"result": result},
        )
    except Exception:
        safe_record_upload_audit(
            audit_path=audit_path, outcome="failed", actor="system",
            tenant_id=args.tenant, entry_point="custom_ingest",
            uploads=validated, session_name=args.name, reason="ingest_error",
        )
        raise
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
