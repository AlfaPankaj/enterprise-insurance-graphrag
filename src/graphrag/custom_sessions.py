"""Custom user-uploaded sessions (Phase 6) — bring your own PDF/CSV dataset.

Users can upload their own file (a PDF or a CSV), give it a **session name**
(which must not collide with the built-in sessions), and the pipeline builds a
graph for it — then the session selector / API can switch to it like any other
session and the user can query their own data.

Sessions are persisted in ``data/custom_sessions.json``::

    [{"name": "my_claims", "kind": "csv", "sources": ["data/custom/my_claims/claims.csv"],
      "created_at": "2026-08-13T…", "note": "12 rows · 2 fraud flags"}]

Processing (run by ``scripts/ingest_custom_dataset.py``, so it streams through
the normal session-switch path):

  * **CSV** — bounded profiling proposes a relational schema; strict human
    approval gates streaming node/FK writes. The legacy single-file adapter is
    retained only for backwards compatibility with old registry records.
  * **PDF**  — the standard extraction pipeline (pdf_processor →
    entity_extractor → graph_updater) with entity-derived edges.

Both stamp a ``(:Dataset {name})`` marker so session detection, the dashboard
and the audit UI work unchanged. The pure logic here (validation, the CSV
adapter) is unit-testable without Neo4j.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from graphrag.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "data" / "custom_sessions.json"
CUSTOM_DIR = PROJECT_ROOT / "data" / "custom"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_ -]{0,47}$")
_REGISTRY_LOCK = threading.RLock()


@contextmanager
def _registry_file_lock():
    """Serialize registry read-modify-write cycles across app/worker processes."""
    lock_path = REGISTRY_PATH.with_suffix(REGISTRY_PATH.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows uses process-local lock
            fcntl = None
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def tenant_storage_dir(tenant_id: str, session_name: str) -> Path:
    """Non-guessable, traversal-safe storage directory for a tenant session."""
    tenant_key = hashlib.sha256(
        str(tenant_id or settings.DEFAULT_TENANT).encode("utf-8")
    ).hexdigest()[:16]
    safe_session = re.sub(r"[^A-Za-z0-9_-]+", "_", session_name).strip("_")
    return CUSTOM_DIR / tenant_key / safe_session


def _builtin_session_ids() -> frozenset[str]:
    """Ids of the built-in sessions (imported lazily to avoid a cycle)."""
    from graphrag.sessions import SESSIONS

    return frozenset(s["id"] for s in SESSIONS)

# --- CSV generic adapter heuristics ----------------------------------------

_CLAIM_COL_RE = re.compile(r"claim|fraud|amount|loss|incident|coverage")
_FRAUD_COL_RE = re.compile(r"fraud")
_ID_COL_RE = re.compile(r"(^|_)(id|number|no)$|_id$|^id$")
_TRUE_VALUES = {"1", "y", "yes", "true", "fraud", "fraudulent", "flagged"}


def _to_prop(value: str):
    """Numbers become floats (so threshold queries work); else a string."""
    s = str(value).strip()
    if s == "":
        return None
    compact = s.replace("$", "").replace(",", "")
    try:
        return float(compact) if re.fullmatch(r"-?\d+(\.\d+)?", compact) else s
    except ValueError:
        return s


def adapt_csv_to_graph(csv_path: Path) -> tuple[dict[str, list[dict]], list[tuple]]:
    """Map an arbitrary CSV onto the graph schema (pure, no DB access).

    Returns ``(nodes: {label: [{"id", "props"}]}, rels: [(a,b,a_id,b_id,type)])``.

    Heuristic: if any column looks claim-like (claim/fraud/amount/loss/
    incident/coverage) the CSV becomes ``(:Claim)`` rows — with a
    ``(:FraudFlag)`` + ``FRAUD_DETECTED`` edge per row whose fraud column is
    truthy (1/Y/yes/true/fraud/...). Otherwise each row becomes a generic
    ``(:Record)`` node. The id column (``*_id`` / ``*_number`` / ``id``) is
    used for node ids when present; otherwise zero-padded ``CLM-``/``REC-`` ids
    are generated (which match the retriever's id regex for exact-id queries).
    """
    nodes: dict[str, list[dict]] = {}
    rels: list[tuple] = []
    with csv_path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.DictReader(fh)
        headers = [h.strip() for h in (reader.fieldnames or [])]
        if not headers:
            return nodes, rels
        lower = {h: h.lower() for h in headers}
        rows = list(reader)

    is_claims = any(_CLAIM_COL_RE.search(lower[h]) for h in headers)
    id_col = next((h for h in headers if _ID_COL_RE.search(lower[h])), None)
    fraud_col = next((h for h in headers if _FRAUD_COL_RE.search(lower[h])), None)

    if is_claims:
        claim_nodes: list[dict] = []
        flag_nodes: list[dict] = []
        for i, row in enumerate(rows, start=1):
            cid = str(row.get(id_col) or "").strip() or f"CLM-{i:05d}"
            props = {h: _to_prop(row.get(h)) for h in headers if h != id_col}
            props = {k: v for k, v in props.items() if v is not None}
            props["id"] = cid
            claim_nodes.append({"id": cid, "props": props})
            flagged = str(row.get(fraud_col, "")).strip().lower() in _TRUE_VALUES if fraud_col else False
            if flagged:
                fid = f"FRD-{i:05d}"
                flag_nodes.append({
                    "id": fid,
                    "props": {"id": fid, "claim_id": cid,
                              "reason": "flagged in source CSV"},
                })
                rels.append(("Claim", "FraudFlag", cid, fid, "FRAUD_DETECTED"))
        nodes["Claim"] = claim_nodes
        if flag_nodes:
            nodes["FraudFlag"] = flag_nodes
    else:
        record_nodes: list[dict] = []
        for i, row in enumerate(rows, start=1):
            rid = str(row.get(id_col) or "").strip() or f"REC-{i:05d}"
            props = {h: _to_prop(row.get(h)) for h in headers if h != id_col}
            props = {k: v for k, v in props.items() if v is not None}
            props["id"] = rid
            record_nodes.append({"id": rid, "props": props})
        nodes["Record"] = record_nodes

    return nodes, rels


# ---------------------------------------------------------------------------
# registry (data/custom_sessions.json)
# ---------------------------------------------------------------------------

def _load_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_registry(records: list[dict]) -> None:
    """Atomically replace the registry so interrupted writes cannot corrupt it."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{REGISTRY_PATH.name}-", suffix=".tmp", dir=REGISTRY_PATH.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, REGISTRY_PATH)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temp_name).unlink(missing_ok=True)
        raise


def _owned(record: dict, tenant_id: str | None) -> bool:
    """Legacy records belong to DEFAULT_TENANT; explicit lookups never cross tenants."""
    return tenant_id is None or record.get("tenant_id", settings.DEFAULT_TENANT) == tenant_id


def list_custom_sessions(tenant_id: str | None = None) -> list[dict]:
    """Registered custom sessions visible to ``tenant_id``.

    ``None`` retains the administrative/backwards-compatible all-tenant view;
    user-facing callers must always pass the authenticated tenant.
    """
    with _REGISTRY_LOCK, _registry_file_lock():
        return [r for r in _load_registry() if _owned(r, tenant_id)]


def get_custom_session(name: str, tenant_id: str | None = None) -> dict | None:
    with _REGISTRY_LOCK, _registry_file_lock():
        matches = [r for r in _load_registry() if r["name"] == name and _owned(r, tenant_id)]
    if tenant_id is None and len(matches) > 1:
        # Preserve deterministic behavior for old CLI callers while preventing
        # accidental cross-tenant selection in new paths.
        return next((r for r in matches
                     if r.get("tenant_id", settings.DEFAULT_TENANT) == settings.DEFAULT_TENANT),
                    None)
    return matches[0] if matches else None


def _build_manifest(kind: str, sources: list[str], uploads: list[dict],
                    tenant_id: str) -> dict:
    files = []
    for index, source in enumerate(sources):
        metadata = dict(uploads[index]) if index < len(uploads) else {}
        files.append({"source": source, **metadata})
    body = {"version": 1, "kind": kind, "tenant_id": tenant_id, "files": files}
    body["sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return body


def validate_session_name(name: str, exclude: str | None = None,
                          tenant_id: str | None = None) -> str:
    """Validate a name and enforce uniqueness within one tenant."""
    cleaned = " ".join(str(name).split())
    if not cleaned:
        raise ValueError("Session name cannot be empty.")
    if not _NAME_RE.fullmatch(cleaned):
        raise ValueError(
            "Session name must be 1-48 characters: letters, digits, spaces, "
            "underscores or hyphens (e.g. 'my_claims')."
        )
    if cleaned in _builtin_session_ids():
        raise ValueError(
            f"'{cleaned}' is a built-in session — pick a different name."
        )
    tenant_id = tenant_id or settings.DEFAULT_TENANT
    if cleaned != exclude and any(
        r["name"] == cleaned and _owned(r, tenant_id) for r in _load_registry()
    ):
        raise ValueError(f"A custom session named '{cleaned}' already exists.")
    return cleaned


def add_custom_session(name: str, kind: str, sources: list[str],
                       note: str = "", uploads: list[dict] | None = None,
                       *, tenant_id: str | None = None,
                       owner: str | None = None) -> dict:
    """Register a tenant-owned custom session and its atomic upload manifest."""
    tenant_id = tenant_id or settings.DEFAULT_TENANT
    with _REGISTRY_LOCK, _registry_file_lock():
        name = validate_session_name(name, tenant_id=tenant_id)
        if kind not in ("csv", "pdf"):
            raise ValueError("kind must be 'csv' or 'pdf'")
        if not sources:
            raise ValueError("at least one source file is required")
        custom_root = CUSTOM_DIR.resolve()
        tenant_root = tenant_storage_dir(tenant_id, name).resolve()
        for src in sources:
            path = PROJECT_ROOT / src
            if not path.exists():
                raise ValueError(f"source file missing: {src}")
            if path.suffix.lower() != f".{kind}":
                raise ValueError(f"source file type does not match session kind: {src}")
            resolved = path.resolve()
            if resolved.is_relative_to(custom_root) and not resolved.is_relative_to(tenant_root):
                raise ValueError("Custom-session source belongs to another tenant")
        if uploads is not None and len(uploads) != len(sources):
            raise ValueError("upload metadata must match the source files")
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        manifest_body = _build_manifest(
            kind, list(sources), list(uploads or []), tenant_id
        )
        record = {
            "name": name,
            "kind": kind,
            "sources": list(sources),
            "manifest": manifest_body,
            "created_at": now,
            "updated_at": now,
            "note": note,
            "uploads": list(uploads or []),
            "tenant_id": tenant_id,
            "owner": owner or "unknown",
            "ingestion": {
                "status": "awaiting_profile" if kind == "csv" else "ready",
                "stage": "persisted",
                "warnings": [],
                "cost_usd": 0.0,
                "timings_ms": {},
            },
        }
        records = _load_registry()
        records.append(record)
        _save_registry(records)
        return record


def verify_session_manifest(record: dict) -> None:
    """Detect registry/source-list tampering before an ingest starts."""
    manifest = record.get("manifest")
    if not manifest:  # backwards compatibility for pre-manifest sessions
        return
    body = dict(manifest)
    expected = str(body.pop("sha256", ""))
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not expected or not hmac.compare_digest(expected, actual):
        raise ValueError("custom-session manifest integrity check failed")
    if [item.get("source") for item in body.get("files", [])] != record.get("sources"):
        raise ValueError("custom-session manifest does not match registered sources")
    if body.get("tenant_id") != record.get("tenant_id", settings.DEFAULT_TENANT):
        raise ValueError("custom-session manifest tenant mismatch")


def update_custom_session(name: str, *, tenant_id: str | None = None,
                          **updates) -> dict | None:
    """Atomically update one owned record (workflow metadata, never ownership)."""
    tenant_id = tenant_id or settings.DEFAULT_TENANT
    forbidden = {"name", "tenant_id", "owner", "sources"}.intersection(updates)
    if forbidden:
        raise ValueError(f"immutable custom-session fields: {sorted(forbidden)}")
    with _REGISTRY_LOCK, _registry_file_lock():
        records = _load_registry()
        for record in records:
            if record["name"] == name and _owned(record, tenant_id):
                record.update(updates)
                record["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                _save_registry(records)
                return record
    return None


def rename_custom_session(old: str, new: str,
                          tenant_id: str | None = None) -> dict | None:
    """Rename only the tenant-owned registry entry."""
    tenant_id = tenant_id or settings.DEFAULT_TENANT
    with _REGISTRY_LOCK, _registry_file_lock():
        new = validate_session_name(new, exclude=old, tenant_id=tenant_id)
        records = _load_registry()
        for record in records:
            if record["name"] == old and _owned(record, tenant_id):
                record["name"] = new
                record["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                _save_registry(records)
                return record
    return None


def remove_custom_session(name: str, tenant_id: str | None = None) -> bool:
    """Remove only a tenant-owned registry entry (graph data is left as-is)."""
    tenant_id = tenant_id or settings.DEFAULT_TENANT
    with _REGISTRY_LOCK, _registry_file_lock():
        records = _load_registry()
        kept = [r for r in records
                if not (r["name"] == name and _owned(r, tenant_id))]
        if len(kept) == len(records):
            return False
        _save_registry(kept)
        storage = tenant_storage_dir(tenant_id, name)
        if storage.exists():
            shutil.rmtree(storage)
        return True


# ---------------------------------------------------------------------------
# processing (used by scripts/ingest_custom_dataset.py — streams log lines)
# ---------------------------------------------------------------------------

def _stamp_marker(session, name: str, tenant_id: str | None = None,
                  id_patterns: list[dict] | None = None) -> None:
    """Stamp a tenant-owned marker carrying learned retrieval metadata."""
    if tenant_id and settings.TENANT_MODE == "column":
        session.run(
            "MERGE (d:Dataset {name: $name, tenant_id: $tenant}) "
            "ON CREATE SET d.rev = 0 "
            "ON MATCH SET d.rev = coalesce(d.rev, 0) + 1 "
            "SET d.id_patterns_json=$patterns, d.active=true, d.updated_at=datetime()",
            name=name, tenant=tenant_id,
            patterns=json.dumps(id_patterns or [], separators=(",", ":")),
        )
    else:
        session.run(
            "MERGE (d:Dataset {name: $name}) "
            "ON CREATE SET d.rev = 0 "
            "ON MATCH SET d.rev = coalesce(d.rev, 0) + 1 "
            "SET d.id_patterns_json=$patterns, d.active=true, d.updated_at=datetime()",
            name=name, patterns=json.dumps(id_patterns or [], separators=(",", ":")),
        )


def _clear_graph_scope(session, tenant_id: str | None) -> None:
    if tenant_id and settings.TENANT_MODE == "column":
        session.run("MATCH (n) WHERE n.tenant_id=$tenant DETACH DELETE n",
                    tenant=tenant_id)
    else:
        session.run("MATCH (n) DETACH DELETE n")


def build_from_csv(driver, csv_path: Path, session_name: str,
                   reset: bool = False, line_cb=None,
                   tenant_id: str | None = None) -> dict:
    """Load an arbitrary CSV into the graph and stamp the Dataset marker."""
    from scripts.seed_graph import load_nodes, load_relationships

    def log(msg: str) -> None:
        if line_cb:
            line_cb(msg)

    t0 = time.perf_counter()
    nodes, rels = adapt_csv_to_graph(csv_path)
    log(f"  parsed {sum(len(v) for v in nodes.values()):,} nodes / "
        f"{len(rels):,} fraud edges from {csv_path.name}")
    tenant = (tenant_id or settings.DEFAULT_TENANT) \
        if settings.TENANT_MODE == "column" else None
    with driver.session() as session:
        if reset:
            _clear_graph_scope(session, tenant)
            log("  graph scope cleared (--reset)")
        load_nodes(session, nodes, tenant_id=tenant)
        load_relationships(session, rels, tenant_id=tenant)
        _stamp_marker(session, session_name, tenant)
        if tenant:
            counts = session.run(
                "MATCH (n) WHERE n.tenant_id=$tenant "
                "RETURN labels(n)[0] AS label, count(*) AS c ORDER BY label",
                tenant=tenant,
            ).data()
        else:
            counts = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY label"
            ).data()
    log(f"  loaded in {time.perf_counter() - t0:.1f}s — graph nodes: "
        + ", ".join(f"{r['label']}={r['c']}" for r in counts))
    return {"nodes": sum(len(v) for v in nodes.values()),
            "relationships": len(rels), "label_counts": counts}


def build_from_pdfs(driver, pdf_paths: list[Path], session_name: str,
                    reset: bool = False, line_cb=None,
                    tenant_id: str | None = None) -> dict:
    """Extract entities from PDFs and build the graph (same machinery as the
    CDC upload path), then stamp the Dataset marker."""
    from graphrag.entity_extractor import extract_entities
    from graphrag.graph_updater import update_graph_surgically
    from graphrag.pdf_processor import extract_document_from_pdf

    def log(msg: str) -> None:
        if line_cb:
            line_cb(msg)

    total_entities = 0
    tenant = (tenant_id or settings.DEFAULT_TENANT) \
        if settings.TENANT_MODE == "column" else None
    with driver.session() as session:
        if reset:
            _clear_graph_scope(session, tenant)
            log("  graph scope cleared (--reset)")
    replace_vectors = reset
    for pdf in pdf_paths:
        document = extract_document_from_pdf(pdf.read_bytes())
        for warning in document.warnings:
            log(f"  WARN {pdf.name}: {warning}")
        result = extract_entities(document.text, doc_id_hint=pdf.name)
        entities = result["entities"]
        if not entities:
            log(f"  WARN: no entities extracted from {pdf.name}")
            continue
        doc_id = result["doc_id"] or pdf.stem
        changes = {
            "added": [{"label": label, "id": eid, "props": props}
                      for label, ents in entities.items()
                      for eid, props in ents.items()],
            "modified": [],
            "deleted": [],
        }
        stats = update_graph_surgically(
            driver, doc_id, changes, new_entities=entities,
            tenant_id=tenant, dataset_name=session_name,
            replace_vectors=replace_vectors,
        )
        if "embedding_warning" not in stats:
            replace_vectors = False
        n = sum(len(e) for e in entities.values())
        total_entities += n
        log(f"  {pdf.name}: {n} entities ({result['mode']}) in "
            f"{stats['update_time_ms']:.0f}ms")
    with driver.session() as session:
        _stamp_marker(session, session_name, tenant)
        if tenant:
            counts = session.run(
                "MATCH (n) WHERE n.tenant_id=$tenant "
                "RETURN labels(n)[0] AS label, count(*) AS c ORDER BY label",
                tenant=tenant,
            ).data()
        else:
            counts = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY label"
            ).data()
    log("  graph nodes: " + ", ".join(f"{r['label']}={r['c']}" for r in counts))
    return {"entities": total_entities, "label_counts": counts}
