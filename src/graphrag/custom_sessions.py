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

  * **CSV**  — a generic adapter: claim-like columns (claim/fraud/amount/…)
    build a ``(:Claim)``/``(:FraudFlag)`` graph, anything else becomes generic
    ``(:Record)`` nodes with every column as a property.
  * **PDF**  — the standard extraction pipeline (pdf_processor →
    entity_extractor → graph_updater) with entity-derived edges.

Both stamp a ``(:Dataset {name})`` marker so session detection, the dashboard
and the audit UI work unchanged. The pure logic here (validation, the CSV
adapter) is unit-testable without Neo4j.
"""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

from graphrag.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "data" / "custom_sessions.json"
CUSTOM_DIR = PROJECT_ROOT / "data" / "custom"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_ -]{0,47}$")


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
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def list_custom_sessions() -> list[dict]:
    """All registered custom sessions (name, kind, sources, created_at, note)."""
    return _load_registry()


def get_custom_session(name: str) -> dict | None:
    return next((r for r in _load_registry() if r["name"] == name), None)


def validate_session_name(name: str, exclude: str | None = None) -> str:
    """Validate + normalize a custom session name; raises ValueError.

    Rules: 1-48 chars of letters/digits/space/underscore/hyphen; must not
    collide with the built-in sessions or another custom session (``exclude``
    allows renames to keep their current name).
    """
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
    if cleaned != exclude and any(r["name"] == cleaned for r in _load_registry()):
        raise ValueError(f"A custom session named '{cleaned}' already exists.")
    return cleaned


def add_custom_session(name: str, kind: str, sources: list[str],
                       note: str = "") -> dict:
    """Register a new custom session (validates name + kind + sources exist)."""
    name = validate_session_name(name)
    if kind not in ("csv", "pdf"):
        raise ValueError("kind must be 'csv' or 'pdf'")
    if not sources:
        raise ValueError("at least one source file is required")
    for src in sources:
        if not (PROJECT_ROOT / src).exists():
            raise ValueError(f"source file missing: {src}")
    record = {
        "name": name,
        "kind": kind,
        "sources": list(sources),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note,
    }
    records = _load_registry()
    records.append(record)
    _save_registry(records)
    return record


def rename_custom_session(old: str, new: str) -> dict | None:
    """Rename a custom session in the registry (the caller re-stamps the graph
    marker by switching sessions). Returns the updated record or None."""
    new = validate_session_name(new, exclude=old)
    records = _load_registry()
    for r in records:
        if r["name"] == old:
            r["name"] = new
            _save_registry(records)
            return r
    return None


def remove_custom_session(name: str) -> bool:
    """Remove a custom session from the registry (graph is left as-is)."""
    records = _load_registry()
    kept = [r for r in records if r["name"] != name]
    if len(kept) == len(records):
        return False
    _save_registry(kept)
    return True


# ---------------------------------------------------------------------------
# processing (used by scripts/ingest_custom_dataset.py — streams log lines)
# ---------------------------------------------------------------------------

def _stamp_marker(session, name: str) -> None:
    session.run("MERGE (d:Dataset {name: $name})", name=name)


def build_from_csv(driver, csv_path: Path, session_name: str,
                   reset: bool = False, line_cb=None) -> dict:
    """Load an arbitrary CSV into the graph and stamp the Dataset marker."""
    from scripts.seed_graph import load_nodes, load_relationships

    def log(msg: str) -> None:
        if line_cb:
            line_cb(msg)

    t0 = time.perf_counter()
    nodes, rels = adapt_csv_to_graph(csv_path)
    log(f"  parsed {sum(len(v) for v in nodes.values()):,} nodes / "
        f"{len(rels):,} fraud edges from {csv_path.name}")
    with driver.session() as session:
        if reset:
            session.run("MATCH (n) DETACH DELETE n")
            log("  graph cleared (--reset)")
        load_nodes(session, nodes)
        load_relationships(session, rels)
        _stamp_marker(session, session_name)
        counts = session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY label"
        ).data()
    log(f"  loaded in {time.perf_counter() - t0:.1f}s — graph nodes: "
        + ", ".join(f"{r['label']}={r['c']}" for r in counts))
    return {"nodes": sum(len(v) for v in nodes.values()),
            "relationships": len(rels), "label_counts": counts}


def build_from_pdfs(driver, pdf_paths: list[Path], session_name: str,
                    reset: bool = False, line_cb=None) -> dict:
    """Extract entities from PDFs and build the graph (same machinery as the
    CDC upload path), then stamp the Dataset marker."""
    from graphrag.entity_extractor import extract_entities
    from graphrag.graph_updater import update_graph_surgically
    from graphrag.pdf_processor import extract_text_from_pdf

    def log(msg: str) -> None:
        if line_cb:
            line_cb(msg)

    total_entities = 0
    with driver.session() as session:
        if reset:
            session.run("MATCH (n) DETACH DELETE n")
            log("  graph cleared (--reset)")
    for pdf in pdf_paths:
        t0 = time.perf_counter()
        text = extract_text_from_pdf(pdf.read_bytes())
        result = extract_entities(text, doc_id_hint=pdf.name)
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
        stats = update_graph_surgically(driver, doc_id, changes,
                                        new_entities=entities)
        n = sum(len(e) for e in entities.values())
        total_entities += n
        log(f"  {pdf.name}: {n} entities ({result['mode']}) in "
            f"{stats['update_time_ms']:.0f}ms")
    with driver.session() as session:
        _stamp_marker(session, session_name)
        counts = session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY label"
        ).data()
    log("  graph nodes: " + ", ".join(f"{r['label']}={r['c']}" for r in counts))
    return {"entities": total_entities, "label_counts": counts}
