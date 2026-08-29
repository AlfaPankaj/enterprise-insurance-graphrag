"""Bounded-memory relational CSV-to-graph ingestion.

Rows are streamed in configurable batches, duplicate/exact-resolution indexes
live in temporary SQLite, and foreign-key edges are created only after all
nodes exist.  The approved mapping is the sole authority for interpolated
labels/property names and is validated again immediately before graph writes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

from graphrag.config import settings
from graphrag.entity_resolution import normalize_value
from graphrag.schema_induction import (
    mapping_fingerprint,
    safe_identifier,
    validate_mapping,
)


def _batches(items: Iterator[dict], size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _native(value: str, kind: str):
    value = str(value or "").strip()
    if not value:
        return None
    compact = value.replace(",", "").replace("$", "").replace("₹", "")
    try:
        if kind == "integer":
            return int(compact)
        if kind == "number":
            return float(compact)
        if kind == "boolean":
            return value.casefold() in {"1", "true", "yes", "y"}
    except ValueError:
        return value
    return value


def _profiles_from_paths(paths: list[Path]) -> list[dict]:
    profiles = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            headers = list(csv.DictReader(handle).fieldnames or [])
        profiles.append({
            "filename": path.name,
            "columns": [{"name": h} for h in headers],
        })
    return profiles


def _checkpoint_path(paths: list[Path]) -> Path:
    return paths[0].parent / ".ingestion_checkpoint.json"


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        Path(tmp).unlink(missing_ok=True)
        raise


def _load_checkpoint(path: Path, fingerprint: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("mapping_fingerprint") == fingerprint:
            return value
    except (OSError, ValueError):
        pass
    return {
        "mapping_fingerprint": fingerprint,
        "reset_complete": False,
        "node_files_complete": [],
        "relationship_indexes_complete": [],
        "status": "running",
    }


def _row_id(filename: str, row_number: int) -> str:
    prefix = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:8].upper()
    return f"ROW-{prefix}-{row_number:09d}"


class _DiskIndex:
    """Disk-backed uniqueness, alias, and FK membership indexes."""

    def __init__(self):
        with tempfile.NamedTemporaryFile(
            prefix="graphrag-ingest-", suffix=".db", delete=False
        ) as tmp:
            self.path = Path(tmp.name)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=OFF")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.executescript(
            "CREATE TABLE ids (file TEXT, id TEXT, PRIMARY KEY(file,id));"
            "CREATE TABLE exact_keys (label TEXT, field TEXT, value_hash TEXT, "
            " canonical_id TEXT, PRIMARY KEY(label,field,value_hash));"
        )

    def close(self) -> None:
        self.conn.close()
        self.path.unlink(missing_ok=True)

    def add_id(self, filename: str, source_id: str) -> bool:
        try:
            self.conn.execute("INSERT INTO ids(file,id) VALUES (?,?)", (filename, source_id))
            return True
        except sqlite3.IntegrityError:
            return False

    def contains_id(self, filename: str, source_id: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM ids WHERE file=? AND id=?", (filename, source_id)
        ).fetchone() is not None

    def canonical_for(self, label: str, fields: list[tuple[str, str]], source_id: str) -> tuple[str, str | None]:
        for field, value in fields:
            normalized = normalize_value(field, value)
            if not normalized:
                continue
            hashed = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            found = self.conn.execute(
                "SELECT canonical_id FROM exact_keys WHERE label=? AND field=? AND value_hash=?",
                (label, field, hashed),
            ).fetchone()
            if found:
                return str(found[0]), field
        for field, value in fields:
            normalized = normalize_value(field, value)
            if not normalized:
                continue
            hashed = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            self.conn.execute(
                "INSERT OR IGNORE INTO exact_keys(label,field,value_hash,canonical_id) "
                "VALUES (?,?,?,?)", (label, field, hashed, source_id),
            )
        return source_id, None


def _stream_nodes(
    path: Path,
    item: dict,
    index: _DiskIndex,
    report: dict,
) -> Iterator[dict]:
    label = item["label"]
    id_col = item["id_column"]
    props_map = item["properties"]
    types = item.get("types") or {}
    resolution = report["mapping"]["entity_resolution"]
    exact_keys = set(resolution.get("exact_keys") or [])
    auto_exact = resolution.get("mode") in {"review", "auto_exact"}
    accepted = {
        tuple(sorted(map(str, pair)))
        for pair in resolution.get("accepted_merges", []) if len(pair) == 2
    }
    rejected = {
        tuple(sorted(map(str, pair)))
        for pair in resolution.get("rejected_merges", []) if len(pair) == 2
    }

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=1):
            source_id = (str(row.get(id_col) or "").strip()
                         if id_col != "__generated_id__" else "")
            source_id = source_id or _row_id(path.name, row_number)
            if not index.add_id(path.name, source_id):
                report["duplicate_ids"][path.name] += 1
            props = {
                target: _native(row.get(source), types.get(source, "string"))
                for source, target in props_map.items()
            }
            props = {key: value for key, value in props.items() if value is not None}
            fields = [(target, props.get(target)) for target in props
                      if any(key == target or key in target for key in exact_keys)]
            canonical, exact_field = index.canonical_for(label, fields, source_id)
            pair = tuple(sorted((source_id, canonical)))
            if canonical != source_id and (pair in rejected or (not auto_exact and pair not in accepted)):
                canonical, exact_field = source_id, None
            # Human-approved fuzzy pairs are authoritative and reversible via
            # source_ids/provenance, even when no exact key matches.
            reviewed_pair = next((pair for pair in accepted if source_id in pair), None)
            if reviewed_pair and reviewed_pair not in rejected:
                canonical = min(reviewed_pair)
                exact_field = "reviewed_fuzzy"
            if canonical != source_id:
                report["resolution_decision_count"] += 1
                if len(report["resolution_decisions"]) < 1000:
                    report["resolution_decisions"].append({
                        "left_id": canonical,
                        "right_id": source_id,
                        "canonical_id": canonical,
                        "score": 1.0,
                        "method": f"exact:{exact_field}",
                        "status": "accepted",
                    })
            yield {
                "id": canonical,
                "source_id": source_id,
                "row_number": row_number,
                "props": props,
            }


def _node_query(label: str, tenant: bool) -> str:
    label = safe_identifier(label, kind="node label")
    if tenant:
        return (
            f"UNWIND $rows AS r MERGE (n:{label} {{tenant_id:$tenant, id:r.id}}) "
            "SET n += r.props, n.dataset_name=$dataset "
            "SET n.source_ids = CASE WHEN r.source_id IN coalesce(n.source_ids, []) "
            "THEN n.source_ids ELSE coalesce(n.source_ids, []) + r.source_id END"
        )
    return (
        f"UNWIND $rows AS r MERGE (n:{label} {{id:r.id}}) "
        "SET n += r.props, n.dataset_name=$dataset "
        "SET n.source_ids = CASE WHEN r.source_id IN coalesce(n.source_ids, []) "
        "THEN n.source_ids ELSE coalesce(n.source_ids, []) + r.source_id END"
    )


def _stream_relationship_rows(path: Path, relationship: dict,
                              index: _DiskIndex, report: dict) -> Iterator[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=1):
            from_map = report["mapping"]["files"][relationship["from_file"]]
            source_id = (str(row.get(from_map["id_column"]) or "").strip()
                         if from_map["id_column"] != "__generated_id__" else "")
            source_id = source_id or _row_id(path.name, row_number)
            target_id = str(row.get(relationship["from_column"]) or "").strip()
            if not target_id:
                continue
            # Exact entity resolution may have changed the source canonical ID.
            # The node's source_ids list makes this source-side match reversible.
            if not index.contains_id(relationship["to_file"], target_id):
                report["orphan_count"] += 1
                if len(report["orphans"]) < 1000:
                    report["orphans"].append({
                        "relationship": relationship["type"],
                        "source_file": path.name,
                        "row": row_number,
                        "missing_target_id": target_id,
                    })
            yield {"source": source_id, "target": target_id}


def _relationship_query(rel: dict, mapping: dict, tenant: bool) -> str:
    from_label = safe_identifier(mapping["files"][rel["from_file"]]["label"], kind="node label")
    to_label = safe_identifier(mapping["files"][rel["to_file"]]["label"], kind="node label")
    rtype = safe_identifier(rel["type"], kind="relationship type")
    tenant_match = " {tenant_id:$tenant}" if tenant else ""
    # source IDs may be aliases; match either the canonical id or source_ids.
    query = (
        f"UNWIND $rows AS r MATCH (a:{from_label}{tenant_match}) "
        "WHERE a.id=r.source OR r.source IN coalesce(a.source_ids, []) "
        f"MATCH (b:{to_label}{tenant_match}) "
        "WHERE b.id=r.target OR r.target IN coalesce(b.source_ids, []) "
    )
    if rel["direction"] == "to_from":
        query += f"MERGE (b)-[edge:{rtype}]->(a) RETURN count(edge) AS created"
    else:
        query += f"MERGE (a)-[edge:{rtype}]->(b) RETURN count(edge) AS created"
    return query


def _ensure_indexes(session, mapping: dict, tenant: bool) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    fulltext_indexes: list[str] = []
    for item in mapping["files"].values():
        safe = safe_identifier(item["label"], kind="node label")
        suffix = hashlib.sha256(safe.encode()).hexdigest()[:8]
        try:
            if tenant:
                session.run(
                    f"CREATE CONSTRAINT custom_{suffix}_tenant_id IF NOT EXISTS "
                    f"FOR (n:{safe}) REQUIRE (n.tenant_id, n.id) IS UNIQUE"
                ).consume()
            else:
                session.run(
                    f"CREATE CONSTRAINT custom_{suffix}_id IF NOT EXISTS "
                    f"FOR (n:{safe}) REQUIRE n.id IS UNIQUE"
                ).consume()
        except Exception as exc:  # noqa: BLE001 - backend/version compatibility
            warnings.append(f"Could not create ID constraint for {safe}: {type(exc).__name__}")
        properties = ["id", *dict.fromkeys(item["properties"].values())]
        if settings.TENANT_MODE == "column" and "tenant_id" not in properties:
            properties.append("tenant_id")
        # Keep index definitions manageable on very wide real-world tables.
        properties = [safe_identifier(prop, kind="property name")
                      for prop in properties[:64]]
        index_name = f"custom_fulltext_{suffix}"
        try:
            prop_expr = ", ".join(f"n.{prop}" for prop in properties)
            session.run(
                f"CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS "
                f"FOR (n:{safe}) ON EACH [{prop_expr}]"
            ).consume()
            fulltext_indexes.append(index_name)
        except Exception as exc:  # noqa: BLE001 - backend/version compatibility
            warnings.append(f"Could not create full-text index for {safe}: {type(exc).__name__}")
    return warnings, fulltext_indexes


def ingest_csv_bundle(
    driver,
    paths: list[Path],
    mapping: dict,
    session_name: str,
    *,
    tenant_id: str | None = None,
    reset: bool = False,
    batch_size: int | None = None,
    line_cb: Callable[[str], None] | None = None,
    resume: bool = True,
) -> dict:
    """Stream an approved CSV bundle into Neo4j and return a quality report."""
    if not paths:
        raise ValueError("at least one CSV path is required")
    paths = [Path(p) for p in paths]
    profiles = _profiles_from_paths(paths)
    mapping = validate_mapping(mapping, profiles)
    if mapping.get("status") != "approved":
        raise ValueError("schema mapping must be human-approved before graph writes")
    by_name = {path.name: path for path in paths}
    if set(by_name) != set(mapping["files"]):
        raise ValueError("mapping/files do not match the persisted upload manifest")

    size = max(1, int(batch_size or settings.BATCH_SIZE))
    tenant = bool(tenant_id) and settings.TENANT_MODE == "column"
    fingerprint = mapping_fingerprint(mapping)
    checkpoint_path = _checkpoint_path(paths)
    checkpoint = (_load_checkpoint(checkpoint_path, fingerprint) if resume else
                  _load_checkpoint(Path("/nonexistent"), fingerprint))
    report = {
        "session": session_name,
        "tenant_id": tenant_id,
        "mapping_fingerprint": fingerprint,
        "mapping": mapping,
        "rows": Counter(),
        "nodes_written": Counter(),
        "relationships_written": Counter(),
        "duplicate_ids": Counter(),
        "orphan_count": 0,
        "orphans": [],
        "resolution_decision_count": 0,
        "resolution_decisions": [],
        "warnings": [],
        "timings_ms": {},
        "cost_usd": 0.0,
    }

    def log(message: str) -> None:
        if line_cb:
            line_cb(message)

    started = time.perf_counter()
    index = _DiskIndex()
    try:
        with driver.session() as session:
            should_reset = reset and (
                not checkpoint.get("reset_complete")
                or checkpoint.get("status") == "succeeded"
                or not resume
            )
            if should_reset:
                if tenant:
                    session.run(
                        "MATCH (n) WHERE n.tenant_id=$tenant DETACH DELETE n",
                        tenant=tenant_id,
                    ).consume()
                else:
                    session.run("MATCH (n) DETACH DELETE n").consume()
                checkpoint.update({
                    "reset_complete": True,
                    "node_files_complete": [],
                    "relationship_indexes_complete": [],
                    "status": "running",
                })
                _write_json_atomic(checkpoint_path, checkpoint)
                log("  graph scope cleared (--reset)")
            index_warnings, fulltext_indexes = _ensure_indexes(
                session, mapping, tenant
            )
            report["warnings"].extend(index_warnings)
            replace_vectors = should_reset

        node_started = time.perf_counter()
        for filename, item in mapping["files"].items():
            # Re-running a file is safe (MERGE), and is necessary to reconstruct
            # the disk FK index after process restart. Checkpoints are therefore
            # audit markers rather than unsafe skip instructions.
            path = by_name[filename]
            query = _node_query(item["label"], tenant)
            with driver.session() as session:
                for rows in _batches(_stream_nodes(path, item, index, report), size):
                    kwargs = {"rows": rows, "dataset": session_name}
                    if tenant:
                        kwargs["tenant"] = tenant_id
                    session.run(query, **kwargs).consume()
                    report["rows"][filename] += len(rows)
                    report["nodes_written"][item["label"]] += len(rows)
                    if (settings.VECTOR_BACKEND or "").strip().lower() in {
                        "neo4j", "pgvector",
                    } and settings.VECTOR_INCREMENTAL_ENABLED:
                        try:
                            from graphrag.vector_store import update_native_embeddings
                            update_native_embeddings(
                                driver,
                                [{"id": row["id"], "label": item["label"],
                                  "props": row["props"]} for row in rows],
                                tenant_id=tenant_id,
                                dataset_name=session_name,
                                replace=replace_vectors,
                            )
                            replace_vectors = False
                        except Exception as exc:  # noqa: BLE001 - graph already committed
                            warning = ("incremental embedding update failed for "
                                       f"{filename}: {type(exc).__name__}")
                            if warning not in report["warnings"]:
                                report["warnings"].append(warning)
            if filename not in checkpoint["node_files_complete"]:
                checkpoint["node_files_complete"].append(filename)
            _write_json_atomic(checkpoint_path, checkpoint)
            log(f"  {filename}: streamed {report['rows'][filename]:,} rows")
        index.conn.commit()
        report["timings_ms"]["nodes"] = round((time.perf_counter() - node_started) * 1000, 2)

        rel_started = time.perf_counter()
        for number, rel in enumerate(mapping["relationships"]):
            path = by_name[rel["from_file"]]
            query = _relationship_query(rel, mapping, tenant)
            with driver.session() as session:
                for rows in _batches(_stream_relationship_rows(path, rel, index, report), size):
                    kwargs = {"rows": rows}
                    if tenant:
                        kwargs["tenant"] = tenant_id
                    result = session.run(query, **kwargs).single()
                    report["relationships_written"][rel["type"]] += (
                        int(result["created"]) if result and result.get("created") is not None else 0
                    )
            if number not in checkpoint["relationship_indexes_complete"]:
                checkpoint["relationship_indexes_complete"].append(number)
            _write_json_atomic(checkpoint_path, checkpoint)
        report["timings_ms"]["relationships"] = round(
            (time.perf_counter() - rel_started) * 1000, 2
        )

        patterns_json = json.dumps(mapping.get("id_patterns") or [], separators=(",", ":"))
        with driver.session() as session:
            if tenant:
                session.run(
                    "MERGE (d:Dataset {tenant_id:$tenant, name:$name}) "
                    "ON CREATE SET d.rev=0 ON MATCH SET d.rev=coalesce(d.rev,0)+1 "
                    "SET d.id_patterns_json=$patterns, d.fulltext_indexes_json=$fulltext, "
                    "d.mapping_fingerprint=$fingerprint, d.active=true, d.updated_at=datetime()",
                    tenant=tenant_id, name=session_name, patterns=patterns_json,
                    fulltext=json.dumps(fulltext_indexes), fingerprint=fingerprint,
                ).consume()
            else:
                session.run(
                    "MERGE (d:Dataset {name:$name}) "
                    "ON CREATE SET d.rev=0 ON MATCH SET d.rev=coalesce(d.rev,0)+1 "
                    "SET d.id_patterns_json=$patterns, d.fulltext_indexes_json=$fulltext, "
                    "d.mapping_fingerprint=$fingerprint, d.active=true, d.updated_at=datetime()",
                    name=session_name, patterns=patterns_json,
                    fulltext=json.dumps(fulltext_indexes), fingerprint=fingerprint,
                ).consume()
        checkpoint["status"] = "succeeded"
        _write_json_atomic(checkpoint_path, checkpoint)
    finally:
        index.close()

    report["timings_ms"]["total"] = round((time.perf_counter() - started) * 1000, 2)
    for key in ("rows", "nodes_written", "relationships_written", "duplicate_ids"):
        report[key] = dict(report[key])
    report.pop("mapping", None)
    report_path = paths[0].parent / "ingestion_report.json"
    _write_json_atomic(report_path, report)
    report["report_path"] = str(report_path)
    return report
