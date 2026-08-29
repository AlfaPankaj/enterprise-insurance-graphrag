"""Durable staged workflow for custom-dataset ingestion.

The registry stores the human-review state while ``jobs.py`` supplies durable
execution records. Stages are explicit and auditable:

    persisted -> profiling -> awaiting_mapping_review -> approved -> ingesting
              -> succeeded | failed

Only ``approve_mapping`` can cross the review gate for CSV graph writes.
"""

from __future__ import annotations

import json
import time

from graphrag.custom_sessions import (
    CUSTOM_DIR,
    PROJECT_ROOT,
    get_custom_session,
    update_custom_session,
    verify_session_manifest,
)
from graphrag.schema_induction import (
    induce_schema,
    mapping_fingerprint,
    validate_mapping,
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _ingestion(record: dict, **updates) -> dict:
    current = dict(record.get("ingestion") or {})
    current.update(updates)
    current["updated_at"] = _now()
    return current


def profile_custom_session(name: str, tenant_id: str, *, mode: str | None = None,
                           actor: str = "system") -> dict:
    record = get_custom_session(name, tenant_id)
    if not record:
        raise ValueError(f"unknown custom session {name!r} for this tenant")
    verify_session_manifest(record)
    if record["kind"] != "csv":
        raise ValueError("schema induction is only required for CSV sessions")
    paths = [PROJECT_ROOT / source for source in record["sources"]]
    from graphrag.config import settings
    from upload import UploadLimits, revalidate_persisted_uploads, scanner_from_settings
    revalidate_persisted_uploads(
        paths, allowed_root=CUSTOM_DIR,
        recorded_metadata=record.get("uploads") or [],
        limits=UploadLimits.from_settings(settings),
        malware_scanner=scanner_from_settings(settings),
    )
    started = time.perf_counter()
    update_custom_session(
        name, tenant_id=tenant_id,
        ingestion=_ingestion(record, status="running", stage="profiling",
                             started_at=_now()),
    )
    try:
        proposal = induce_schema(paths, mode=mode)
        from graphrag.entity_resolution import find_csv_candidates
        er_candidates = []
        mapping_er = proposal["mapping"].get("entity_resolution") or {}
        for path in paths:
            for candidate in find_csv_candidates(
                path, proposal["mapping"]["files"][path.name],
                threshold=float(mapping_er.get("fuzzy_threshold", 0.92)),
            ):
                candidate["file"] = path.name
                er_candidates.append(candidate)
    except Exception as exc:
        update_custom_session(
            name, tenant_id=tenant_id,
            ingestion=_ingestion(record, status="failed", stage="profiling",
                                 finished_at=_now(), warnings=[str(exc)]),
        )
        raise
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    metadata = proposal.get("metadata") or {}
    ingestion = _ingestion(
        record,
        status="awaiting_review",
        stage="awaiting_mapping_review",
        finished_at=_now(),
        warnings=list(metadata.get("warnings") or []),
        cost_usd=float(metadata.get("cost_usd") or 0.0),
        timings_ms={"profiling": elapsed},
    )
    updated = update_custom_session(
        name, tenant_id=tenant_id,
        profiles=proposal["profiles"],
        schema_mapping=proposal["mapping"],
        entity_resolution_candidates=er_candidates,
        mapping_reviews=[{
            "action": "proposed", "actor": actor, "at": _now(),
            "fingerprint": proposal["mapping"]["fingerprint"],
        }],
        ingestion=ingestion,
    )
    return {
        "session": name,
        "status": "awaiting_review",
        "mapping_fingerprint": proposal["mapping"]["fingerprint"],
        "warnings": ingestion["warnings"],
        "cost_usd": ingestion["cost_usd"],
        "timings_ms": ingestion["timings_ms"],
        "record": updated,
    }


def approve_mapping(name: str, tenant_id: str, mapping: dict, *,
                    actor: str) -> dict:
    """Validate and persist a reviewed mapping, opening the graph-write gate."""
    record = get_custom_session(name, tenant_id)
    if not record:
        raise ValueError(f"unknown custom session {name!r} for this tenant")
    profiles = record.get("profiles")
    if not profiles:
        raise ValueError("dataset has not been profiled")
    validated = validate_mapping(mapping, profiles)
    validated["status"] = "approved"
    validated["fingerprint"] = mapping_fingerprint(validated)
    reviews = list(record.get("mapping_reviews") or [])
    reviews.append({
        "action": "approved", "actor": actor, "at": _now(),
        "fingerprint": validated["fingerprint"],
    })
    updated = update_custom_session(
        name, tenant_id=tenant_id,
        schema_mapping=validated,
        mapping_reviews=reviews,
        ingestion=_ingestion(record, status="approved", stage="approved"),
    )
    return updated or {}


def reject_mapping(name: str, tenant_id: str, *, actor: str,
                   reason: str = "") -> dict:
    record = get_custom_session(name, tenant_id)
    if not record:
        raise ValueError(f"unknown custom session {name!r} for this tenant")
    reviews = list(record.get("mapping_reviews") or [])
    reviews.append({"action": "rejected", "actor": actor, "at": _now(),
                    "reason": reason[:500]})
    return update_custom_session(
        name, tenant_id=tenant_id, mapping_reviews=reviews,
        ingestion=_ingestion(record, status="rejected", stage="mapping_review",
                             warnings=[reason] if reason else []),
    ) or {}


def mark_ingestion_result(name: str, tenant_id: str, result: dict | None,
                          error: str | None = None) -> dict:
    record = get_custom_session(name, tenant_id)
    if not record:
        raise ValueError(f"unknown custom session {name!r} for this tenant")
    previous = record.get("ingestion") or {}
    timings = dict(previous.get("timings_ms") or {})
    warnings = list(previous.get("warnings") or [])
    if result:
        timings.update(result.get("timings_ms") or {})
        warnings.extend(result.get("warnings") or [])
    if error:
        warnings.append(error[:1000])
    return update_custom_session(
        name, tenant_id=tenant_id,
        ingestion=_ingestion(
            record,
            status="failed" if error else "succeeded",
            stage="failed" if error else "complete",
            finished_at=_now(), warnings=warnings, timings_ms=timings,
            cost_usd=float(previous.get("cost_usd") or 0.0)
            + float((result or {}).get("cost_usd") or 0.0),
            report_path=(result or {}).get("report_path"),
        ),
    ) or {}


def register_ingestion_handlers(driver_getter) -> None:
    """Register profile/ingest job kinds (idempotent)."""
    from graphrag.jobs import JobError, register_handler
    from graphrag.relational_ingest import ingest_csv_bundle

    def profile_handler(job, params, progress, cancelled):
        name = params.get("session_id")
        tenant = params.get("tenant_id")
        if not name or not tenant:
            raise JobError("profile_upload requires session_id and tenant_id")
        progress("profiling CSV bundle")
        result = profile_custom_session(
            name, tenant, mode=params.get("mode"), actor=params.get("actor", "system")
        )
        progress("mapping proposal ready for human review")
        # Avoid duplicating the full profiles/mapping in jobs.db; registry is the source of truth.
        result.pop("record", None)
        return result

    def ingest_handler(job, params, progress, cancelled):
        name = params.get("session_id")
        tenant = params.get("tenant_id")
        record = get_custom_session(name, tenant)
        if not record:
            raise JobError("unknown tenant-owned custom session")
        verify_session_manifest(record)
        mapping = record.get("schema_mapping") or {}
        if mapping.get("status") != "approved":
            raise JobError("schema mapping requires human approval before ingest")
        update_custom_session(
            name, tenant_id=tenant,
            ingestion=_ingestion(record, status="running", stage="ingesting",
                                 started_at=_now()),
        )
        paths = [PROJECT_ROOT / source for source in record["sources"]]
        from graphrag.config import settings
        from upload import (
            UploadLimits,
            revalidate_persisted_uploads,
            scanner_from_settings,
        )
        revalidate_persisted_uploads(
            paths, allowed_root=CUSTOM_DIR,
            recorded_metadata=record.get("uploads") or [],
            limits=UploadLimits.from_settings(settings),
            malware_scanner=scanner_from_settings(settings),
        )
        try:
            result = ingest_csv_bundle(
                driver_getter(), paths, mapping, name, tenant_id=tenant,
                reset=bool(params.get("reset", True)), line_cb=progress,
            )
            mark_ingestion_result(name, tenant, result)
            return result
        except Exception as exc:
            mark_ingestion_result(name, tenant, None, str(exc))
            raise

    register_handler("profile_upload", profile_handler)
    register_handler("ingest_custom", ingest_handler)


def export_workflow_record(name: str, tenant_id: str) -> str:
    """Stable JSON representation useful for audit exports and tests."""
    record = get_custom_session(name, tenant_id)
    if not record:
        raise ValueError("unknown session")
    return json.dumps({
        "name": record["name"],
        "tenant_id": record.get("tenant_id"),
        "owner": record.get("owner"),
        "ingestion": record.get("ingestion"),
        "mapping_reviews": record.get("mapping_reviews", []),
        "mapping_fingerprint": (record.get("schema_mapping") or {}).get("fingerprint"),
    }, sort_keys=True)
