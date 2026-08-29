"""Per-document entity snapshots — the CDC baseline state.

Each ingested document stores the exact entity snapshot it produced in a
``(:DocSnapshot {doc_id, entities_json})`` node. ``get_existing_entities`` feeds
``change_detector``; ``save_existing_entities`` records the new baseline after
a successful surgical update.

Snapshots are stored as a JSON *string* because Neo4j node properties cannot
hold nested maps (a {label: {id: props}} structure is three levels deep).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from graphrag.config import settings

SNAPSHOT_LABEL = settings.DOC_SNAPSHOT_LABEL


def get_existing_entities(runner, doc_id: str,
                          tenant_id: str | None = None) -> dict[str, dict[str, dict]]:
    """Return one tenant's stored snapshot (empty dict if none)."""
    scoped = bool(tenant_id) and settings.TENANT_MODE == "column"
    if scoped:
        row = runner.run(
            f"MATCH (n:{SNAPSHOT_LABEL} {{doc_id: $id, tenant_id: $tenant}}) "
            "RETURN n.entities_json AS entities_json",
            id=doc_id, tenant=tenant_id,
        ).single()
    else:
        row = runner.run(
            f"MATCH (n:{SNAPSHOT_LABEL} {{doc_id: $id}}) "
            "RETURN n.entities_json AS entities_json",
            id=doc_id,
        ).single()
    if not row or not row["entities_json"]:
        return {}
    return json.loads(row["entities_json"])


def save_existing_entities(runner, doc_id: str, entities: dict[str, dict[str, dict]],
                           tenant_id: str | None = None) -> None:
    """Upsert a tenant-owned document snapshot (idempotent)."""
    payload = {
        "id": doc_id,
        "entities_json": json.dumps(entities, sort_keys=True),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if tenant_id and settings.TENANT_MODE == "column":
        runner.run(
            f"MERGE (n:{SNAPSHOT_LABEL} {{doc_id: $id, tenant_id: $tenant}}) "
            "SET n.entities_json = $entities_json, n.updated_at = $ts",
            tenant=tenant_id, **payload,
        )
    else:
        runner.run(
            f"MERGE (n:{SNAPSHOT_LABEL} {{doc_id: $id}}) "
            "SET n.entities_json = $entities_json, n.updated_at = $ts",
            **payload,
        )
