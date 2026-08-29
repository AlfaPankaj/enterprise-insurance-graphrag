"""Surgical Neo4j updates — apply only the CDC changes, never a full rebuild.

Contract (see docs/graph_schema.md §7):

  * entity added     -> MERGE node + set props + derive relationships
  * entity modified  -> MERGE node + set only the provided props
  * entity deleted   -> DETACH DELETE, but only if no OTHER document snapshot
                        still references the entity (reference counting);
                        skipped deletions are reported in ``deleted_skipped``
  * edges            -> derived from join props (policy_id, claim_id, ...)

The whole update runs in a single transaction; ``edges_added`` is measured as
the net change in relationships attached to the touched entity ids.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from graphrag.cache import bump_revision
from graphrag.config import settings
from graphrag.graph_store import SNAPSHOT_LABEL, save_existing_entities
from graphrag.pii import classify, encrypt_value, encryption_enabled

# Join fields are used to derive edges and are not stored as node properties.
JOIN_FIELDS = {"policy_id", "claim_id", "investigator_id", "policyholder_id", "doc_id"}

# Edge types each label derives from its join props (mirrors _derive_edges).
_DERIVED_EDGES: dict[str, list[str]] = {
    "Claim": ["HAS_CLAIM", "INVESTIGATES_CLAIM"],
    "Coverage": ["COVERS"],
    "Endorsement": ["ENDORSED_BY"],
    "FraudFlag": ["FRAUD_DETECTED"],
    "Policy": ["HAS_POLICY"],
}


def _derive_edges(tx, label: str, eid: str, props: dict,
                  tenant_id: str | None = None) -> None:
    """Create relationships without ever matching nodes owned by another tenant."""
    scoped = bool(tenant_id) and settings.TENANT_MODE == "column"

    def run(a_label: str, a_id: str, b_label: str, b_id: str,
            relationship: str) -> None:
        if scoped:
            tx.run(
                f"MERGE (a:{a_label} {{id:$aid, tenant_id:$tenant}}) "
                f"MERGE (b:{b_label} {{id:$bid, tenant_id:$tenant}}) "
                f"MERGE (a)-[:{relationship}]->(b)",
                aid=a_id, bid=b_id, tenant=tenant_id,
            )
        else:
            tx.run(
                f"MERGE (a:{a_label} {{id:$aid}}) MERGE (b:{b_label} {{id:$bid}}) "
                f"MERGE (a)-[:{relationship}]->(b)",
                aid=a_id, bid=b_id,
            )

    if label == "Claim":
        if props.get("policy_id"):
            run("Policy", props["policy_id"], "Claim", eid, "HAS_CLAIM")
        if props.get("investigator_id"):
            run("Investigator", props["investigator_id"], "Claim", eid,
                "INVESTIGATES_CLAIM")
    elif label == "Endorsement" and props.get("policy_id"):
        run("Policy", props["policy_id"], "Endorsement", eid, "ENDORSED_BY")
    elif label == "Coverage" and props.get("policy_id"):
        run("Policy", props["policy_id"], "Coverage", eid, "COVERS")
    elif label == "FraudFlag" and props.get("claim_id"):
        run("Claim", props["claim_id"], "FraudFlag", eid, "FRAUD_DETECTED")
    elif label == "Policy" and props.get("policyholder_id"):
        run("Policyholder", props["policyholder_id"], "Policy", eid, "HAS_POLICY")


def _prune_derived_edges(tx, label: str, eid: str,
                         tenant_id: str | None = None) -> None:
    """Remove the edges this label derives, so re-derivation never leaves stale ones.

    Called for *modified* entities before re-deriving — if a claim is re-assigned
    to a new investigator, the old INVESTIGATES_CLAIM edge must not survive.
    """
    edge_types = _DERIVED_EDGES.get(label)
    if not edge_types:
        return
    if tenant_id and settings.TENANT_MODE == "column":
        tx.run(
            f"MATCH (n:{label} {{id:$id, tenant_id:$tenant}})"
            f"-[r:{'|'.join(edge_types)}]->() DELETE r",
            id=eid, tenant=tenant_id,
        )
    else:
        tx.run(
            f"MATCH (n:{label} {{id: $id}})-[r:{'|'.join(edge_types)}]->() DELETE r",
            id=eid,
        )


def _count_rels(tx, ids: list[str], tenant_id: str | None = None) -> int:
    if tenant_id and settings.TENANT_MODE == "column":
        row = tx.run(
            "UNWIND $ids AS id MATCH (n {id:id, tenant_id:$tenant})-[r]-() "
            "RETURN count(r) AS c", ids=ids, tenant=tenant_id,
        ).single()
    else:
        row = tx.run(
            "UNWIND $ids AS id MATCH (n {id: id})-[r]-() RETURN count(r) AS c",
            ids=ids,
        ).single()
    return row["c"] if row else 0


def _referenced_elsewhere(tx, eid: str, doc_id: str,
                          tenant_id: str | None = None) -> bool:
    """True if another DocSnapshot (not this doc) still contains the entity id.

    Snapshots are JSON strings keyed by entity id, so the entity id is quoted
    in the search string — this makes the CONTAINS match exact
    ("POL-0001" will not match a hypothetical "POL-0001x").
    """
    if tenant_id and settings.TENANT_MODE == "column":
        row = tx.run(
            f"MATCH (n:{SNAPSHOT_LABEL}) WHERE n.tenant_id=$tenant "
            "AND n.doc_id <> $doc_id AND n.entities_json CONTAINS $quoted "
            "RETURN count(n) AS c", doc_id=doc_id, quoted=json.dumps(eid),
            tenant=tenant_id,
        ).single()
    else:
        row = tx.run(
            f"MATCH (n:{SNAPSHOT_LABEL}) "
            "WHERE n.doc_id <> $doc_id AND n.entities_json CONTAINS $quoted "
            "RETURN count(n) AS c", doc_id=doc_id, quoted=json.dumps(eid),
        ).single()
    return bool(row and row["c"] > 0)


def update_graph_surgically(driver, doc_id: str, changes: dict,
                            new_entities: dict | None = None,
                            tenant_id: str | None = None,
                            dataset_name: str | None = None,
                            replace_vectors: bool = False) -> dict:
    """Apply CDC changes to Neo4j. Returns timing + count stats.

    If ``new_entities`` is given, the document snapshot is saved in the SAME
    transaction as the graph update — the graph and the CDC baseline can never
    diverge (a crash mid-flight rolls both back).

    ``tenant_id`` (with ``settings.TENANT_MODE="column"``) stamps new/updated
    nodes with ``tenant_id = coalesce(tenant_id, $tenant)`` — **first owner
    wins**, so a CDC write can never hijack a node that already belongs to
    another tenant.
    """
    stamp_tenant = bool(tenant_id) and settings.TENANT_MODE == "column"
    start = time.perf_counter()
    stats = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "doc_id": doc_id,
        "entities_added": len(changes["added"]),
        "entities_updated": len(changes["modified"]),
        "entities_deleted": len(changes["deleted"]),
        "deleted_skipped": 0,  # entities still referenced by other docs
        "edges_added": 0,
        "embeddings_updated": 0,
        "embeddings_deleted": 0,
        "embedding_warning": None,
        "neo4j_query_time_ms": 0.0,
        "update_time_ms": 0.0,
    }
    upserts = changes["added"] + changes["modified"]
    kept_ids = [e["id"] for e in upserts]

    with driver.session() as session:
        query_start = time.perf_counter()
        with session.begin_transaction() as tx:
            rels_before = _count_rels(tx, kept_ids, tenant_id) if kept_ids else 0
            for entity in changes["deleted"]:
                if _referenced_elsewhere(tx, entity["id"], doc_id, tenant_id):
                    stats["deleted_skipped"] += 1
                    continue
                if stamp_tenant:
                    tx.run(
                        f"MATCH (n:{entity['label']} "
                        "{id:$id, tenant_id:$tenant}) DETACH DELETE n",
                        id=entity["id"], tenant=tenant_id,
                    )
                else:
                    tx.run(
                        f"MATCH (n:{entity['label']} {{id: $id}}) DETACH DELETE n",
                        id=entity["id"],
                    )
            for entity in upserts:
                # NOTE: don't use .get(key, entity["props"]) — the default is
                # evaluated eagerly and raises for modified entities.
                full_props = entity["new_props"] if "new_props" in entity else entity["props"]
                stored = {k: v for k, v in full_props.items() if k not in JOIN_FIELDS}
                # v2 PII at rest: classified fields are Fernet-encrypted before
                # they touch the database (non-PII fields stay plaintext so
                # graph queries/filters keep working)
                if encryption_enabled():
                    stored = {k: (encrypt_value(v) if classify(entity["label"], k) else v)
                              for k, v in stored.items()}
                if stamp_tenant:
                    tx.run(
                        f"MERGE (n:{entity['label']} "
                        "{id:$id, tenant_id:$tenant}) SET n += $props "
                        "SET n.tenant_id = coalesce(n.tenant_id, $tenant)",
                        id=entity["id"], props=stored, tenant=tenant_id,
                    )
                else:
                    tx.run(
                        f"MERGE (n:{entity['label']} {{id: $id}}) SET n += $props",
                        id=entity["id"],
                        props=stored,
                    )
                if "new_props" in entity:
                    _prune_derived_edges(tx, entity["label"], entity["id"], tenant_id)
                _derive_edges(tx, entity["label"], entity["id"], full_props,
                              tenant_id)
            # v2 cache invalidation: any effective write bumps the graph
            # revision so cached answers can never survive a mutation
            effective_deletes = len(changes["deleted"]) - stats["deleted_skipped"]
            if upserts or effective_deletes > 0:
                bump_revision(tx, tenant_id)
            rels_after = _count_rels(tx, kept_ids, tenant_id) if kept_ids else 0
            if new_entities is not None:
                save_existing_entities(tx, doc_id, new_entities,
                                       tenant_id=tenant_id)
        stats["neo4j_query_time_ms"] = round((time.perf_counter() - query_start) * 1000, 2)
        stats["edges_added"] = max(rels_after - rels_before, 0)

    if settings.VECTOR_INCREMENTAL_ENABLED and (upserts or changes["deleted"]):
        try:
            from graphrag.vector_store import (
                delete_vector_embeddings,
                update_native_embeddings,
            )
            if upserts:
                stats["embeddings_updated"] = update_native_embeddings(
                    driver, upserts, tenant_id=tenant_id,
                    dataset_name=dataset_name, replace=replace_vectors,
                )
            if changes["deleted"]:
                stats["embeddings_deleted"] = delete_vector_embeddings(
                    driver, [item["id"] for item in changes["deleted"]],
                    tenant_id=tenant_id, dataset_name=dataset_name,
                )
        except Exception as exc:  # noqa: BLE001 - graph commit already succeeded
            stats["embedding_warning"] = (
                f"incremental embedding update failed: {type(exc).__name__}"
            )

    stats["update_time_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return stats
