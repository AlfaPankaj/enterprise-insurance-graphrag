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

from graphrag.graph_store import SNAPSHOT_LABEL, save_existing_entities

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


def _derive_edges(tx, label: str, eid: str, props: dict) -> None:
    """Create the relationships implied by an entity's join props."""
    if label == "Claim":
        if props.get("policy_id"):
            tx.run(
                "MERGE (p:Policy {id: $pid}) MERGE (c:Claim {id: $cid}) "
                "MERGE (p)-[:HAS_CLAIM]->(c)",
                pid=props["policy_id"], cid=eid,
            )
        if props.get("investigator_id"):
            tx.run(
                "MERGE (i:Investigator {id: $iid}) MERGE (c:Claim {id: $cid}) "
                "MERGE (i)-[:INVESTIGATES_CLAIM]->(c)",
                iid=props["investigator_id"], cid=eid,
            )
    elif label == "Endorsement" and props.get("policy_id"):
        tx.run(
            "MERGE (p:Policy {id: $pid}) MERGE (e:Endorsement {id: $eid}) "
            "MERGE (p)-[:ENDORSED_BY]->(e)",
            pid=props["policy_id"], eid=eid,
        )
    elif label == "Coverage" and props.get("policy_id"):
        tx.run(
            "MERGE (p:Policy {id: $pid}) MERGE (c:Coverage {id: $cid}) "
            "MERGE (p)-[:COVERS]->(c)",
            pid=props["policy_id"], cid=eid,
        )
    elif label == "FraudFlag" and props.get("claim_id"):
        tx.run(
            "MERGE (c:Claim {id: $cid}) MERGE (f:FraudFlag {id: $fid}) "
            "MERGE (c)-[:FRAUD_DETECTED]->(f)",
            cid=props["claim_id"], fid=eid,
        )
    elif label == "Policy" and props.get("policyholder_id"):
        tx.run(
            "MERGE (ph:Policyholder {id: $phid}) MERGE (p:Policy {id: $pid}) "
            "MERGE (ph)-[:HAS_POLICY]->(p)",
            phid=props["policyholder_id"], pid=eid,
        )


def _prune_derived_edges(tx, label: str, eid: str) -> None:
    """Remove the edges this label derives, so re-derivation never leaves stale ones.

    Called for *modified* entities before re-deriving — if a claim is re-assigned
    to a new investigator, the old INVESTIGATES_CLAIM edge must not survive.
    """
    edge_types = _DERIVED_EDGES.get(label)
    if not edge_types:
        return
    tx.run(
        f"MATCH (n:{label} {{id: $id}})-[r:{'|'.join(edge_types)}]->() DELETE r",
        id=eid,
    )


def _count_rels(tx, ids: list[str]) -> int:
    row = tx.run(
        "UNWIND $ids AS id MATCH (n {id: id})-[r]-() RETURN count(r) AS c",
        ids=ids,
    ).single()
    return row["c"] if row else 0


def _referenced_elsewhere(tx, eid: str, doc_id: str) -> bool:
    """True if another DocSnapshot (not this doc) still contains the entity id.

    Snapshots are JSON strings keyed by entity id, so the entity id is quoted
    in the search string — this makes the CONTAINS match exact
    ("POL-0001" will not match a hypothetical "POL-0001x").
    """
    row = tx.run(
        f"MATCH (n:{SNAPSHOT_LABEL}) "
        "WHERE n.doc_id <> $doc_id AND n.entities_json CONTAINS $quoted "
        "RETURN count(n) AS c",
        doc_id=doc_id,
        quoted=json.dumps(eid),
    ).single()
    return bool(row and row["c"] > 0)


def update_graph_surgically(driver, doc_id: str, changes: dict,
                            new_entities: dict | None = None) -> dict:
    """Apply CDC changes to Neo4j. Returns timing + count stats.

    If ``new_entities`` is given, the document snapshot is saved in the SAME
    transaction as the graph update — the graph and the CDC baseline can never
    diverge (a crash mid-flight rolls both back).
    """
    start = time.perf_counter()
    stats = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "doc_id": doc_id,
        "entities_added": len(changes["added"]),
        "entities_updated": len(changes["modified"]),
        "entities_deleted": len(changes["deleted"]),
        "deleted_skipped": 0,  # entities still referenced by other docs
        "edges_added": 0,
        "neo4j_query_time_ms": 0.0,
        "update_time_ms": 0.0,
    }
    upserts = changes["added"] + changes["modified"]
    kept_ids = [e["id"] for e in upserts]

    with driver.session() as session:
        query_start = time.perf_counter()
        with session.begin_transaction() as tx:
            rels_before = _count_rels(tx, kept_ids) if kept_ids else 0
            for entity in changes["deleted"]:
                if _referenced_elsewhere(tx, entity["id"], doc_id):
                    stats["deleted_skipped"] += 1
                    continue
                tx.run(
                    f"MATCH (n:{entity['label']} {{id: $id}}) DETACH DELETE n",
                    id=entity["id"],
                )
            for entity in upserts:
                # NOTE: don't use .get(key, entity["props"]) — the default is
                # evaluated eagerly and raises for modified entities.
                full_props = entity["new_props"] if "new_props" in entity else entity["props"]
                tx.run(
                    f"MERGE (n:{entity['label']} {{id: $id}}) SET n += $props",
                    id=entity["id"],
                    props={k: v for k, v in full_props.items() if k not in JOIN_FIELDS},
                )
                if "new_props" in entity:
                    _prune_derived_edges(tx, entity["label"], entity["id"])
                _derive_edges(tx, entity["label"], entity["id"], full_props)
            rels_after = _count_rels(tx, kept_ids) if kept_ids else 0
            if new_entities is not None:
                save_existing_entities(tx, doc_id, new_entities)
        stats["neo4j_query_time_ms"] = round((time.perf_counter() - query_start) * 1000, 2)
        stats["edges_added"] = max(rels_after - rels_before, 0)

    stats["update_time_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return stats
