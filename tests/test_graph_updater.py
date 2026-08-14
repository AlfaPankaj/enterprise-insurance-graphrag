"""Integration tests for graph_updater against a live Neo4j.

Skipped automatically when Neo4j is unreachable. Uses a unique test doc id
and cleans up after itself, so it never touches the seeded demo graph.
"""

import pytest
from neo4j import GraphDatabase

from graphrag.change_detector import detect_changes
from graphrag.graph_store import get_existing_entities
from graphrag.graph_updater import update_graph_surgically

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "graphrag-demo"
TEST_DOC = "TEST-DOC-001"
TEST_DOC_2 = "TEST-DOC-002"
TEST_IDS = ["POL-TEST-1", "PH-TEST-1", "CLM-TEST-1", "FRD-TEST-1", "INV-TEST-1", "INV-TEST-2"]

_BASE = {
    "Policy": {
        "POL-TEST-1": {
            "status": "ACTIVE",
            "premium": 1000.0,
            "policyholder_id": "PH-TEST-1",
        }
    },
    "Policyholder": {"PH-TEST-1": {"name": "Test Person"}},
}

_MODIFIED = {
    "Policy": {
        "POL-TEST-1": {
            "status": "EXPIRED",
            "premium": 1000.0,
            "policyholder_id": "PH-TEST-1",
        }
    },
    "Policyholder": {"PH-TEST-1": {"name": "Test Person"}},
}

_CLAIM_DOC = {
    "Claim": {
        "CLM-TEST-1": {
            "id": "CLM-TEST-1",
            "policy_id": "POL-TEST-1",
            "investigator_id": "INV-TEST-1",
            "amount": 1000.0,
        }
    },
    "Policy": {"POL-TEST-1": {"id": "POL-TEST-1", "policyholder_id": "PH-TEST-1", "status": "ACTIVE"}},
    "Policyholder": {"PH-TEST-1": {"id": "PH-TEST-1", "name": "Test Person"}},
    "FraudFlag": {"FRD-TEST-1": {"id": "FRD-TEST-1", "claim_id": "CLM-TEST-1", "severity": "HIGH"}},
    "Investigator": {"INV-TEST-1": {"id": "INV-TEST-1", "name": "Inv One"}},
}


def _neo4j_available() -> bool:
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _neo4j_available(), reason="Neo4j not reachable")


@pytest.fixture()
def driver():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    # clean slate for the test docs
    with driver.session() as session:
        session.run("MATCH (n) WHERE n.id IN $ids DETACH DELETE n", ids=TEST_IDS)
        session.run("MATCH (n:DocSnapshot) WHERE n.doc_id IN [$a, $b] DELETE n", a=TEST_DOC, b=TEST_DOC_2)
    yield driver
    with driver.session() as session:
        session.run("MATCH (n) WHERE n.id IN $ids DETACH DELETE n", ids=TEST_IDS)
        session.run("MATCH (n:DocSnapshot) WHERE n.doc_id IN [$a, $b] DELETE n", a=TEST_DOC, b=TEST_DOC_2)
    driver.close()


def test_full_cdc_flow(driver):
    # 1) first ingest: no snapshot -> everything added + edges derived
    with driver.session() as session:
        assert get_existing_entities(session, TEST_DOC) == {}
        stats = update_graph_surgically(driver, TEST_DOC, detect_changes({}, _BASE))
    assert stats["entities_added"] == 2
    assert stats["entities_updated"] == 0
    assert stats["entities_deleted"] == 0
    assert stats["edges_added"] >= 1  # HAS_POLICY derived from policyholder_id

    # 2) idempotent re-upload: no changes
    stats = update_graph_surgically(driver, TEST_DOC, detect_changes(_BASE, _BASE))
    assert stats["entities_added"] == 0 and stats["entities_updated"] == 0

    # 3) modified props -> surgical SET + accurate counts
    stats = update_graph_surgically(driver, TEST_DOC, detect_changes(_BASE, _MODIFIED))
    assert stats["entities_updated"] == 1

    # 4) verify the node actually changed and the edge exists
    with driver.session() as session:
        row = session.run("MATCH (p:Policy {id: 'POL-TEST-1'}) RETURN p.status AS status").single()
        assert row["status"] == "EXPIRED"
        has_edge = session.run(
            "MATCH (ph:Policyholder {id: 'PH-TEST-1'})-[:HAS_POLICY]->(p:Policy {id: 'POL-TEST-1'}) "
            "RETURN count(*) AS c"
        ).single()["c"]
        assert has_edge == 1

    # 5) deletion
    stats = update_graph_surgically(driver, TEST_DOC, detect_changes(_MODIFIED, {}))
    assert stats["entities_deleted"] == 2
    with driver.session() as session:
        assert session.run("MATCH (p:Policy {id: 'POL-TEST-1'}) RETURN count(*) AS c").single()["c"] == 0


def test_edge_derivation_and_restoration(driver):
    """CDC-added entities must get their derived edges; modifications must not
    leave stale ones; re-added entities must restore them."""

    # direction-agnostic: HAS_CLAIM/INVESTIGATES_CLAIM point INTO the claim,
    # FRAUD_DETECTED points OUT of it.
    def edges_of(rtype: str, start: str) -> int:
        with driver.session() as s:
            return s.run(
                f"MATCH (n {{id: $start}})-[r:{rtype}]-() RETURN count(r) AS c", start=start
            ).single()["c"]

    def edge_targets(rtype: str, start: str) -> set[str]:
        with driver.session() as s:
            return {r["target"] for r in s.run(
                f"MATCH (n {{id: $start}})-[r:{rtype}]-(m) RETURN m.id AS target", start=start
            )}

    # 1) ingest a claim doc: HAS_CLAIM + FRAUD_DETECTED + INVESTIGATES_CLAIM all derived
    stats = update_graph_surgically(driver, TEST_DOC, detect_changes({}, _CLAIM_DOC))
    assert stats["entities_added"] == 5
    assert edges_of("HAS_CLAIM", "CLM-TEST-1") == 1
    assert edges_of("FRAUD_DETECTED", "CLM-TEST-1") == 1
    assert edges_of("INVESTIGATES_CLAIM", "CLM-TEST-1") == 1

    # 2) re-assign the investigator -> old edge pruned, only the new one remains
    re_assigned = {label: dict(ents) for label, ents in _CLAIM_DOC.items()}
    re_assigned["Claim"]["CLM-TEST-1"] = {
        **_CLAIM_DOC["Claim"]["CLM-TEST-1"], "investigator_id": "INV-TEST-2"
    }
    re_assigned["Investigator"] = {"INV-TEST-2": {"id": "INV-TEST-2", "name": "Inv Two"}}
    stats = update_graph_surgically(driver, TEST_DOC, detect_changes(_CLAIM_DOC, re_assigned))
    assert stats["entities_updated"] == 1
    assert edges_of("INVESTIGATES_CLAIM", "CLM-TEST-1") == 1
    assert edge_targets("INVESTIGATES_CLAIM", "CLM-TEST-1") == {"INV-TEST-2"}

    # 3) fraud flag deleted then re-added -> FRAUD_DETECTED edge restored
    without_flag = {k: v for k, v in _CLAIM_DOC.items() if k != "FraudFlag"}
    stats = update_graph_surgically(driver, TEST_DOC, detect_changes(_CLAIM_DOC, without_flag))
    assert stats["entities_deleted"] == 1
    assert edges_of("FRAUD_DETECTED", "CLM-TEST-1") == 0

    stats = update_graph_surgically(driver, TEST_DOC, detect_changes(without_flag, _CLAIM_DOC))
    assert stats["entities_added"] == 1
    assert edges_of("FRAUD_DETECTED", "CLM-TEST-1") == 1


def test_snapshot_roundtrip(driver):
    from graphrag.graph_store import save_existing_entities

    with driver.session() as session:
        save_existing_entities(session, TEST_DOC, _BASE)
        assert get_existing_entities(session, TEST_DOC) == _BASE


def test_reference_counting(driver):
    """An entity still referenced by another doc's snapshot must survive deletion."""
    from graphrag.graph_store import save_existing_entities

    # ingest normally — creates the nodes + TEST_DOC snapshot
    stats = update_graph_surgically(driver, TEST_DOC, detect_changes({}, _BASE))
    assert stats["entities_added"] == 2

    # a second document claims the same entities (e.g. an endorsement that
    # also appears in its parent policy PDF)
    with driver.session() as session:
        save_existing_entities(session, TEST_DOC_2, _BASE)

    # Delete everything from TEST_DOC — entities are still referenced by
    # TEST_DOC_2's snapshot, so they must be skipped, not deleted.
    stats = update_graph_surgically(driver, TEST_DOC, detect_changes(_BASE, {}))
    assert stats["entities_deleted"] == 2
    assert stats["deleted_skipped"] == 2
    with driver.session() as session:
        assert session.run(
            "MATCH (p:Policy {id: 'POL-TEST-1'}) RETURN count(*) AS c"
        ).single()["c"] == 1

    # Once the last referencing document is re-ingested empty, the entities go.
    stats = update_graph_surgically(driver, TEST_DOC_2, detect_changes(_BASE, {}))
    assert stats["entities_deleted"] == 2
    assert stats["deleted_skipped"] == 0
    with driver.session() as session:
        assert session.run(
            "MATCH (p:Policy {id: 'POL-TEST-1'}) RETURN count(*) AS c"
        ).single()["c"] == 0
