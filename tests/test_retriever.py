"""Retriever integration tests against the live seeded graph.

Skipped automatically when Neo4j is unreachable. Uses seeded entities
(CLM-0003 has fraud flag FRD-CLM-0003 in the demo dataset).
"""

import pytest
from neo4j import GraphDatabase

from graphrag.graph_retriever import (
    _numeric_hits,
    _numeric_tokens,
    _threshold_direction,
    extract_seed_ids,
    query_tokens,
    retrieve_subgraph,
    serialize_subgraph,
)

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "graphrag-demo"


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


@pytest.fixture(scope="module")
def driver():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    yield driver
    driver.close()


def test_query_tokens_and_seed_ids():
    # schema nouns are stopwords for keyword seeding; value tokens survive
    assert "fire" in query_tokens("Show me claims caused by Fire damage")
    assert "claim" not in query_tokens("Does claim CLM-0003 have a fraud flag?")
    assert extract_seed_ids("Does claim CLM-0003 have a fraud flag?") == ["CLM-0003"]
    assert extract_seed_ids("list everything") == []


def test_numeric_threshold_parsing():
    assert _numeric_tokens("Show me claims over $100,000") == [100000]
    assert _numeric_tokens("claims with amount 5000") == [5000]
    assert _numeric_tokens("no numbers here") == []
    assert _threshold_direction("claims over $100,000") == 1
    assert _threshold_direction("claims under $5,000") == -1
    assert _threshold_direction("claims with amount 5000") == 0


def test_numeric_hits_respects_owning_label():
    # only the owning label's prop counts: Claim.amount hits, Investigator does
    # not (it owns no numeric prop), and a threshold below the value misses
    claim = {"_label": "Claim", "amount": 125000.0}
    investigator = {"_label": "Investigator", "amount": 125000.0}
    assert _numeric_hits(claim, [100000], 1) == 1
    assert _numeric_hits(investigator, [100000], 1) == 0
    assert _numeric_hits(claim, [125000], 0) == 1
    assert _numeric_hits(claim, [100000], -1) == 0


def test_entity_id_seed_found(driver):
    sub = retrieve_subgraph(driver, "Does claim CLM-0003 have a fraud flag?", max_hops=1)
    seed_ids = [s["id"] for s in sub["seeds"]]
    assert "CLM-0003" in seed_ids
    node_ids = {n["id"] for n in sub["nodes"]}
    assert "CLM-0003" in node_ids
    assert "FRD-CLM-0003" in node_ids  # fraud flag is 1 hop away


def test_depth_limits_expansion(driver):
    sub1 = retrieve_subgraph(driver, "claim CLM-0003", max_hops=1)
    sub2 = retrieve_subgraph(driver, "claim CLM-0003", max_hops=2)
    ids1 = {n["id"] for n in sub1["nodes"]}
    ids2 = {n["id"] for n in sub2["nodes"]}
    # policyholder is 2 hops from the claim -> only visible at depth 2
    assert any(i.startswith("PH-") for i in ids2)
    assert not any(i.startswith("PH-") for i in ids1)
    assert ids1 <= ids2


def test_keyword_seeds(driver):
    sub = retrieve_subgraph(driver, "Show me claims caused by Fire damage", max_hops=1)
    seeds = {s["id"] for s in sub["seeds"]}
    # keyword matching must surface fire-damage claims as seeds
    assert any(s.startswith("CLM-") for s in seeds)


def test_numeric_threshold_seeds(driver):
    # "over $100,000" without any id anchor must seed large claims via amounts
    sub = retrieve_subgraph(driver, "Show me all claims over $100,000", max_hops=1)
    seeds = {s["id"] for s in sub["seeds"]}
    assert seeds
    with driver.session() as s:
        for sid in seeds:
            row = s.run("MATCH (n {id: $id}) RETURN n.amount AS a", id=sid).single()
            assert row is not None and row["a"] >= 100000


def test_serialization_contains_nodes_and_edges(driver):
    sub = retrieve_subgraph(driver, "Does claim CLM-0003 have a fraud flag?", max_hops=1)
    text = serialize_subgraph(sub)
    assert "CLM-0003" in text and "FRD-CLM-0003" in text
    assert "FRAUD_DETECTED" in text
    assert text.startswith("QUERY:")
