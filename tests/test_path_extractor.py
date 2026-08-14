"""Unit tests for traversal path extraction + Cypher reconstruction (Phase 4)."""

from graphrag.path_extractor import (build_cypher, extract_traversal_paths,
                                     traversal_summary)

SUBGRAPH = {
    "query": "Does claim CLM-0003 have a fraud flag?",
    "seeds": [{"id": "CLM-0003", "label": "Claim", "kind": "id"}],
    "nodes": [
        {"id": "CLM-0003", "label": "Claim", "props": {"amount": 5000.0}},
        {"id": "FRD-CLM-0003", "label": "FraudFlag", "props": {"severity": "MEDIUM"}},
        {"id": "POL-0005", "label": "Policy", "props": {"status": "ACTIVE"}},
    ],
    "edges": [
        {"source": "CLM-0003", "type": "FRAUD_DETECTED", "target": "FRD-CLM-0003"},
        {"source": "POL-0005", "type": "HAS_CLAIM", "target": "CLM-0003"},
    ],
}


def test_extract_traversal_paths_from_seed():
    paths = extract_traversal_paths(SUBGRAPH)
    assert paths, "expected at least one chain from the seed"
    starts = {p["nodes"][0] for p in paths}
    assert starts == {"CLM-0003"}
    # the fraud edge must appear with its stored direction
    assert any(e == ["CLM-0003", "FRAUD_DETECTED", "FRD-CLM-0003"]
               for p in paths for e in p["edges"])


def test_paths_never_revisit_nodes():
    for p in extract_traversal_paths(SUBGRAPH):
        assert len(p["nodes"]) == len(set(p["nodes"])), "chain must be simple"
        assert len(p["edges"]) == len(p["nodes"]) - 1


def test_traversal_summary_dedupes_edges():
    s = traversal_summary(SUBGRAPH)
    assert s["nodes_visited"] == ["CLM-0003", "FRD-CLM-0003", "POL-0005"]
    keys = {(e[0], e[1], e[2]) for e in s["edges_traversed"]}
    assert len(keys) == len(s["edges_traversed"]), "edges must be deduplicated"
    assert ("POL-0005", "HAS_CLAIM", "CLM-0003") in keys


def test_build_cypher_includes_seed_and_rel():
    cypher = build_cypher(SUBGRAPH, max_hops=2)
    assert 'MATCH (s0:Claim {id: "CLM-0003"})' in cypher
    assert "FRAUD_DETECTED" in cypher
    assert "max 2 hops" in cypher
    # concrete answer path uses the extracted chain
    assert "MATCH p = (n0 {id: \"CLM-0003\"})-[:FRAUD_DETECTED]->(n1 {id: \"FRD-CLM-0003\"})" in cypher


def test_build_cypher_no_seeds_keyword_scan():
    sub = {
        "query": "Show me claims caused by Fire",
        "seeds": [],
        "nodes": [{"id": "CLM-0009", "label": "Claim", "props": {"cause": "fire"}}],
        "edges": [],
    }
    cypher = build_cypher(sub, max_hops=2)
    assert "keyword scan" in cypher
    assert '"fire"' in cypher  # keyword tokens embedded
    # expansion uses plain variable-length MATCH — executable without APOC
    assert "MATCH p = (seed)-[*1..2]-(n)" in cypher
    assert "apoc" not in cypher


def test_build_cypher_executable_pattern():
    """The reported expansion must be runnable on plain Neo4j (no APOC)."""
    cypher = build_cypher(SUBGRAPH, max_hops=2)
    assert "MATCH p = (s0 {id: \"CLM-0003\"})-[*1..2]-(n)" in cypher
    assert "apoc" not in cypher


def test_chains_bounded_by_max_hops():
    # chains never exceed seed + max_hops hops, whatever the graph shape
    deep = {
        "query": "q",
        "seeds": [{"id": "A", "label": "Claim", "kind": "id"}],
        "nodes": [{"id": x, "label": "Claim", "props": {}} for x in "ABCDEFG"],
        "edges": [{"source": a, "type": "LINK", "target": b}
                   for a, b in zip("ABCDEF", "BCDEFG")],
    }
    for p in extract_traversal_paths(deep, max_hops=2):
        assert len(p["nodes"]) <= 3
    # max_hops=4 reaches further (seed + 4 hops)
    assert any(len(p["nodes"]) >= 5 for p in extract_traversal_paths(deep, max_hops=4))


def test_build_cypher_empty_subgraph():
    cypher = build_cypher({"query": "?", "seeds": [], "nodes": [], "edges": []}, max_hops=1)
    assert "seed lookup" in cypher  # still renders something usable
