"""Context pruner tests — token budget enforcement + edge filtering."""

from graphrag.context_pruner import prune_context
from graphrag.graph_retriever import serialize_node
from graphrag.token_counter import count_tokens

NODES = [
    {"id": "POL-0005", "label": "Policy",
     "props": {"policy_number": "CP-2023-0005", "status": "ACTIVE", "premium": 54036.0}},
    {"id": "PH-0005", "label": "Policyholder", "props": {"name": "Scott Brown"}},
    {"id": "COV-0017", "label": "Coverage",
     "props": {"code": "CPP-A", "category": "PROPERTY", "limit": 5000000.0}},
    {"id": "COV-0018", "label": "Coverage",
     "props": {"code": "CPP-B", "category": "PROPERTY", "limit": 2000000.0}},
]

EDGES = [
    {"source": "PH-0005", "type": "HAS_POLICY", "target": "POL-0005"},
    {"source": "POL-0005", "type": "COVERS", "target": "COV-0017"},
    {"source": "POL-0005", "type": "COVERS", "target": "COV-0018"},
]


def test_full_budget_keeps_everything():
    pruned = prune_context([(n, 0.0) for n in NODES], token_budget=10_000, edges=EDGES)
    assert pruned["node_count"] == 4
    assert pruned["dropped"] == []
    assert len(pruned["edges"]) == 3


def test_small_budget_keeps_first_drops_rest():
    # budget = exactly the first node's cost -> only it fits
    first_cost = count_tokens(serialize_node(NODES[0]))
    pruned = prune_context([(n, 0.0) for n in NODES], token_budget=first_cost, edges=EDGES)
    assert 0 < pruned["tokens"] <= first_cost
    assert pruned["node_count"] == 1
    assert pruned["kept"] == ["POL-0005"]  # first (best-ranked) node always kept
    assert set(pruned["dropped"]) == {"PH-0005", "COV-0017", "COV-0018"}


def test_edges_filtered_to_kept_nodes():
    first_cost = count_tokens(serialize_node(NODES[0]))
    pruned = prune_context([(n, 0.0) for n in NODES], token_budget=first_cost, edges=EDGES)
    # only the first node survives -> no edge has both endpoints present
    assert pruned["edges"] == []


def test_budget_respected_in_tokens():
    pruned = prune_context([(n, 0.0) for n in NODES], token_budget=64, edges=EDGES)
    assert pruned["tokens"] <= 64
    # tokens reported == tokens of the serialized pruned text
    assert count_tokens(pruned["text"]) == pruned["tokens"]


def test_bare_node_list_accepted():
    pruned = prune_context(NODES, token_budget=10_000, edges=EDGES)
    assert pruned["node_count"] == 4
