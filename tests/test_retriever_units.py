"""Pure-function unit tests for the retriever's seed-detection helpers.

No Neo4j needed — these run everywhere (kept out of test_retriever.py, which is
gated on a live database).
"""

from graphrag.graph_retriever import _numeric_prop_focus, _singular


def test_singular_strips_plurals():
    assert _singular("doctors") == "doctor"
    assert _singular("losses") == "loss"
    assert _singular("thefts") == "theft"
    assert _singular("claims") == "claim"
    assert _singular("policies") == "policy"
    # must not mangle singulars or "ss"/"is"-ending words
    assert _singular("doctor") == "doctor"
    assert _singular("address") == "address"
    assert _singular("analysis") == "analysis"
    assert _singular("urban") == "urban"
    assert _singular("gas") == "gas"


def test_numeric_prop_focus():
    # naming the prop narrows the scan to that prop's labels
    assert _numeric_prop_focus("Show me policies with premium over $5,000") == [("premium", "Policy")]
    assert _numeric_prop_focus("policies with deductible under $1,000") == [
        ("deductible", "Policy"), ("deductible", "Coverage")]
    # no prop word -> fall back by answer type
    assert _numeric_prop_focus("Show me all claims over $100,000") == [("amount", "Claim")]
    assert _numeric_prop_focus("coverage limit above 5,000,000") == [("limit", "Coverage")]
    assert _numeric_prop_focus("show me everything") is None
