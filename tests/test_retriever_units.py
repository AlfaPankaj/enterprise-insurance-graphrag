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


def test_threshold_numbers_exclude_id_digits():
    """Id digits are anchors, not amounts (negative-probe fix, v2 re-run).

    A nonexistent-id query must not degrade into "amount >= 99999" and
    return real high-amount claims — it must refuse (empty seeding).
    """
    from graphrag.graph_retriever import _threshold_numbers
    assert _threshold_numbers("What is the status of claim CLM-99999?") == []
    assert _threshold_numbers("Is claim CLM-99999 flagged as fraud?") == []
    assert _threshold_numbers("Show me claims under policy POL-99999") == []
    # genuine thresholds keep working, beside ids or standalone
    assert _threshold_numbers("claims over $100,000") == [100000]
    assert _threshold_numbers("claim CLM-0003 with amount 5000") == [5000]
    assert _threshold_numbers(
        "paid claims under policy POL-0084 over $50,000") == [50000]
