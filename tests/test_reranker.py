"""Reranker tests — deterministic BM25 lexical mode + interface contract.

The cross-encoder model is NOT downloaded in tests; the lexical (pure-Python
BM25) backend is the deterministic path exercised here. `make_reranker("auto")`
must return a working reranker either way.
"""

from graphrag.reranker import CrossEncoderReranker, LexicalReranker, make_reranker

NODES = [
    {"id": "POL-0005", "label": "Policy",
     "props": {"policy_number": "CP-2023-0005", "status": "ACTIVE", "premium": 54036.0}},
    {"id": "PH-0005", "label": "Policyholder", "props": {"name": "Scott Brown"}},
    {"id": "COV-0017", "label": "Coverage",
     "props": {"code": "CPP-A", "category": "PROPERTY", "limit": 5000000.0}},
]


def _rank(query: str, nodes: list[dict]) -> list[str]:
    return [node["id"] for node, _ in LexicalReranker().rank(query, nodes)]


def test_lexical_ranks_id_mention_first():
    assert _rank("What is the status of policy POL-0005?", NODES)[0] == "POL-0005"


def test_lexical_sorted_desc():
    ranked = LexicalReranker().rank("scott brown", NODES)
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_interface_returns_tuples():
    ranked = make_reranker("lexical").rank("coverage of POL-0005", NODES)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in ranked)
    assert all(isinstance(s, float) for _, s in ranked)


def test_bm25_length_normalization():
    # a terse node must outrank a much longer node that mentions the term once
    # (BM25's length normalization — no unfair win for verbosity)
    short = {"id": "CLM-A", "label": "Claim",
             "props": {"status": "PAID", "cause": "Fire damage"}}
    long = {"id": "CLM-B", "label": "Claim",
            "props": {"status": "PAID", "cause": "Fire damage",
                      "description": "word " * 200}}
    assert _rank("paid fire damage", [long, short])[0] == "CLM-A"


def test_bm25_term_frequency_saturation():
    # doubling the term frequency must NOT double the score (saturation)
    one = {"id": "CLM-1", "label": "Claim",
           "props": {"status": "PAID", "cause": "Theft / burglary"}}
    two = {"id": "CLM-2", "label": "Claim",
           "props": {"status": "PAID", "cause": "Fire damage",
                      "description": "payment status confirmed paid"}}
    scored = LexicalReranker().rank("paid", [one, two])
    scores = dict((n["id"], s) for n, s in scored)
    assert scores["CLM-2"] > scores["CLM-1"]                # more hits -> higher
    assert scores["CLM-2"] < 2 * scores["CLM-1"]            # but saturated, < 2x


def test_bm25_prefix_matching():
    # "caused" prefix-matches "cause" without a stemmer
    claim = {"id": "CLM-A", "label": "Claim",
             "props": {"status": "SUBMITTED", "cause": "Fire damage"}}
    policy = {"id": "POL-B", "label": "Policy",
              "props": {"status": "ACTIVE", "premium": 5000.0}}
    assert _rank("claims caused by fire", [policy, claim])[0] == "CLM-A"


def test_bm25_entity_id_bonus_dominates():
    # an explicitly-named claim id jumps to rank 1 even with zero token overlap
    named = {"id": "CLM-0003", "label": "Claim",
             "props": {"status": "SUBMITTED", "cause": "Theft / burglary"}}
    relevant = {"id": "CLM-9999", "label": "Claim",
                "props": {"status": "PAID", "cause": "Fire damage"}}
    assert _rank("paid fire damage claims CLM-0003", [relevant, named])[0] == "CLM-0003"


def test_factory_resolves_modes():
    assert isinstance(make_reranker("lexical"), LexicalReranker)
    # auto must resolve to something callable
    assert hasattr(make_reranker("auto"), "rank")


def test_cross_encoder_lazy_load():
    # constructing must NOT download the model — only rank() loads it
    r = CrossEncoderReranker()
    assert r._model is None
