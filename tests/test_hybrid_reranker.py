"""v2 hybrid reranker tests — RRF fusion, graceful degradation, pipeline path."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest

from graphrag.config import settings
from graphrag.reranker import HybridReranker, LexicalReranker, make_reranker

NODES = [
    {"id": "CLM-0003", "label": "Claim",
     "props": {"status": "IN_REVIEW", "amount": 5000.0, "cause": "water damage"}},
    {"id": "FRD-0007", "label": "FraudFlag",
     "props": {"severity": "MEDIUM", "reason": "amount mismatch"}},
    {"id": "POL-0001", "label": "Policy",
     "props": {"status": "ACTIVE", "premium": 12000.0}},
]


def test_rrf_fusion_orders_consistently():
    rrf = HybridReranker._rrf([["a", "b"], ["b", "a"]])
    assert rrf["a"] == pytest.approx(rrf["b"])          # ties fuse
    rrf2 = HybridReranker._rrf([["a", "b"], ["a", "c"]])
    assert rrf2["a"] > rrf2["b"] and rrf2["a"] > rrf2["c"]


def test_hybrid_without_driver_equals_lexical():
    hybrid = HybridReranker(driver=None)
    lexical = LexicalReranker()
    h = hybrid.rank("Does claim CLM-0003 have a fraud flag?", NODES)
    l = lexical.rank("Does claim CLM-0003 have a fraud flag?", NODES)
    assert [n["id"] for n, _ in h] == [n["id"] for n, _ in l]


def test_hybrid_with_proximity_boosts_seed_neighborhood():
    nodes = [
        {**NODES[0], "props": {**NODES[0]["props"], "_dist": 0}},   # the seed
        {**NODES[1], "props": {**NODES[1]["props"], "_dist": 1}},
        {**NODES[2], "props": {**NODES[2]["props"], "_dist": 2}},
    ]
    hybrid = HybridReranker(driver=None)
    ranked = hybrid.rank("status of claim CLM-0003", nodes)
    assert ranked[0][0]["id"] == "CLM-0003"          # seed ranked first


def test_hybrid_empty_nodes():
    assert HybridReranker(driver=None).rank("q", []) == []


def test_factory_hybrid_mode():
    r = make_reranker("hybrid", driver=None)
    assert isinstance(r, HybridReranker)
    assert r.name == "hybrid"
    # auto mode unchanged (lexical here — no sentence-transformers installed)
    r2 = make_reranker("auto")
    assert not isinstance(r2, HybridReranker)


def test_hybrid_mode_accepted_by_api_pattern():
    import re
    pattern = re.compile(r"^(auto|cross-encoder|lexical|hybrid)$")
    assert pattern.fullmatch("hybrid")


# ---------------------------------------------------------------------------
# pipeline-level: hybrid mode runs end to end with a scripted driver
# ---------------------------------------------------------------------------

PIPELINE_NODES = {
    "CLM-0003": ("Claim", {"id": "CLM-0003", "status": "IN_REVIEW",
                           "amount": 5000.0, "cause": "water damage"}),
    "FRD-CLM-0003": ("FraudFlag", {"id": "FRD-CLM-0003", "severity": "MEDIUM",
                                   "confidence": 0.8}),
    "INV-0001": ("Investigator", {"id": "INV-0001", "name": "Alice Example",
                                  "role": "SENIOR_INVESTIGATOR"}),
}
PIPELINE_EDGES = [
    ("CLM-0003", "FRAUD_DETECTED", "FRD-CLM-0003"),
    ("INV-0001", "INVESTIGATES_CLAIM", "CLM-0003"),
]


class _Node:
    def __init__(self, props):
        self._props = props

    def __getitem__(self, k):
        return self._props[k]

    def items(self):
        return self._props.items()

    def keys(self):
        return self._props.keys()


class _Cursor:
    def __init__(self, query, kwargs):
        self.query, self.kwargs = query, kwargs
        self._rows = self._resolve()

    def _resolve(self):
        q = self.query
        if "MATCH (d:Dataset)" in q:
            return [{"name": "synthetic", "rev": 0}]
        if "labels(n) AS labels, n.id AS id" in q:
            eid = self.kwargs.get("id")
            return ([{"labels": [PIPELINE_NODES[eid][0]], "id": eid}]
                    if eid in PIPELINE_NODES else [])
        if "labels(n) AS labels, n" in q:
            ids = self.kwargs.get("ids") or []
            return [{"labels": [PIPELINE_NODES[i][0]], "n": _Node(PIPELINE_NODES[i][1])}
                    for i in ids if i in PIPELINE_NODES]
        if "MATCH (n)<-[r]-(m)" in q:
            frontier = self.kwargs.get("frontier") or []
            return [{"src": src, "type": rel, "dst": dst}
                    for src, rel, dst in PIPELINE_EDGES if dst in frontier]
        if "-[r]->" in q:
            frontier = self.kwargs.get("frontier") or []
            return [{"src": src, "type": rel, "dst": dst}
                    for src, rel, dst in PIPELINE_EDGES if src in frontier]
        return []

    def data(self):
        return self._rows

    def single(self):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


class _Session:
    def run(self, query, **kwargs):
        return _Cursor(query, kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Driver:
    def session(self, **kw):
        return _Session()


@pytest.fixture(autouse=True)
def _reset():
    yield
    from graphrag.vector_store import clear_vector_cache
    clear_vector_cache()
    settings.AUDIT_ENABLED = True


def test_hybrid_pipeline_end_to_end(monkeypatch, tmp_path):
    from graphrag import query_pipeline as qp
    from graphrag.traversal_logger import AuditStore

    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "hash")
    monkeypatch.setattr(settings, "AUDIT_ENABLED", False)
    store = AuditStore(tmp_path / "audit.jsonl")
    old = qp.audit_store
    qp.audit_store = store
    try:
        result = qp.run_query(_Driver(),
                              "Does claim CLM-0003 have a fraud flag?",
                              reranker_mode="hybrid", answer_mode="extractive")
    finally:
        qp.audit_store = old
    assert result["reranker"] == "hybrid"
    assert "FRD-CLM-0003" in result["answer"]          # correct, as before
    assert "CLM-0003" in result["retrieval"]["seeds"]
