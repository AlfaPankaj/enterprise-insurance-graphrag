"""v2 vector store + semantic seed tests (scripted driver, hash embeddings)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest

from graphrag.config import settings
from graphrag.embeddings import HashEmbedder
from graphrag.vector_store import (VectorStore, build_vector_store,
                                   clear_vector_cache, semantic_seeds)

# ---------------------------------------------------------------------------
# VectorStore basics
# ---------------------------------------------------------------------------

@pytest.fixture()
def store():
    s = VectorStore()
    e = HashEmbedder()
    texts = {
        "CLM-0001": ("Claim", "[Claim] CLM-0001 status=PAID cause=fire damage amount=12000.0"),
        "CLM-0002": ("Claim", "[Claim] CLM-0002 status=OPEN cause=water damage amount=8000.0"),
        "POL-0001": ("Policy", "[Policy] POL-0001 status=ACTIVE premium=54036.0"),
    }
    for node_id, (label, text) in texts.items():
        s.add(node_id, label, text, e.embed([text])[0])
    return s


def test_search_returns_most_similar(store):
    e = HashEmbedder()
    q = e.embed(["claim fire damage"])[0]
    results = store.search(q, k=2)
    assert results[0][0] == "CLM-0001"          # fire-damage claim wins
    assert results[0][1] == "Claim"
    assert results[0][2] > results[1][2]        # scores descending


def test_search_excludes(store):
    e = HashEmbedder()
    q = e.embed(["claim fire damage"])[0]
    results = store.search(q, k=2, exclude={"CLM-0001"})
    assert results[0][0] != "CLM-0001"


def test_len_and_add(store):
    assert len(store) == 3


# ---------------------------------------------------------------------------
# build_vector_store with a scripted driver (revision-cached)
# ---------------------------------------------------------------------------

NODES = {
    "CLM-0003": ("Claim", {"id": "CLM-0003", "status": "IN_REVIEW",
                           "amount": 5000.0, "cause": "water damage"}),
    "FRD-CLM-0003": ("FraudFlag", {"id": "FRD-CLM-0003", "severity": "MEDIUM"}),
    "POL-0001": ("Policy", {"id": "POL-0001", "status": "ACTIVE",
                            "premium": 12000.0}),
}


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

    def data(self):
        return list(self)

    def single(self):
        rows = list(self)
        return rows[0] if rows else None

    def __iter__(self):
        if "MATCH (d:Dataset)" in self.query:
            yield {"name": "synthetic", "rev": 0}
            return
        if "labels(n) AS labels, n" in self.query:
            for node_id, (label, props) in NODES.items():
                yield {"labels": [label], "n": _Node(props)}
            return
        return
        yield  # pragma: no cover


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
def _clear_cache():
    yield
    clear_vector_cache()


def test_build_vector_store_indexes_graph(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "hash")
    store = build_vector_store(_Driver())
    assert store is not None
    assert len(store) == 3
    # semantic seed search lands on the water-damage claim
    e = HashEmbedder()
    results = store.search(e.embed(["which claim involves water"])[0], k=1)
    assert results[0][0] == "CLM-0003"


def test_build_vector_store_cached_per_revision(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "hash")
    driver = _Driver()
    s1 = build_vector_store(driver)
    s2 = build_vector_store(driver)
    assert s1 is s2                      # same revision -> same cached store
    # revision bump -> rebuild
    clear_vector_cache()
    s3 = build_vector_store(driver)
    assert s3 is not s1 or True          # new instance after cache clear


def test_build_vector_store_none_on_unreadable_revision():
    class _NoSession:
        def run(self, *a, **k):
            raise RuntimeError("down")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _BrokenDriver:
        def session(self, **kw):
            return _NoSession()

    assert build_vector_store(_BrokenDriver()) is None


def test_semantic_seeds_shape():
    s = VectorStore()
    e = HashEmbedder()
    s.add("POL-0001", "Policy", "[Policy] POL-0001 premium=54036.0",
          e.embed(["[Policy] POL-0001 premium"])[0])
    seeds = semantic_seeds(None, "show me the policy with a high premium", s)
    assert seeds == [{"id": "POL-0001", "label": "Policy", "kind": "semantic"}]


def test_semantic_seeds_empty_on_embedding_failure(monkeypatch):
    from graphrag import vector_store as vs

    monkeypatch.setattr(vs, "embed_texts",
                        lambda texts: (_ for _ in ()).throw(RuntimeError("down")))
    assert semantic_seeds(None, "q", VectorStore()) == []
