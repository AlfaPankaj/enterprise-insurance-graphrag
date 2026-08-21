"""v2 answer-cache tests (unit + pipeline-level with a scripted driver)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest

from graphrag.cache import QueryCache, build_cache_key, graph_revision
from graphrag.config import settings
from graphrag.identity import UserIdentity
from graphrag.query_pipeline import run_query

# ---------------------------------------------------------------------------
# scripted graph driver (same shape as test_query_pipeline_v2's, + revision)
# ---------------------------------------------------------------------------

NODES = {
    "CLM-0003": ("Claim", {"id": "CLM-0003", "status": "IN_REVIEW",
                           "amount": 5000.0, "cause": "water damage"}),
    "FRD-CLM-0003": ("FraudFlag", {"id": "FRD-CLM-0003", "severity": "MEDIUM",
                                   "confidence": 0.8, "reason": "amount mismatch"}),
}
EDGES = [("CLM-0003", "FRAUD_DETECTED", "FRD-CLM-0003")]


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
    def __init__(self, query, kwargs, driver):
        self.query, self.kwargs, self.driver = query, kwargs, driver
        self._rows = self._resolve()

    def _resolve(self):
        q = self.query
        if "MATCH (d:Dataset)" in q:                       # revision lookup
            return [{"name": self.driver.dataset_name, "rev": self.driver.dataset_rev}]
        if "labels(n) AS labels, n.id AS id" in q:
            eid = self.kwargs.get("id")
            return [{"labels": [NODES[eid][0]], "id": eid}] if eid in NODES else []
        if "labels(n) AS labels, n" in q:
            ids = self.kwargs.get("ids") or []
            return [{"labels": [NODES[i][0]], "n": _Node(NODES[i][1])}
                    for i in ids if i in NODES]
        if "MATCH (n)<-[r]-(m)" in q:
            frontier = self.kwargs.get("frontier") or []
            return [{"src": src, "type": rel, "dst": dst}
                    for src, rel, dst in EDGES if dst in frontier]
        if "-[r]->" in q:
            frontier = self.kwargs.get("frontier") or []
            return [{"src": src, "type": rel, "dst": dst}
                    for src, rel, dst in EDGES if src in frontier]
        return []

    def data(self):
        return self._rows

    def single(self):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


class _Session:
    def __init__(self, driver):
        self.driver = driver

    def run(self, query, **kwargs):
        self.driver.calls.append((query, kwargs))
        return _Cursor(query, kwargs, self.driver)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Driver:
    def __init__(self, name="synthetic", rev=0):
        self.dataset_name = name
        self.dataset_rev = rev
        self.calls: list = []

    def session(self, **kw):
        return _Session(self)


IDENTITY = UserIdentity("u-1", frozenset({"analyst"}), "bank-a", "jwt")


@pytest.fixture(autouse=True)
def _reset():
    yield
    settings.CACHE_ENABLED = False
    from graphrag.cache import query_cache
    query_cache.clear()


@pytest.fixture
def audit_dir(tmp_path):
    from graphrag import query_pipeline as qp
    from graphrag.traversal_logger import AuditStore

    store = AuditStore(tmp_path / "audit.jsonl")
    old = qp.audit_store
    qp.audit_store = store
    yield store
    qp.audit_store = old


# ---------------------------------------------------------------------------
# QueryCache unit behavior
# ---------------------------------------------------------------------------

def test_cache_ttl_expiry(monkeypatch):
    import graphrag.cache as cache_mod
    now = [1000.0]
    monkeypatch.setattr(cache_mod.time, "monotonic", lambda: now[0])
    c = QueryCache(max_entries=10, ttl_s=5)
    c.put("k", {"v": 1})
    assert c.get("k")["v"] == 1
    now[0] += 6
    assert c.get("k") is None          # expired


def test_cache_lru_eviction():
    c = QueryCache(max_entries=2, ttl_s=60)
    c.put("a", {"v": "a"})
    c.put("b", {"v": "b"})
    c.get("a")                          # refresh a's recency
    c.put("c", {"v": "c"})
    assert c.get("b") is None           # least-recently-used evicted
    assert c.get("a")["v"] == "a"
    assert c.get("c")["v"] == "c"
    assert len(c) == 2


def test_cache_key_stability_and_sensitivity():
    k1 = build_cache_key(query="q", hops=2, tenant="t")
    k2 = build_cache_key(query="q", hops=2, tenant="t")
    k3 = build_cache_key(query="q", hops=2, tenant="other")
    assert k1 == k2
    assert k1 != k3


def test_graph_revision_unreadable_returns_none():
    class _NoSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def run(self, *a, **k):
            raise RuntimeError("down")

    class _BrokenDriver:
        def session(self, **kw):
            return _NoSession()

    assert graph_revision(_BrokenDriver()) is None


# ---------------------------------------------------------------------------
# pipeline-level cache behavior
# ---------------------------------------------------------------------------

def test_cache_hit_skips_retrieval(monkeypatch, audit_dir):
    monkeypatch.setattr(settings, "CACHE_ENABLED", True)
    driver = _Driver()
    res1 = run_query(driver, "Does claim CLM-0003 have a fraud flag?",
                     identity=IDENTITY, answer_mode="extractive")
    assert not res1.get("cached")
    calls_after_first = len(driver.calls)

    res2 = run_query(driver, "Does claim CLM-0003 have a fraud flag?",
                     identity=IDENTITY, answer_mode="extractive")
    assert res2["cached"] is True
    assert res2["answer"] == res1["answer"]
    # a hit performs NO retrieval — only the revision lookup
    assert len(driver.calls) == calls_after_first + 1
    assert "MATCH (d:Dataset)" in driver.calls[-1][0]
    # fresh audit record, marked cached
    recs = audit_dir.recent(2)
    assert recs[0]["cached"] is True and recs[0]["audit_id"] != recs[1]["audit_id"]
    assert audit_dir.verify()["valid"]


def test_cache_invalidates_on_revision_bump(monkeypatch, audit_dir):
    monkeypatch.setattr(settings, "CACHE_ENABLED", True)
    driver = _Driver(rev=0)
    run_query(driver, "Does claim CLM-0003 have a fraud flag?",
              identity=IDENTITY, answer_mode="extractive")
    driver.dataset_rev = 1                       # a write happened elsewhere
    res = run_query(driver, "Does claim CLM-0003 have a fraud flag?",
                    identity=IDENTITY, answer_mode="extractive")
    assert not res.get("cached")                 # stale entry rejected


def test_cache_separates_tenants(monkeypatch, audit_dir):
    monkeypatch.setattr(settings, "CACHE_ENABLED", True)
    monkeypatch.setattr(settings, "TENANT_MODE", "column")
    driver = _Driver()
    a = UserIdentity("a", frozenset({"analyst"}), "bank-a", "jwt")
    b = UserIdentity("b", frozenset({"analyst"}), "bank-b", "jwt")
    r1 = run_query(driver, "Does claim CLM-0003 have a fraud flag?",
                   identity=a, answer_mode="extractive")
    r2 = run_query(driver, "Does claim CLM-0003 have a fraud flag?",
                   identity=b, answer_mode="extractive")
    assert not r1.get("cached") and not r2.get("cached")  # distinct entries


def test_cache_separates_pii_scopes(monkeypatch, audit_dir):
    monkeypatch.setattr(settings, "CACHE_ENABLED", True)
    monkeypatch.setattr(settings, "PII_MODE", "mask")
    driver = _Driver()
    analyst = UserIdentity("a", frozenset({"analyst"}), "t", "jwt")
    auditor = UserIdentity("ad", frozenset({"admin", "auditor"}), "t", "jwt")
    run_query(driver, "Who investigates claim CLM-0003?",
              identity=analyst, answer_mode="extractive")
    res = run_query(driver, "Who investigates claim CLM-0003?",
                    identity=auditor, answer_mode="extractive")
    assert not res.get("cached")                 # auditor must not get analyst's masked entry


def test_cache_disabled_by_default(monkeypatch, audit_dir):
    driver = _Driver()
    res1 = run_query(driver, "Does claim CLM-0003 have a fraud flag?",
                     identity=IDENTITY, answer_mode="extractive")
    res2 = run_query(driver, "Does claim CLM-0003 have a fraud flag?",
                     identity=IDENTITY, answer_mode="extractive")
    assert not res1.get("cached") and not res2.get("cached")
