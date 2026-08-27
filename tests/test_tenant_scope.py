"""v2 tenant isolation plumbing tests (predicate injection, opt-in mode)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag.config import settings
from graphrag.graph_retriever import (tenant_active, tenant_predicate,
                                      retrieve_subgraph)


class _RecordingSession:
    """Fake Neo4j session: records (query, kwargs) and returns empty rows."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def run(self, query, **kwargs):
        self.calls.append((query, kwargs))

        class _Cursor:
            def data(self):
                return []

            def single(self):
                return None

            def __iter__(self):
                return iter(())

        return _Cursor()


class _SessionCtx:
    """Context-manager wrapper so `with driver.session() as session:` works."""

    def __init__(self, session):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, *exc):
        return False


class _FakeDriver:
    def __init__(self):
        self.session_obj = _RecordingSession()

    def session(self, **kw):
        return _SessionCtx(self.session_obj)


def test_predicate_string_shapes():
    p = tenant_predicate("n")
    assert "$tenant IS NULL" in p
    assert "n.tenant_id" in p
    p2 = tenant_predicate("m")
    assert "m.tenant_id" in p2


def test_tenant_active_respects_mode():
    assert tenant_active("t1") is False          # TENANT_MODE defaults to off
    assert tenant_active(None) is False


def test_unscoped_retrieval_sends_null_tenant():
    driver = _FakeDriver()
    retrieve_subgraph(driver, "Does claim CLM-0003 have a fraud flag?")
    # every Cypher statement carries $tenant=NULL (predicate passes everything)
    assert driver.session_obj.calls
    for query, kwargs in driver.session_obj.calls:
        assert "tenant" in kwargs and kwargs["tenant"] is None
        assert "$tenant" in query


def test_scoped_retrieval_injects_tenant_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "TENANT_MODE", "column")
    driver = _FakeDriver()
    try:
        retrieve_subgraph(driver, "Does claim CLM-0003 have a fraud flag?",
                          tenant_id="bank-a")
    finally:
        monkeypatch.setattr(settings, "TENANT_MODE", "off")
    assert driver.session_obj.calls
    for query, kwargs in driver.session_obj.calls:
        assert kwargs.get("tenant") == "bank-a"
        assert "tenant_id" in query          # scoping predicate present
