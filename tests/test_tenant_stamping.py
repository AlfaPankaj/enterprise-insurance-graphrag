"""v2 tenant stamping tests — ingest + CDC write paths (no Neo4j needed)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # scripts.* imports
sys.path.insert(0, str(ROOT / "src"))  # graphrag.* imports

import pytest

from graphrag.config import settings
from graphrag.graph_updater import update_graph_surgically
from scripts.seed_graph import load_nodes


# ---------------------------------------------------------------------------
# load_nodes (bulk ingest path: seed_graph / ingest_real_dataset / custom CSV)
# ---------------------------------------------------------------------------

class _RecordingRunner:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def run(self, query, **kwargs):
        self.calls.append((query, kwargs))

    def begin_transaction(self):
        return self  # seed() uses `with session.begin_transaction() as tx`

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_load_nodes_stamps_tenant_when_given():
    runner = _RecordingRunner()
    nodes = {"Claim": [{"id": "CLM-0001", "props": {"amount": 100.0}},
                       {"id": "CLM-0002", "props": {"amount": 200.0}}]}
    load_nodes(runner, nodes, tenant_id="bank-a")
    query, kwargs = runner.calls[0]
    assert kwargs["tenant"] == "bank-a"
    assert "coalesce(n.tenant_id, $tenant)" in query      # first-owner-wins


def test_load_nodes_without_tenant_keeps_v1_query():
    runner = _RecordingRunner()
    load_nodes(runner, {"Claim": [{"id": "CLM-0001", "props": {}}]})
    query, kwargs = runner.calls[0]
    assert "tenant" not in kwargs and "tenant_id" not in query


# ---------------------------------------------------------------------------
# update_graph_surgically (CDC upload path)
# ---------------------------------------------------------------------------

class _Cursor:
    def single(self):
        return None

    def data(self):
        return []


class _Tx:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def run(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return _Cursor()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Session:
    def __init__(self, tx):
        self._tx = tx

    def begin_transaction(self):
        return self._tx

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Driver:
    def __init__(self, tx):
        self._tx = tx

    def session(self, **kw):
        return _Session(self._tx)


_CHANGES = {"added": [{"label": "Claim", "id": "CLM-9",
                       "props": {"amount": 500.0, "status": "OPEN"}}],
            "modified": [], "deleted": []}


def test_cdc_stamps_tenant_when_column_mode(monkeypatch):
    monkeypatch.setattr(settings, "TENANT_MODE", "column")
    tx = _Tx()
    update_graph_surgically(_Driver(tx), "DOC-1", _CHANGES, tenant_id="bank-a")
    upserts = [c for c in tx.calls if "MERGE (n:Claim" in c[0]]
    assert upserts, "expected an upsert MERGE"
    query, kwargs = upserts[0]
    assert kwargs.get("tenant") == "bank-a"
    assert "coalesce(n.tenant_id, $tenant)" in query


def test_cdc_without_tenant_keeps_v1_upsert(monkeypatch):
    monkeypatch.setattr(settings, "TENANT_MODE", "column")
    tx = _Tx()
    update_graph_surgically(_Driver(tx), "DOC-1", _CHANGES, tenant_id=None)
    upserts = [c for c in tx.calls if "MERGE (n:Claim" in c[0]]
    assert upserts
    query, kwargs = upserts[0]
    assert "tenant" not in kwargs
    assert "tenant_id" not in query


def test_cdc_ignores_tenant_when_mode_off(monkeypatch):
    monkeypatch.setattr(settings, "TENANT_MODE", "off")
    tx = _Tx()
    update_graph_surgically(_Driver(tx), "DOC-1", _CHANGES, tenant_id="bank-a")
    upserts = [c for c in tx.calls if "MERGE (n:Claim" in c[0]]
    assert "tenant_id" not in upserts[0][0]


# ---------------------------------------------------------------------------
# sessions._seed_command passes --tenant to the standalone seed script
# ---------------------------------------------------------------------------

def test_seed_command_adds_tenant_flag_in_column_mode(monkeypatch):
    from graphrag import sessions as sessions_mod

    pdf_meta = sessions_mod.get_session_meta("pdf_demo")
    monkeypatch.setattr(settings, "TENANT_MODE", "column")
    monkeypatch.setattr(settings, "DEFAULT_TENANT", "org-x")
    cmd = sessions_mod._seed_command(pdf_meta)
    assert "--tenant" in cmd and "org-x" in cmd
    # off mode: no tenant flag (v1 command unchanged)
    monkeypatch.setattr(settings, "TENANT_MODE", "off")
    cmd = sessions_mod._seed_command(pdf_meta)
    assert "--tenant" not in cmd
