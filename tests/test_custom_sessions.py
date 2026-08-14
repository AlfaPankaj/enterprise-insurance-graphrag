"""Unit tests for custom (user-uploaded) sessions (Phase 6).

Registry and validation tests use a tmp registry file so the real
``data/custom_sessions.json`` is never touched. The CSV adapter is pure (no
Neo4j); the graph-writing paths are covered by the live E2E flow instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag.custom_sessions import (adapt_csv_to_graph, add_custom_session,
                                      get_custom_session, list_custom_sessions,
                                      remove_custom_session,
                                      rename_custom_session,
                                      validate_session_name)
from graphrag.sessions import all_sessions, session_for_marker


@pytest.fixture(autouse=True)
def _tmp_registry(monkeypatch, tmp_path):
    """Point the registry at a tmp file for every test in this module."""
    reg = tmp_path / "custom_sessions.json"
    monkeypatch.setattr("graphrag.custom_sessions.REGISTRY_PATH", reg)
    return reg


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_validate_session_name_rules():
    with pytest.raises(ValueError, match="empty"):
        validate_session_name("   ")
    with pytest.raises(ValueError, match="1-48 characters"):
        validate_session_name("bad name!!")
    with pytest.raises(ValueError, match="built-in"):
        validate_session_name("fraud_oracle")
    with pytest.raises(ValueError, match="built-in"):
        validate_session_name("pdf_demo")
    assert validate_session_name("  my claims  ") == "my claims"


def test_validate_session_name_no_duplicate_custom(tmp_path):
    src = tmp_path / "claims.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    add_custom_session("my_claims", "csv", [str(src)])
    with pytest.raises(ValueError, match="already exists"):
        validate_session_name("my_claims")
    # renaming keeps the current name
    assert validate_session_name("my_claims", exclude="my_claims") == "my_claims"


# ---------------------------------------------------------------------------
# generic CSV adapter
# ---------------------------------------------------------------------------

def test_adapt_claims_csv_builds_claims_and_flags(tmp_path):
    p = tmp_path / "claims.csv"
    p.write_text("claim_id,amount,fraud\nCLM-1,1000,1\nCLM-2,500,0\nCLM-3,200,Yes\n",
                 encoding="utf-8")
    nodes, rels = adapt_csv_to_graph(p)
    assert set(nodes) == {"Claim", "FraudFlag"}
    assert len(nodes["Claim"]) == 3
    assert nodes["Claim"][0]["id"] == "CLM-1"
    assert nodes["Claim"][0]["props"]["amount"] == 1000.0
    assert nodes["Claim"][0]["props"]["fraud"] == 1.0
    # rows 1 and 3 are flagged (1 / Yes); row 2 (0) is not
    assert len(nodes["FraudFlag"]) == 2
    assert len(rels) == 2
    assert all(r[4] == "FRAUD_DETECTED" for r in rels)


def test_adapt_generic_csv_builds_records(tmp_path):
    p = tmp_path / "people.csv"
    p.write_text("name,age,city\nalice,30,NYC\nbob,25,LA\n", encoding="utf-8")
    nodes, rels = adapt_csv_to_graph(p)
    assert set(nodes) == {"Record"}
    assert len(nodes["Record"]) == 2
    assert nodes["Record"][0]["id"] == "REC-00001"
    assert nodes["Record"][0]["props"]["name"] == "alice"
    assert nodes["Record"][0]["props"]["age"] == 30.0
    assert rels == []


def test_adapt_csv_uses_id_column_when_present(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("customer_id,value\nabc,10\nxyz,20\n", encoding="utf-8")
    nodes, _ = adapt_csv_to_graph(p)
    assert [n["id"] for n in nodes["Record"]] == ["abc", "xyz"]


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def test_registry_add_get_rename_remove(tmp_path):
    src = tmp_path / "claims.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    add_custom_session("my_claims", "csv", [str(src)], note="2 rows")
    assert get_custom_session("my_claims")["kind"] == "csv"
    assert list_custom_sessions()[0]["name"] == "my_claims"

    assert rename_custom_session("my_claims", "renamed_claims") is not None
    assert get_custom_session("my_claims") is None
    assert get_custom_session("renamed_claims")["name"] == "renamed_claims"

    assert remove_custom_session("renamed_claims") is True
    assert list_custom_sessions() == []


def test_add_custom_session_requires_kind_and_source():
    with pytest.raises(ValueError, match="kind"):
        add_custom_session("x", "exe", [])
    with pytest.raises(ValueError, match="source"):
        add_custom_session("x", "csv", [])
    with pytest.raises(ValueError, match="missing"):
        add_custom_session("x", "csv", ["data/custom/nope.csv"])


# ---------------------------------------------------------------------------
# session integration
# ---------------------------------------------------------------------------

def test_custom_session_appears_in_all_sessions_and_marker(tmp_path):
    src = tmp_path / "claims.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    add_custom_session("my_claims", "csv", [str(src)])
    ids = [s["id"] for s in all_sessions()]
    assert "my_claims" in ids
    # a custom session stamps its own name as the Dataset marker
    assert session_for_marker("my_claims") == "my_claims"
    # built-in mapping is unaffected
    assert session_for_marker("fraud_oracle") == "fraud_oracle"
    assert session_for_marker("synthetic") == "pdf_demo"
