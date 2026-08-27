"""Option A — unified PostgreSQL data stack (Apache AGE + pgvector) tests.

Unit level: Cypher→SQL translation, agtype parsing, the neo4j-compatible
result objects, transaction semantics, config plumbing, pgvector SQL — all
against fakes (no PostgreSQL needed; psycopg2 stays an optional dep).

Integration level: one live parity test gated on ``TEST_AGE_DSN`` (see
docs/DEPLOYMENT_SCALE_OPTIONS.md) proving seed → keyword retrieval → CDC
update → cleanup against a real AGE instance.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest

from graphrag import postgres_backend as pb
from graphrag.config import settings
from graphrag.db import open_driver


# ---------------------------------------------------------------------------
# fakes — a psycopg2-shaped connection/cursor pair
# ---------------------------------------------------------------------------

class FakeCursor:
    def __init__(self, conn):
        self._conn = conn
        self.description = None
        self.executed: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        self._conn.statements.append((sql, params))
        if self._conn.fetchone_result is not None:
            self.description = True
        elif self._conn.fetchall_result is not None:
            self.description = True

    def fetchone(self):
        return self._conn.fetchone_result or (0,)

    def fetchall(self):
        return self._conn.fetchall_result or []


class FakeConn:
    """Records SQL; hands back canned rows."""

    def __init__(self, fetchall_result=None, fetchone_result=None):
        self.statements: list[tuple] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self.fetchall_result = fetchall_result
        self.fetchone_result = fetchone_result

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# translation: parameters
# ---------------------------------------------------------------------------

def test_inline_params_scalars_and_collections():
    q = ("MERGE (c:Claim {id: $cid}) SET c.amount = $amt, c.flag = $flag, "
         "c.note = $note, c.meta = $meta, c.x = $none")
    out = pb._inline_params(q, {
        "cid": "CLM-1", "amt": 1200.5, "flag": True, "note": 'say "hi"',
        "meta": ["a", 1, None], "none": None,
    })
    assert '{id: "CLM-1"}' in out
    assert "c.amount = 1200.5" in out
    assert "c.flag = true" in out
    assert '"say \\"hi\\""' in out          # JSON-escaped string
    assert "c.meta = [\"a\", 1, null]" in out
    assert "c.x = null" in out


def test_inline_params_unknown_placeholder_left_alone():
    assert pb._inline_params("MATCH (n {id: $missing})", {}) == "MATCH (n {id: $missing})"


def test_inline_params_nested_lists_for_unwind():
    out = pb._inline_params("UNWIND $pairs AS p RETURN p",
                            {"pairs": [["amount", "Claim"]]})
    assert '[["amount", "Claim"]]' in out


# ---------------------------------------------------------------------------
# translation: RETURN clause parsing
# ---------------------------------------------------------------------------

def test_return_specs_aliases_and_vars():
    specs = pb._return_specs(
        "MATCH (n) WITH n, labels(n) AS labels RETURN labels, n.id AS id LIMIT 5")
    assert specs == ["labels", "id"]


def test_return_specs_distinct_and_order_by():
    specs = pb._return_specs(
        "MATCH (n) WITH labels(n) AS l UNWIND l AS label "
        "RETURN DISTINCT label ORDER BY label")
    assert specs == ["label"]


def test_return_specs_count_and_parens():
    specs = pb._return_specs("MATCH (n) RETURN count(*) AS c ORDER BY c DESC")
    assert specs == ["c"]
    specs = pb._return_specs(
        "MATCH (n) RETURN labels(n) AS labels, n LIMIT 10")
    assert specs == ["labels", "n"]


def test_return_specs_plain_constant():
    assert pb._return_specs("RETURN 1") == ["1"]


def test_return_specs_none_for_pure_writes():
    assert pb._return_specs("MATCH (n:Dataset) SET n.rev = 1") is None


def test_return_specs_keyword_inside_string_ignored():
    specs = pb._return_specs(
        "MATCH (n) WHERE n.note = 'sort ORDER BY x' RETURN n.id AS id")
    assert specs == ["id"]


# ---------------------------------------------------------------------------
# translation: full SQL wrapping
# ---------------------------------------------------------------------------

def test_translate_wraps_cypher_in_age_select(monkeypatch):
    monkeypatch.setattr(settings, "AGE_GRAPH_NAME", "graphrag")
    sql, names = pb.translate(
        "MATCH (c:Claim {id: $cid}) RETURN c.amount AS amount",
        {"cid": "CLM-1"})
    assert sql.startswith("SELECT * FROM ag_catalog.cypher('graphrag', '")
    assert sql.endswith("') AS (c0 agtype)")
    assert '"CLM-1"' in sql                       # param inlined
    assert names == ["amount"]


def test_translate_write_without_return_gets_dummy_column(monkeypatch):
    monkeypatch.setattr(settings, "AGE_GRAPH_NAME", "graphrag")
    sql, names = pb.translate("MATCH (n) DETACH DELETE n")
    assert sql.endswith("AS (c agtype)")
    assert names == []


def test_translate_show_returns_none():
    assert pb.translate("SHOW INDEXES YIELD name") is None


def test_translate_escapes_single_quotes_in_cypher(monkeypatch):
    monkeypatch.setattr(settings, "AGE_GRAPH_NAME", "gr'aph")
    sql, _ = pb.translate("MATCH (n) WHERE n.x = $v RETURN n", {"v": "a'b"})
    assert "ag_catalog.cypher('gr''aph'," in sql


# ---------------------------------------------------------------------------
# agtype parsing
# ---------------------------------------------------------------------------

def test_parse_agtype_vertex_to_property_map():
    v = '{"id": 844424930131969, "label": "Claim", ' \
        '"properties": {"id": "CLM-0001", "amount": 5000.0}}::vertex'
    assert pb.parse_agtype(v) == {"id": "CLM-0001", "amount": 5000.0}


def test_parse_agtype_edge_to_property_map():
    e = '{"id": 1, "label": "HAS_CLAIM", "start_id": 1, "end_id": 2, ' \
        '"properties": {"since": 2024}}::edge'
    assert pb.parse_agtype(e) == {"since": 2024}


def test_parse_agtype_scalars_and_lists():
    assert pb.parse_agtype("42") == 42
    assert pb.parse_agtype('"text"') == "text"
    assert pb.parse_agtype("true") is True
    assert pb.parse_agtype("null") is None
    assert pb.parse_agtype('["a", 1]') == ["a", 1]
    assert pb.parse_agtype(None) is None


def test_parse_agtype_plain_map_passthrough():
    assert pb.parse_agtype('{"name": "synthetic", "rev": 3}') == \
        {"name": "synthetic", "rev": 3}


# ---------------------------------------------------------------------------
# result objects (the exact surface the codebase consumes)
# ---------------------------------------------------------------------------

def _result():
    rows = [
        ['{"id": 1, "label": "Claim", "properties": {"id": "CLM-1"}}::vertex', "7"],
        ['{"id": 2, "label": "Claim", "properties": {"id": "CLM-2"}}::vertex', "3"],
    ]
    return pb.AgeResult(["n", "hits"], rows)


def test_result_data_single_iteration():
    r = _result()
    data = r.data()
    assert data[0]["n"] == {"id": "CLM-1"}
    assert data[0]["hits"] == 7
    assert r.single()["hits"] == 7
    assert [rec["hits"] for rec in r] == [7, 3]
    assert r.keys() == ["n", "hits"]


def test_result_single_empty_and_consume():
    r = pb.AgeResult(["c"], [])
    assert r.single() is None
    assert r.data() == []
    assert len(r) == 0
    assert r.consume() is None


def test_record_zip_keys_values_pattern():
    rec = _result().single()
    job = dict(zip(rec.keys(), rec))
    assert job == {"n": {"id": "CLM-1"}, "hits": 7}


# ---------------------------------------------------------------------------
# session semantics against a fake connection
# ---------------------------------------------------------------------------

def _session(conn):
    return pb.AgeSession(conn, "graphrag", auto_create=True)


def test_session_run_commits_outside_tx(monkeypatch):
    monkeypatch.setattr(settings, "AGE_GRAPH_NAME", "graphrag")
    conn = FakeConn(fetchone_result=(1,), fetchall_result=[["0"]])
    s = _session(conn)
    res = s.run("MATCH (n:Claim) RETURN count(*) AS c")
    assert res.single()["c"] == 0                   # agtype "0" → int 0
    assert conn.commits == 1
    # every statement (catalog check + query) hit PostgreSQL
    assert any("ag_catalog.cypher" in sql for sql, _ in conn.statements)


def test_session_run_show_yields_empty_result():
    conn = FakeConn(fetchone_result=(1,))
    res = _session(conn).run("SHOW INDEXES YIELD name")
    assert res.data() == []
    assert res.keys() == ["name"]


def test_explicit_transaction_commits_once_and_rolls_back_on_error():
    conn = FakeConn(fetchone_result=(1,))
    s = _session(conn)
    with s.begin_transaction() as tx:
        tx.run("MERGE (n:Claim {id: $id})", id="CLM-9")
        tx.run("MATCH (d:Dataset) SET d.rev = coalesce(d.rev, 0) + 1")
    assert conn.commits == 1                        # single ACID transaction

    conn2 = FakeConn(fetchone_result=(1,))
    s2 = _session(conn2)
    with pytest.raises(RuntimeError):
        with s2.begin_transaction() as tx:
            tx.run("MERGE (n:Claim {id: $id})", id="CLM-9")
            raise RuntimeError("cdc write failed")
    assert conn2.rollbacks == 1
    assert conn2.commits == 0


def test_nested_transaction_rejected():
    conn = FakeConn(fetchone_result=(1,))
    s = _session(conn)
    with s.begin_transaction():
        with pytest.raises(RuntimeError):
            s.begin_transaction()


def test_session_close_rolls_back_open_tx():
    conn = FakeConn(fetchone_result=(1,))
    s = _session(conn)
    s.begin_transaction()
    s.close()
    assert conn.rollbacks == 1 and conn.closed


def test_missing_psycopg2_gives_actionable_error(monkeypatch):
    # a None entry in sys.modules makes `import psycopg2` raise ImportError,
    # reproducing the not-installed path regardless of the host environment
    monkeypatch.setitem(sys.modules, "psycopg2", None)
    with pytest.raises(RuntimeError, match="requirements-postgres"):
        pb._connect("dsn")


# ---------------------------------------------------------------------------
# driver factory (db.py)
# ---------------------------------------------------------------------------

def test_open_driver_default_is_neo4j():
    assert settings.GRAPH_BACKEND == "neo4j"        # default unchanged (v1/v2)
    d = open_driver()
    assert type(d).__module__.startswith("neo4j")
    d.close()


def test_open_driver_age_requires_dsn(monkeypatch):
    monkeypatch.setattr(settings, "GRAPH_BACKEND", "age")
    monkeypatch.setattr(settings, "POSTGRES_DSN", "")
    with pytest.raises(RuntimeError, match="POSTGRES_DSN"):
        open_driver()


def test_open_driver_age_returns_age_driver(monkeypatch):
    monkeypatch.setattr(settings, "GRAPH_BACKEND", "age")
    monkeypatch.setattr(settings, "POSTGRES_DSN", "postgresql://x/y")
    monkeypatch.setattr(settings, "AGE_GRAPH_NAME", "graphrag")
    d = open_driver()
    assert isinstance(d, pb.AgeDriver)
    d.close()


def test_open_driver_unknown_backend_rejected(monkeypatch):
    monkeypatch.setattr(settings, "GRAPH_BACKEND", "mysql")
    with pytest.raises(ValueError):
        open_driver()


# ---------------------------------------------------------------------------
# pgvector store
# ---------------------------------------------------------------------------

def _pg_store_conn(pgvector_present=True, create_ext_fails=False):
    conn = FakeConn(fetchone_result=(1 if pgvector_present else 0,))

    class C(FakeCursor):
        def execute(self, sql, params=None):
            if "CREATE EXTENSION" in sql and create_ext_fails:
                raise Exception("permission denied")
            super().execute(sql, params)

    conn.cursor = lambda: C(conn)
    return conn


def test_pgvector_store_requires_extension():
    conn = _pg_store_conn(pgvector_present=False, create_ext_fails=True)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(pb, "_connect", lambda dsn=None: conn)
    try:
        with pytest.raises(RuntimeError, match="pgvector"):
            pb.PgVectorStore(dataset="demo", dim=8)
    finally:
        monkey.undo()


def test_pgvector_store_search_sql_and_results():
    conn = _pg_store_conn()
    captured = {}

    class C(FakeCursor):
        def execute(self, sql, params=None):
            captured["sql"] = sql
            super().execute(sql, params)

        def fetchall(self):
            return [("CLM-1", "Claim", 0.93)]

    conn.cursor = lambda: C(conn)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(pb, "_connect", lambda dsn=None: conn)
    try:
        store = pb.PgVectorStore(dataset="demo", dim=4)
        hits = store.search([0.1, 0.2, 0.3, 0.4], k=5, exclude={"CLM-0"})
        assert hits == [("CLM-1", "Claim", 0.93)]
        sql = captured["sql"]
        assert "<=>" in sql and "ORDER BY" in sql and "LIMIT 5" in sql
        assert "NOT IN ('CLM-0')" in sql
        assert "dataset = 'demo'" in sql
    finally:
        monkey.undo()


def test_pgvector_store_unsafe_table_name_rejected():
    with pytest.raises(ValueError):
        pb.PgVectorStore(table="graphrag_vectors; DROP TABLE x")


# ---------------------------------------------------------------------------
# vector_store.py fallback guarantee
# ---------------------------------------------------------------------------

def test_build_vector_store_pgvector_failure_falls_back_to_memory(monkeypatch):
    import graphrag.vector_store as vs

    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "hash")
    monkeypatch.setattr(settings, "VECTOR_BACKEND", "pgvector")
    monkeypatch.setattr(vs, "_build_pgvector_store",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no pg")))
    vs.clear_vector_cache()

    def fake_scan(driver, limit):
        yield {"id": "CLM-1", "label": "Claim", "props": {"status": "OPEN"}}

    monkeypatch.setattr(vs, "_scan_nodes", fake_scan)

    class RevDriver:
        def session(self):
            return None

    # graph_revision is imported lazily inside build_vector_store
    monkeypatch.setattr("graphrag.cache.graph_revision", lambda d: ("demo", 1))
    store = vs.build_vector_store(RevDriver(), revision=("demo", 1), force=True)
    assert store is not None and len(store) == 1     # in-memory fallback
    assert isinstance(store, vs.VectorStore)
    vs.clear_vector_cache()


# ---------------------------------------------------------------------------
# live parity (opt-in: TEST_AGE_DSN must point at a real AGE instance)
# ---------------------------------------------------------------------------

LIVE_DSN = os.environ.get("TEST_AGE_DSN", "")
pytestmark_live = pytest.mark.skipif(
    not LIVE_DSN, reason="set TEST_AGE_DSN to run live AGE parity tests")


@pytestmark_live
def test_age_live_parity_seed_query_cleanup():
    """End-to-end against real AGE: write → read → transactional CDC → clean."""
    from graphrag.graph_retriever import _keyword_seeds

    driver = pb.AgeDriver(dsn=LIVE_DSN, graph="graphrag_test")
    with driver.session() as s:
        s.run("MATCH (n) WHERE n.id IN $ids DETACH DELETE n",
              ids=["PAR-CLM-1", "PAR-POL-1"])
        with s.begin_transaction() as tx:            # one ACID transaction
            tx.run("MERGE (c:Claim {id: $cid}) SET c.status = $st, c.amount = $amt",
                   cid="PAR-CLM-1", st="IN_REVIEW", amt=5000)
            tx.run("MERGE (p:Policy {id: $pid}) MERGE (c:Claim {id: $cid}) "
                   "MERGE (p)-[:HAS_CLAIM]->(c)",
                   pid="PAR-POL-1", cid="PAR-CLM-1")
        row = s.run("MATCH (c:Claim {id: $id}) RETURN c.status AS status",
                    id="PAR-CLM-1").single()
        assert row and row["status"] == "IN_REVIEW"
        cnt = s.run("MATCH (:Policy {id: $pid})-[:HAS_CLAIM]->(c) "
                    "RETURN count(c) AS c", id="PAR-POL-1").single()
        assert cnt and cnt["c"] == 1
        seeds = _keyword_seeds(s, ["in_review"], limit=5)
        assert any(x["id"] == "PAR-CLM-1" for x in seeds)
        s.run("MATCH (n) WHERE n.id IN $ids DETACH DELETE n",
              ids=["PAR-CLM-1", "PAR-POL-1"])
        gone = s.run("MATCH (c:Claim {id: $id}) RETURN count(c) AS c",
                     id="PAR-CLM-1").single()
        assert gone and gone["c"] == 0
    driver.close()
