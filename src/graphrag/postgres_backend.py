"""Option A — unified PostgreSQL data stack (Apache AGE + pgvector).

An **opt-in deployment backend** (see ``docs/DEPLOYMENT_SCALE_OPTIONS.md``):
``GRAPH_BACKEND=age`` routes every graph read/write through **Apache AGE**
(openCypher inside PostgreSQL) behind a neo4j-driver-compatible shim, and
``VECTOR_BACKEND=pgvector`` moves the semantic index into **pgvector**.
Defaults keep the Neo4j reference implementation exactly as-is — importing
this module changes nothing unless the settings say so.

Why a shim works: the whole codebase talks to the graph through a narrow
surface — ``driver.session()`` → ``session.run(cypher, **params)`` →
``.data()`` / ``.single()`` / iteration / ``.consume()`` — and every Cypher
statement is plain openCypher (no APOC, no ``CALL {}`` subqueries), which is
the subset AGE implements. ``SHOW INDEXES / SHOW CONSTRAINTS`` (Neo4j admin
syntax used only for seed verification prints) are answered with empty
results so seeding scripts run unchanged.

PostgreSQL access is **optional** — ``psycopg2`` lives in
``requirements-postgres.txt`` and is imported lazily with a clear error.

agtype notes
------------
* Parameters: AGE's ``cypher()`` has no session parameter binding, so
  ``$name`` placeholders are inlined client-side as agtype literals
  (strings JSON-escaped, lists/bools/None native) — the same approach the
  official AGE drivers take.
* Results: agtype values arrive as text (``{...}::vertex``); they are parsed
  back to Python. Vertices/edges become their **property maps** — exactly
  what every consumer in this codebase does with Neo4j nodes
  (``dict(node)``, ``node["id"]``).
* Writes without RETURN are wrapped with a dummy one-column definition
  (``AS (c agtype)``); AGE yields zero rows for them.
"""

from __future__ import annotations

import json
import logging
import re
import threading

from graphrag.config import settings

logger = logging.getLogger("graphrag.age")

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PARAM_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_ANNOTATION_RE = re.compile(r"::(vertex|edge|path|agtype)\b")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# keywords that terminate a RETURN clause at top level
_CLAUSE_END = {"ORDER", "SKIP", "LIMIT", "UNION", "CALL", "WITH"}


# ---------------------------------------------------------------------------
# psycopg2 (optional dependency)
# ---------------------------------------------------------------------------

def _psycopg2():
    """Import psycopg2 lazily with an actionable error message."""
    try:
        import psycopg2  # noqa: PLC0415 - deliberately lazy, optional dep
        return psycopg2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "GRAPH_BACKEND=age requires psycopg2. Install the optional "
            "dependency with:  pip install -r requirements-postgres.txt"
        ) from exc


def _sql_ident(name: str) -> str:
    """Quote an identifier for inline SQL (defense in depth)."""
    if not _IDENT_RE.match(name or ""):
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return name


def _sql_str(text: str) -> str:
    """Escape into a single-quoted SQL string literal."""
    return "'" + str(text).replace("'", "''") + "'"


def _connect(dsn: str | None = None):
    """Open a PostgreSQL connection with AGE loaded and on the search path."""
    psycopg2 = _psycopg2()
    conn = psycopg2.connect(dsn or settings.POSTGRES_DSN)
    with conn.cursor() as cur:
        cur.execute("LOAD 'age'")
        cur.execute('SET search_path = ag_catalog, "$user", public')
    conn.commit()
    return conn


def graph_exists(conn, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM ag_catalog.ag_graph WHERE name = %s", (name,))
        return cur.fetchone()[0] > 0


def ensure_graph(conn, name: str, create: bool = True) -> bool:
    """Create the AGE graph when missing. Returns True when it exists."""
    if graph_exists(conn, name):
        return True
    if not create:
        return False
    with conn.cursor() as cur:
        cur.execute("SELECT ag_catalog.create_graph(%s)", (_sql_str(name),))
    conn.commit()
    logger.info("created AGE graph %r", name)
    return True


# ---------------------------------------------------------------------------
# Cypher → SQL translation
# ---------------------------------------------------------------------------

def _inline_params(cypher: str, params: dict) -> str:
    """Replace ``$name`` placeholders with agtype literals."""
    params = params or {}

    def literal(value) -> str:
        if value is None:
            return "null"
        if value is True:
            return "true"
        if value is False:
            return "false"
        if isinstance(value, (int, float)):
            return json.dumps(value)
        if isinstance(value, str):
            return json.dumps(value)          # JSON string == agtype string
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(literal(v) for v in value) + "]"
        if isinstance(value, dict):
            return json.dumps({str(k): v for k, v in value.items()})
        raise TypeError(f"cannot inline {type(value).__name__} as agtype")

    def repl(m: re.Match) -> str:
        name = m.group(1)
        return literal(params[name]) if name in params else m.group(0)

    return _PARAM_RE.sub(repl, cypher)


def _scan_top_level(text: str):
    """Yield (index, char) for chars at bracket-depth 0, skipping strings."""
    depth = 0
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "'\"`":
            quote = ch
            i += 1
            while i < n:
                if text[i] == "\\" and quote != "`":
                    i += 2
                    continue
                if text[i] == quote:
                    if quote != "'" or i + 1 >= n or text[i + 1] != "'":
                        break
                    i += 2  # doubled '' inside a single-quoted string
                    continue
                i += 1
            i += 1
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0:
            yield i, ch
        i += 1


def _split_top_level(text: str) -> list[str]:
    """Split on commas that sit at bracket-depth 0 and outside strings."""
    parts, start = [], 0
    for i, ch in _scan_top_level(text):
        if ch == ",":
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def _find_last_return(cypher: str) -> int | None:
    """Index of the final top-level RETURN keyword, or None."""
    hits = [m.start() for m in re.finditer(r"\bRETURN\b", cypher, re.I)]
    top = {i for i, _ in _scan_top_level(cypher)}
    for start in reversed(hits):
        # keyword must start at top level and not be inside a larger word
        if start in top and (start == 0 or not cypher[start - 1].isalnum()):
            return start
    return None


def _display_name(expr: str) -> str:
    """Record key for one RETURN expression (alias > stripped expression)."""
    stripped = re.sub(r"\s+", " ", expr.strip())
    if stripped.upper().startswith("DISTINCT "):
        stripped = stripped[len("DISTINCT "):].strip()
    # top-level "AS alias" suffix wins
    words = list(_WORD_RE.finditer(stripped))
    for m in words:
        if m.group(0).upper() == "AS":
            before = stripped[: m.start()].rstrip()
            after = stripped[m.end():].strip()
            # ensure AS is top-level: only whitespace/brackets around it
            opens = sum(before.count(c) for c in "([{")
            closes = sum(before.count(c) for c in ")]}")
            if opens == closes and _IDENT_RE.match(after):
                return after
    return stripped


def _return_specs(cypher: str) -> list[str] | None:
    """Display names of the final RETURN's columns (None when no RETURN)."""
    pos = _find_last_return(cypher)
    if pos is None:
        return None
    m = re.match(r"RETURN\b", cypher[pos:], re.I)
    body = cypher[pos + m.end():]

    # the clause runs until a top-level ORDER/SKIP/LIMIT/UNION/WITH keyword
    cut = len(body)
    for i, ch in _scan_top_level(body):
        if not (ch.isalpha() or ch == "_"):
            continue
        if i > 0 and body[i - 1] not in " \t\n\r,(":  # word start only
            continue
        m2 = _WORD_RE.match(body, i)
        if m2 and m2.group(0).upper() in _CLAUSE_END:
            cut = i
            break

    return [_display_name(e) for e in _split_top_level(body[:cut])]


def translate(query: str, params: dict | None = None,
              graph: str | None = None) -> tuple[str, list[str]] | None:
    """Translate one driver call → ``(sql, display_names)``.

    Returns ``None`` for Neo4j-admin statements (``SHOW …``) which have no
    AGE equivalent — callers answer those with an empty result so shared
    scripts (seed verification) run unchanged.
    """
    q = query.strip()
    if q.upper().startswith("SHOW"):
        return None

    graph = graph or settings.AGE_GRAPH_NAME
    cypher = _inline_params(q, params or {})
    specs = _return_specs(cypher)

    graph_lit = _sql_str(graph)
    body_lit = "'" + cypher.replace("'", "''") + "'"
    if specs is None:  # pure write, no RETURN → dummy zero-row column
        sql = (f"SELECT * FROM ag_catalog.cypher({graph_lit}, {body_lit}) "
               "AS (c agtype)")
        return sql, []

    cols = ", ".join(f"c{i} agtype" for i in range(len(specs)))
    sql = (f"SELECT * FROM ag_catalog.cypher({graph_lit}, {body_lit}) "
           f"AS ({cols})")
    return sql, specs


# ---------------------------------------------------------------------------
# agtype → Python
# ---------------------------------------------------------------------------

def parse_agtype(value):
    """Parse an agtype text value (``{...}::vertex``) back to Python."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if not isinstance(value, str):
        return value  # e.g. already-decoded JSON from a future driver
    text = _ANNOTATION_RE.sub("", value).strip()
    if text == "":
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return value  # opaque scalar (rare) — hand back the raw text

    def convert(node):
        if isinstance(node, dict):
            # AGE vertex/edge envelope → plain property map (what every
            # consumer here expects: dict(node), node["id"])
            envelope_keys = {"id", "label", "start_id", "end_id", "properties"}
            if set(node) <= envelope_keys and "label" in node \
                    and "properties" in node:
                return convert(node.get("properties") or {})
            return {k: convert(v) for k, v in node.items()}
        if isinstance(node, list):
            return [convert(v) for v in node]
        return node

    return convert(parsed)


# ---------------------------------------------------------------------------
# neo4j-driver-compatible result objects
# ---------------------------------------------------------------------------

class AgeRecord(dict):
    """A result record. Iterating yields **values** (zip(keys, row) pattern)."""

    def __iter__(self):
        return iter(self.values())


class AgeResult:
    """Materialized result with the surface the codebase uses."""

    def __init__(self, keys: list[str], rows: list[list]):
        self._keys = list(keys)
        self._records = [AgeRecord(zip(self._keys, [parse_agtype(v) for v in row]))
                         for row in rows]

    def keys(self) -> list[str]:
        return list(self._keys)

    def data(self) -> list[dict]:
        return [dict(r) for r in self._records]

    def single(self) -> AgeRecord | None:
        return self._records[0] if self._records else None

    def consume(self):  # health probes only check that it doesn't raise
        return None

    def __iter__(self):
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)


class AgeTransaction:
    """Explicit write transaction (``with session.begin_transaction() as tx``).

    CDC updates get the Option-A headline benefit here: the entity diff plus
    the derived-edge writes plus the revision bump commit (or roll back) as
    **one PostgreSQL transaction**.
    """

    def __init__(self, session: "AgeSession"):
        self._session = session

    def run(self, query: str, **params) -> AgeResult:
        return self._session.run(query, **params)

    def __enter__(self) -> "AgeTransaction":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._session._end_transaction(rollback=exc_type is not None)
        return False


class AgeSession:
    """One PostgreSQL connection acting as a neo4j session."""

    def __init__(self, conn, graph: str, auto_create: bool = True):
        self._conn = conn
        self._graph = graph
        self._auto_create = auto_create
        self._in_tx = False
        self._closed = False

    def _prepare(self):
        if self._closed:
            raise RuntimeError("session is closed")
        ensure_graph(self._conn, self._graph, create=self._auto_create)

    def run(self, query: str, **params) -> AgeResult:
        self._prepare()
        translated = translate(query, params, self._graph)
        if translated is None:  # Neo4j SHOW … → empty answer (prints only)
            m = re.search(r"YIELD\s+(.+)$", query.strip(), re.I)
            cols = [c.strip() for c in m.group(1).split(",")] if m else []
            return AgeResult([c for c in cols if _IDENT_RE.match(c)], [])
        sql, names = translated
        with self._conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall() if cur.description else []
        if not self._in_tx:
            self._conn.commit()
        return AgeResult(names, [list(r) for r in rows])

    def begin_transaction(self) -> AgeTransaction:
        if self._in_tx:
            raise RuntimeError("nested transactions are not supported")
        self._prepare()
        self._in_tx = True
        return AgeTransaction(self)

    def _end_transaction(self, rollback: bool = False) -> None:
        if self._in_tx:
            self._in_tx = False
            if rollback:
                self._conn.rollback()
            else:
                self._conn.commit()

    def close(self) -> None:
        if not self._closed:
            try:
                if self._in_tx:
                    self._conn.rollback()
                self._conn.close()
            finally:
                self._closed = True

    def __enter__(self) -> "AgeSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


class AgeDriver:
    """neo4j-driver-shaped handle over PostgreSQL/AGE.

    One connection per session keeps things simple and thread-safe; for the
    request rates this system serves (rate-limited API, single Streamlit app)
    that is well within budget — see the deployment-options doc.
    """

    def __init__(self, dsn: str | None = None, graph: str | None = None,
                 auto_create: bool | None = None):
        self._dsn = dsn or settings.POSTGRES_DSN
        self._graph = graph or settings.AGE_GRAPH_NAME
        self._auto_create = (settings.AGE_AUTO_CREATE
                             if auto_create is None else auto_create)
        self._closed = False
        self._lock = threading.Lock()

    def session(self, **_) -> AgeSession:
        if self._closed:
            raise RuntimeError("driver is closed")
        return AgeSession(_connect(self._dsn), self._graph, self._auto_create)

    def execute(self, sql: str, params: tuple | None = None, fetch: bool = False):
        """Raw SQL escape hatch (migrations, checks). Never used by queries."""
        with self.session() as s:
            with s._conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall() if (fetch and cur.description) else None
            s._conn.commit()
            return rows

    def verify(self) -> dict:
        """Connectivity + capability report, no side effects (check_config)."""
        out: dict = {"postgres": False, "age": False,
                     "graph": False, "pgvector": False}
        conn = _connect(self._dsn)   # raises when PG/AGE unreachable
        try:
            out["postgres"] = True
            out["age"] = True        # LOAD 'age' succeeded inside _connect
            out["graph"] = graph_exists(conn, self._graph)
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
                out["pgvector"] = cur.fetchone()[0] > 0
        finally:
            conn.close()
        return out

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "AgeDriver":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


# ---------------------------------------------------------------------------
# pgvector-backed VectorStore (VECTOR_BACKEND=pgvector)
# ---------------------------------------------------------------------------

class PgVectorStore:
    """Same contract as ``vector_store.VectorStore`` (add/search/__len__),
    but the index lives in pgvector so it scales past RAM — the 50M-row
    scenario from the deployment-options doc."""

    def __init__(self, dsn: str | None = None, dataset: str = "",
                 dim: int = 384, table: str = "graphrag_vectors"):
        self._dsn = dsn or settings.POSTGRES_DSN
        self._dataset = dataset          # raw name (used as a bind parameter)
        self._dim = int(dim)
        self._table = _sql_ident(table)
        self._conn = _connect(self._dsn)
        with self._conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
            if cur.fetchone()[0] == 0:
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    self._conn.commit()
                except Exception:  # noqa: BLE001 - needs privileges
                    self._conn.rollback()
                    raise RuntimeError(
                        "VECTOR_BACKEND=pgvector requires the pgvector "
                        "extension (see docs/DEPLOYMENT_SCALE_OPTIONS.md); "
                        "the built-in in-memory store remains the fallback")
        with self._conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                "dataset text NOT NULL, node_id text NOT NULL, label text NOT NULL,"
                f"text text NOT NULL, embedding vector({self._dim}))")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {self._table}_emb_hnsw "
                        f"ON {self._table} USING hnsw (embedding)")
        self._conn.commit()

    def rebuild(self, entries: list[tuple[str, str, str, list[float]]]) -> None:
        """Replace this dataset's index in one transaction."""
        from psycopg2.extras import execute_values  # noqa: PLC0415
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table} WHERE dataset = %s",
                        (self._dataset,))
            execute_values(
                cur,
                f"INSERT INTO {self._table} (dataset, node_id, label, text, embedding)"
                " VALUES %s",
                [(self._dataset, nid, label, text, _vec_literal(vec))
                 for nid, label, text, vec in entries],
                template="(%s, %s, %s, %s, %s::vector)",
            )
        self._conn.commit()

    # -- VectorStore-compatible surface -----------------------------------

    def add(self, node_id: str, label: str, text: str, vector: list[float]) -> None:
        self.rebuild([(node_id, label, text, vector)])

    def __len__(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self._table} WHERE dataset = %s",
                        (self._dataset,))
            return cur.fetchone()[0]

    def search(self, query_vector: list[float], k: int = 5,
               exclude: set[str] | None = None) -> list[tuple[str, str, float]]:
        vec = _vec_literal(query_vector)
        sql = (f"SELECT node_id, label, 1 - (embedding <=> {vec}::vector) AS score "
               f"FROM {self._table} WHERE dataset = {_sql_str(self._dataset)}")
        if exclude:
            ids = ",".join(_sql_str(e) for e in exclude)
            sql += f" AND node_id NOT IN ({ids})"
        sql += f" ORDER BY embedding <=> {vec}::vector LIMIT {int(k)}"
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return [(r[0], r[1], float(r[2])) for r in cur.fetchall()]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass


def _vec_literal(vec: list[float]) -> str:
    return "'[" + ",".join(repr(float(x)) for x in vec) + "]'"
