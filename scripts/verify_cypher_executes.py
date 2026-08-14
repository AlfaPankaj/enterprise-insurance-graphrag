"""Verify the audit-record Cypher is executable on plain Neo4j (no APOC)."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neo4j import GraphDatabase  # noqa: E402

from graphrag.config import settings  # noqa: E402
from graphrag.query_pipeline import run_query  # noqa: E402
from graphrag.traversal_logger import audit_store  # noqa: E402

driver = GraphDatabase.driver(settings.NEO4J_URI,
                              auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))

res = run_query(driver, "Does claim CLM-0003 have a fraud flag?",
                token_budget=1280, reranker_mode="lexical")
cypher = res["traversal"]["cypher"]
assert "apoc" not in cypher, "audit cypher must not require APOC"
print("cypher (no apoc) OK")

# execute the variable-length expansion statement from the record against Neo4j
m = re.search(r"MATCH p = \(s0 \{id: \"[^\"]+\"\}\)-\[\*1\.\.\d+\]-\(n\)\n"
              r"RETURN DISTINCT nodes\(p\) AS visited, relationships\(p\) AS edges", cypher)
assert m, "expansion pattern not found in cypher"
stmt = m.group(0)
with driver.session() as session:
    rows = session.run(stmt).data()
print(f"executed expansion stmt -> {len(rows)} distinct paths returned")
assert rows, "expansion returned nothing"

# the concrete answer path must also run (chain extracted from real edges)
m2 = re.search(r"MATCH p = \(n0 \{id: \"[^\"]+\"\}\).*?RETURN p", cypher, re.S)
if m2:
    with driver.session() as session:
        rows2 = session.run(m2.group(0)).data()
    print(f"executed answer-path stmt -> {len(rows2)} path(s) returned")
    assert rows2, "answer path returned nothing"

driver.close()
print("CYPHER EXECUTES OK")
