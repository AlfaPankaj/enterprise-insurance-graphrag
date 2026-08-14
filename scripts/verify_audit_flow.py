"""Live end-to-end check of Phase 4: query -> audit record -> exports."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neo4j import GraphDatabase  # noqa: E402

from graphrag.audit_reporter import save_exports  # noqa: E402
from graphrag.config import settings  # noqa: E402
from graphrag.query_pipeline import run_query  # noqa: E402
from graphrag.traversal_logger import audit_store  # noqa: E402

driver = GraphDatabase.driver(settings.NEO4J_URI,
                              auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))

QUERIES = [
    "Does claim CLM-0003 have a fraud flag?",
    "Which coverages apply to claim CLM-0106?",
    "Who investigates claim CLM-0009?",
    "Show me all claims over $100,000",
]
for q in QUERIES:
    res = run_query(driver, q, token_budget=1280, reranker_mode="lexical")
    tr = res["traversal"]
    print(f"{q!r}")
    print(f"  answer: {res['answer'][:80]}")
    print(f"  audit_id={tr['audit_id']} | visited={len(tr['nodes_visited'])} "
          f"edges={len(tr['edges_traversed'])} paths={len(tr['paths'])}")
    print(f"  tokens {res['tokens']['before']} -> {res['tokens']['after']} "
          f"({res['tokens']['savings_percent']}%) | timings {tr['timings_ms']}")
    assert tr["cypher"], "missing cypher"
    assert tr["nodes_visited"], "missing visited nodes"
    assert len(tr["timings_ms"]) == 4

# exports for the most recent record
record = audit_store.recent(1)[0]
out = save_exports(record)
print("exports:", out)
for kind, rel in out.items():
    p = ROOT / rel
    assert p.exists() and p.stat().st_size > 0, f"{kind} export missing"
    if kind == "pdf":
        assert p.read_bytes()[:5] == b"%PDF-"

# store file exists and API-shape checks
store_file = ROOT / settings.AUDIT_DIR / "audit_trail.jsonl"
assert store_file.exists()
with store_file.open(encoding="utf-8") as fh:
    lines = [l for l in fh if l.strip()]
assert len(lines) >= len(QUERIES), f"expected >= {len(QUERIES)} records, got {len(lines)}"
json.loads(lines[-1])  # last line is valid JSON

driver.close()
print(f"AUDIT FLOW OK — {len(lines)} records in audit_trail.jsonl")
