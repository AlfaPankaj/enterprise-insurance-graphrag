"""Live check: answer modes end-to-end (Ollama is NOT running on this box,
so 'auto' must fall back to extractive; 'extractive' stays deterministic)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neo4j import GraphDatabase  # noqa: E402

from graphrag.answer_generator import ollama_available  # noqa: E402
from graphrag.config import settings  # noqa: E402
from graphrag.query_pipeline import run_query  # noqa: E402

print("ollama_available:", ollama_available())

driver = GraphDatabase.driver(settings.NEO4J_URI,
                              auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
for mode in ("extractive", "auto"):
    res = run_query(driver, "Does claim CLM-0003 have a fraud flag?",
                    token_budget=1280, reranker_mode="lexical", answer_mode=mode)
    tr = res["traversal"]
    print(f"[{mode}] answer_mode={res['answer_mode']} | answer_model={res.get('answer_model')}")
    print(f"  answer: {res['answer'][:90]}")
    print(f"  timings: {tr['timings_ms']}")
    assert res["answer_mode"] == "extractive", f"{mode} must fall back while Ollama is down"
    assert "answer_ms" in tr["timings_ms"] and "total_ms" in tr["timings_ms"]

# hard-require mode must fail loudly when Ollama is down
try:
    run_query(driver, "Does claim CLM-0003 have a fraud flag?", answer_mode="llm")
    print("[llm] ERROR: should have raised with Ollama down")
    raise SystemExit(1)
except RuntimeError as e:
    print(f"[llm] raised as required: {str(e)[:80]}")

# the extractive answer is unchanged from before the LLM feature
res = run_query(driver, "Does claim CLM-0003 have a fraud flag?",
                token_budget=1280, reranker_mode="lexical", answer_mode="extractive")
assert "FRD-CLM-0003" in res["answer"] and "MEDIUM" in res["answer"]
print("EXTRACTIVE ANSWER STILL CORRECT")

driver.close()
print("ANSWER MODES OK")
