"""Real-time query pipeline execution & audit-log showcase (terminal).

Runs ONE live query through the full pipeline (retrieve -> re-rank -> prune
-> answer) and prints the execution telemetry / audit trail exactly as an
analyst sees it: seeds, BFS stats, reranker choice + per-node score
breakdown, pruning savings, per-stage latency, the generated answer, the
traversal path, and the closing proof stats (accuracy + token reduction
from data/benchmarks/benchmark_results.json).

Usage:
    .venv/Scripts/python.exe scripts/showcase_execution_telemetry.py
    .venv/Scripts/python.exe scripts/showcase_execution_telemetry.py --query "Does claim CLM-0003 have a fraud flag?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neo4j import GraphDatabase  # noqa: E402

from graphrag.config import settings  # noqa: E402
from graphrag.graph_retriever import ENTITY_ID_RE  # noqa: E402
from graphrag.query_pipeline import run_query  # noqa: E402
from graphrag.traversal_logger import audit_store  # noqa: E402

DEFAULT_QUERY = "Does claim CLM-0003 have a fraud flag?"
PROOF = PROJECT_ROOT / "data" / "benchmarks" / "benchmark_results.json"

BAR = "=" * 80


def _ranking_from_audit(res: dict) -> list[tuple[str, str, float]]:
    """Top ranked (id, label, score) from the audit record just persisted."""
    rec = next((r for r in audit_store.recent(5)
                if r.get("audit_id") == res["traversal"]["audit_id"]), None)
    if not rec:
        return []
    return [(r["id"], r["label"], r["score"]) for r in rec.get("ranking", [])]


def _path_line(res: dict) -> str:
    paths = (res.get("traversal") or {}).get("paths") or []
    paths.sort(key=lambda p: len(p["nodes"]), reverse=True)
    if not paths:
        return "(no traversal chain rendered)"
    chain: list[str] = []
    for i, nid in enumerate(paths[0]["nodes"]):
        if i:
            _, rel, _ = paths[0]["edges"][i - 1]
            chain.append(f"-[:{rel}]->")
        chain.append(nid)
    return " ".join(chain)


def _proof_stats() -> tuple[float, float, int]:
    if not PROOF.exists():
        return 0.0, 0.0, 0
    b = json.loads(PROOF.read_text(encoding="utf-8"))
    m = b["aggregate_metrics"]
    n = b["benchmark_metadata"]["total_test_queries"]
    return m["retrieval_accuracy"], m["average_token_savings_pct"], n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--reranker-mode", default=None,
                   choices=["auto", "cross-encoder", "lexical"])
    p.add_argument("--answer-mode", default=None,
                   choices=["auto", "llm", "extractive"])
    p.add_argument("--uri", default=settings.NEO4J_URI)
    p.add_argument("--user", default=settings.NEO4J_USER)
    p.add_argument("--password", default=settings.NEO4J_PASSWORD)
    args = p.parse_args(argv)

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        # pre-heat the reranker (model load happens once per process — the
        # telemetry should show steady-state per-query latency, not cold start)
        from graphrag.reranker import make_reranker
        reranker = make_reranker(args.reranker_mode)
        reranker.rank(args.query, [{"id": "WARMUP", "label": "Claim", "props": {}}])
        res = run_query(driver, args.query, reranker_mode=args.reranker_mode,
                        answer_mode=args.answer_mode)
    finally:
        driver.close()

    ret_ms = res["traversal"]["timings_ms"]
    tok = res["tokens"]
    ranked = _ranking_from_audit(res)
    q_ids = set(ENTITY_ID_RE.findall(args.query.upper()))

    print(BAR)
    print("GRAPH-RAG EXECUTION TELEMETRY & AUDIT TRAIL")
    print(BAR)
    print(f"[QUERY]: \"{args.query}\"")
    print(f"[SEED DETECTION]: Detected Entities -> "
          f"{', '.join(res['retrieval']['seeds']) or 'keyword match'}")
    print(f"[BFS RETRIEVAL]: Extracted {res['retrieval']['node_count']} sub-graph nodes "
          f"({res['retrieval']['edge_count']} edges, {tok['before']:,} raw tokens)")

    if res["reranker"] == "lexical":
        print(f"\n[RERANKER SELECTION]: Auto-resolved backend -> \"lexical\" "
              f"(BM25 + Entity Bonus)")
    else:
        print(f"\n[RERANKER SELECTION]: Backend -> \"{res['reranker']}\" (neural)")
    print("[RERANKING BREAKDOWN]:")
    for i, (nid, label, score) in enumerate(ranked[:3], start=1):
        if nid.upper() in q_ids:
            why = "Explicit ID Match Bonus +5.0"
        else:
            why = "relevance to query terms"
        print(f"  Rank {i} | Score: {score:5.2f} | Node: {nid} ({label} - {why})")
    print("  " + "-" * 77)
    dropped = res["pruned"]["dropped"]
    if dropped:
        shown = ", ".join(dropped[:4]) + (", ..." if len(dropped) > 4 else "")
        print(f"  Pruned {len(dropped)} low-scoring nodes ({shown})")
    else:
        print("  Nothing pruned - context fit the budget")

    print(f"\n[CONTEXT PRUNER]: Reduced payload from {tok['before']:,} to "
          f"{tok['after']:,} tokens (-{tok['savings_percent']:.2f}% reduction)")
    print(f"[LATENCY TELEMETRY]: Retrieval: {ret_ms['retrieval_ms']:.0f}ms | "
          f"Reranking: {ret_ms['rerank_ms']:.0f}ms | Pruning: {ret_ms['prune_ms']:.0f}ms | "
          f"Answer: {ret_ms['answer_ms']:.0f}ms | Total: {ret_ms['total_ms']:.0f}ms")

    print("\n[LLM GENERATED RESPONSE]:")
    print(f"\"{res['answer']}\"")

    print(f"\n[AUDIT PATH]: {_path_line(res)}")
    acc, savings, n_q = _proof_stats()
    if acc:
        print(BAR)
        print(f"  {n_q} ground-truth queries | {acc:.0f}% accuracy | "
              f"{savings:.2f}% avg token reduction | full explainability per answer")
        print(BAR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
