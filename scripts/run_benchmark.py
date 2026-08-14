"""Run the 20 benchmark queries through the token-optimized pipeline.

For each query: retrieve -> re-rank -> prune, then check that every expected
answer node survived into the pruned context (retrieval accuracy), and record
baseline vs optimized tokens + savings.

Writes data/benchmarks/benchmark_results.json and prints a summary table.

NOTE: `benchmark_results.json` is also the output of
``scripts/export_benchmark_proof.py`` (the consolidated proof file). Running
this script overwrites it with the per-query run summary; regenerate the
proof afterwards with ``scripts/export_benchmark_proof.py``.

Usage:
    .venv/Scripts/python.exe scripts/run_benchmark.py                  # auto mode
    .venv/Scripts/python.exe scripts/run_benchmark.py --reranker-mode lexical
    .venv/Scripts/python.exe scripts/run_benchmark.py --token-budget 1024
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
from graphrag.query_pipeline import run_query  # noqa: E402

QUERIES = PROJECT_ROOT / "data" / "benchmarks" / "benchmark_queries.json"
OUT = PROJECT_ROOT / "data" / "benchmarks" / "benchmark_results.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Phase 3 token-optimization benchmark")
    p.add_argument("--queries", type=Path, default=QUERIES)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--reranker-mode", default=settings.RERANKER_MODE,
                   choices=["auto", "cross-encoder", "lexical"])
    p.add_argument("--token-budget", type=int, default=settings.MAX_TOKENS)
    p.add_argument("--max-hops", type=int, default=settings.MAX_HOPS)
    # benchmark measures RETRIEVAL accuracy + token savings — answer generation
    # stays deterministic (extractive) so latency numbers are comparable
    p.add_argument("--answer-mode", default="extractive",
                   choices=["extractive", "auto", "llm"])
    p.add_argument("--uri", default=settings.NEO4J_URI)
    p.add_argument("--user", default=settings.NEO4J_USER)
    p.add_argument("--password", default=settings.NEO4J_PASSWORD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queries = json.loads(args.queries.read_text(encoding="utf-8"))

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    results = []
    correct = 0
    try:
        for q in queries:
            res = run_query(driver, q["query"], max_hops=args.max_hops,
                            token_budget=args.token_budget,
                            reranker_mode=args.reranker_mode,
                            answer_mode=args.answer_mode)
            kept = set(res["pruned"]["kept"])
            expected = set(q["expected"])
            hit = expected <= kept
            correct += int(hit)
            results.append({
                "id": q["id"], "category": q["category"], "query": q["query"],
                "reranker_used": res["reranker"],
                "expected": sorted(expected), "kept": res["pruned"]["kept"],
                "hit": hit, "missing": sorted(expected - kept),
                "nodes_before": res["retrieval"]["node_count"],
                "tokens_before": res["tokens"]["before"],
                "tokens_after": res["tokens"]["after"],
                "savings_percent": res["tokens"]["savings_percent"],
                "execution_time_ms": res["execution_time_ms"],
            })
            mark = "PASS" if hit else "FAIL"
            print(f"  [{mark}] {q['id']} {q['query'][:58]:<58} "
                  f"{res['tokens']['before']:>5} -> {res['tokens']['after']:>4} tok "
                  f"({res['tokens']['savings_percent']:>5.1f}%) "
                  f"{res['execution_time_ms']:>6.1f}ms")
    finally:
        driver.close()

    n = len(queries)
    savings = [r["savings_percent"] for r in results]
    summary = {
        "generated_at": None,  # filled below without importing datetime
        "reranker_mode": args.reranker_mode,
        "reranker_used": results[0]["reranker_used"] if results else args.reranker_mode,
        "token_budget": args.token_budget,
        "max_hops": args.max_hops,
        "total_queries": n,
        "correct": correct,
        "accuracy_pct": round(correct / n * 100, 2),
        "avg_savings_pct": round(sum(savings) / n, 2),
        "min_savings_pct": round(min(savings), 2),
        "avg_tokens_before": round(sum(r["tokens_before"] for r in results) / n, 1),
        "avg_tokens_after": round(sum(r["tokens_after"] for r in results) / n, 1),
        "avg_latency_ms": round(sum(r["execution_time_ms"] for r in results) / n, 2),
        "results": results,
    }
    import datetime

    summary["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"  Accuracy     : {correct}/{n} queries = {summary['accuracy_pct']}%")
    print(f"  Token savings: avg {summary['avg_savings_pct']}% "
          f"({summary['avg_tokens_before']:.0f} -> {summary['avg_tokens_after']:.0f} tokens)")
    print(f"  Avg latency  : {summary['avg_latency_ms']}ms  (reranker used: {summary['reranker_used']})")
    print(f"  Results      : {args.output}")
    print("=" * 72)
    return 0 if correct / n >= 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())
