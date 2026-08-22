#!/usr/bin/env python3
"""Answer-quality benchmark (v2 — WS-C, G15).

Runs the golden question set (``data/benchmarks/golden_questions.json``)
through the full pipeline against the live demo graph and scores every answer
with the evaluation engine:

* deterministic rules by default (zero API cost, reproducible)
* ``--llm-judge`` uses the provider-backed rubric judge when one is up
  (falls back to rules per question)

Outputs ``data/benchmarks/answer_quality_synthetic.json`` with per-question
scores + aggregate means, and fails (exit 1) when ``--min-faithfulness`` /
``--min-overall`` floors are not met — the CI quality gate.

Usage:
    python scripts/benchmark_answer_quality.py
    python scripts/benchmark_answer_quality.py --llm-judge --answer-mode auto
    python scripts/benchmark_answer_quality.py --min-faithfulness 0.9
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graphrag.config import settings  # noqa: E402
from graphrag.evals import evaluate_answer, evaluate_answer_hybrid  # noqa: E402
from graphrag.fraud_ground_truth import parse_fraud_verdict  # noqa: E402
from graphrag.query_pipeline import run_query  # noqa: E402

GOLDEN = PROJECT_ROOT / "data" / "benchmarks" / "golden_questions.json"
OUT = PROJECT_ROOT / "data" / "benchmarks" / "answer_quality_synthetic.json"


def _neo4j_available() -> bool:
    try:
        driver = GraphDatabase.driver(
            settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-mode", default="extractive",
                        choices=["extractive", "auto", "llm"])
    parser.add_argument("--llm-judge", action="store_true",
                        help="use the provider-backed rubric judge when available")
    parser.add_argument("--queries", type=int, default=0,
                        help="limit to the first N questions (0 = all)")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--min-faithfulness", type=float, default=0.0,
                        help="fail when mean faithfulness drops below this")
    parser.add_argument("--min-overall", type=float, default=0.0,
                        help="fail when mean overall score drops below this")
    args = parser.parse_args(argv)

    if not GOLDEN.exists():
        print(f"ERROR: {GOLDEN} missing — run scripts/build_golden_set.py first")
        return 2
    if not _neo4j_available():
        print("SKIP: Neo4j not reachable — seed the demo graph first "
              "(docker compose up -d neo4j && python scripts/seed_graph.py "
              "--reset --apply-schema)")
        return 0

    payload = json.loads(GOLDEN.read_text(encoding="utf-8"))
    questions = payload["questions"][:args.queries] if args.queries else payload["questions"]

    driver = GraphDatabase.driver(
        settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    rows: list[dict] = []
    engines: set[str] = set()
    t0 = time.perf_counter()
    try:
        for i, q in enumerate(questions, start=1):
            result = run_query(driver, q["query"], answer_mode=args.answer_mode,
                               include_context=True)
            ctx = result.get("context") or {}
            pruned = {
                "nodes": ctx.get("nodes", []),
                "text": ctx.get("text", ""),
                "kept": result["pruned"]["kept"],
                "node_count": result["pruned"]["node_count"],
            }
            answer = result["answer"]
            score = evaluate_answer_hybrid(
                q["query"], answer, pruned,
                answerable=q.get("answerable"), prefer_llm=args.llm_judge,
            )
            engines.add(score["engine"])
            row = {
                "query": q["query"],
                "category": q.get("category", "?"),
                "answerable": q.get("answerable", True),
                "expected_ids": q.get("expected_ids", []),
                "answer": answer,
                "answer_mode": result["answer_mode"],
                "retrieved_nodes": result["retrieval"]["node_count"],
                "cached": bool(result.get("cached")),
                **{k: score[k] for k in
                   ("faithfulness", "relevance", "groundedness", "refusal", "overall")},
                "engine": score["engine"],
                "issues": score["issues"],
            }
            # fraud ground-truth verdict check (fraud-category questions)
            if q.get("category") == "fraud" and q.get("ground_truth"):
                verdict = parse_fraud_verdict(answer)
                row["fraud_verdict_correct"] = verdict == \
                    ("YES" if q["ground_truth"] == "fraud" else "NO")
            rows.append(row)
            if i % 10 == 0:
                print(f"  {i}/{len(questions)} scored")
    finally:
        driver.close()

    means = {
        key: round(statistics.mean(r[key] for r in rows), 4)
        for key in ("faithfulness", "relevance", "groundedness", "refusal", "overall")
    }
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session": "synthetic",
        "answer_mode": args.answer_mode,
        "llm_judge": args.llm_judge,
        "engines_used": sorted(engines),
        "queries": len(rows),
        "means": means,
        "duration_s": round(time.perf_counter() - t0, 1),
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nanswer quality ({args.answer_mode}, engines={sorted(engines)}): "
          f"{len(rows)} questions")
    print("  means:", ", ".join(f"{k}={v}" for k, v in means.items()))
    print(f"  wrote {args.out.relative_to(PROJECT_ROOT)}")

    failed = False
    if means["faithfulness"] < args.min_faithfulness:
        print(f"FAIL: mean faithfulness {means['faithfulness']} < "
              f"{args.min_faithfulness}")
        failed = True
    if means["overall"] < args.min_overall:
        print(f"FAIL: mean overall {means['overall']} < {args.min_overall}")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
