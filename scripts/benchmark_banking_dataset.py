#!/usr/bin/env python3
"""Banking ground-truth benchmark (v2 — WS-E, G18).

Builds natural-language queries from ``data/samples/banking.json`` (ground
truth by construction — same philosophy as the insurance benchmarks) and runs
them through the full pipeline against the loaded banking graph:

* account status id-lookups          (expected: the account node)
* disputes on an account             (expected: its dispute ids)
* who holds an account               (expected: the customer id)
* AML alert status id-lookups        (expected: the alert node)
* balance thresholds                 (expected: accounts ≥ threshold, capped ≤5)
* merchant paraphrase, no id quoted  (expected: the unique-merchant account)
* AML neighborhood (id-anchored)     (expected: customers behind an account's alerts)
* negative probes                    (expected: EMPTY + refusal — no hallucination)

Writes ``data/benchmarks/real_banking.json`` (the dashboard's Pipeline
Validation table reads it for the banking_demo row).

Usage:
    python scripts/ingest_banking_dataset.py --reset   # seed first
    python scripts/benchmark_banking_dataset.py
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graphrag.config import settings  # noqa: E402
from graphrag.query_pipeline import run_query  # noqa: E402

SAMPLE = PROJECT_ROOT / "data" / "samples" / "banking.json"
OUT = PROJECT_ROOT / "data" / "benchmarks" / "real_banking.json"


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


def build_queries(data: dict, seed: int = 42) -> list[dict]:
    """Ground-truth questions computed from the dataset itself."""
    rng = random.Random(seed)
    accounts = data["accounts"]
    by_id = {a["id"]: a for a in accounts}
    disputes_by_account: dict[str, list[str]] = {}
    for d in data["disputes"]:
        disputes_by_account.setdefault(d["account_id"], []).append(d["id"])
    alerts_by_account: dict[str, list[dict]] = {}
    for a in data["aml_alerts"]:
        alerts_by_account.setdefault(a["account_id"], []).append(a)

    queries: list[dict] = []

    # 1) account status id-lookups
    for acc in rng.sample(accounts, min(15, len(accounts))):
        queries.append({
            "category": "id-lookup",
            "query": f"What is the status of account {acc['id']}?",
            "expected": [acc["id"]],
        })

    # 2) disputes on an account
    disputed = [a for a in accounts if a["id"] in disputes_by_account]
    for acc in rng.sample(disputed, min(10, len(disputed))):
        queries.append({
            "category": "dispute",
            "query": f"Is there a dispute on account {acc['id']}?",
            "expected": disputes_by_account[acc["id"]],
        })

    # 3) who holds an account
    for acc in rng.sample(accounts, min(10, len(accounts))):
        queries.append({
            "category": "holder",
            "query": f"Who holds account {acc['id']}?",
            "expected": [acc["customer_id"]],
        })

    # 4) AML alert id-lookups
    for alert in rng.sample(data["aml_alerts"], min(10, len(data["aml_alerts"]))):
        queries.append({
            "category": "aml-id",
            "query": f"What is the status of AML alert {alert['id']}?",
            "expected": [alert["id"]],
        })

    # 5) balance threshold: pick a cutoff with ≤5 qualifying accounts
    balances = sorted(a["balance"] for a in accounts)
    threshold = balances[-min(5, len(balances))]
    over = [a["id"] for a in accounts if a["balance"] >= threshold]
    queries.append({
        "category": "threshold",
        "query": f"Which accounts have a balance over ${threshold:,.2f}?",
        "expected": over,
    })

    # 6) merchant paraphrase — no id quoted; the anchor merchant is unique
    #    in the generated data (exactly one account posted to it)
    queries.append({
        "category": "paraphrase",
        "query": (f"Which account posted a payment to "
                  f"{data['unique_merchant']}?"),
        "expected": [data["unique_merchant_account"]],
    })

    # 7) AML neighborhood, id-anchored ("money laundering" filters within the
    #    account's neighborhood) — expected: customers behind the alerts
    alerted = [a for a in accounts if a["id"] in alerts_by_account]
    for acc in rng.sample(alerted, min(5, len(alerted))):
        expected = sorted({a["customer_id"] for a in alerts_by_account[acc["id"]]})
        queries.append({
            "category": "aml-neighborhood",
            "query": (f"Which customers have been flagged for possible money "
                      f"laundering on account {acc['id']}?"),
            "expected": expected,
        })

    # 8) negative probes — non-existent ids must return EMPTY + refuse
    for q in ("What is the status of account ACC-99999?",
              "Is there a dispute on account ACC-99999?",
              "What is the status of AML alert AML-99999?"):
        queries.append({"category": "negative", "query": q, "expected": []})

    rng.shuffle(queries)
    return queries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--answer-mode", default="extractive",
                        choices=["extractive", "auto", "llm"])
    args = parser.parse_args(argv)

    if not SAMPLE.exists():
        print(f"ERROR: {SAMPLE} missing — run scripts/data_pipeline_banking.py first")
        return 2
    if not _neo4j_available():
        print("SKIP: Neo4j not reachable — seed with "
              "scripts/ingest_banking_dataset.py --reset first")
        return 0

    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
    queries = build_queries(data)

    driver = GraphDatabase.driver(
        settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    rows: list[dict] = []
    t0 = time.perf_counter()
    try:
        for i, q in enumerate(queries, start=1):
            result = run_query(driver, q["query"], answer_mode=args.answer_mode)
            retrieved = (set(result["retrieval"]["seeds"])
                         | set(result["pruned"]["kept"])
                         | set(result["pruned"]["dropped"]))
            expected = set(q["expected"])
            retrieval_ok = expected <= retrieved
            pruning_ok = expected <= set(result["pruned"]["kept"])
            if not expected:
                # negative probes: correct = nothing found AND refusal
                retrieval_ok = result["retrieval"]["node_count"] == 0
                pruning_ok = "no relevant context" in result["answer"].lower() \
                    or "not determinable" in result["answer"].lower()
            rows.append({
                "category": q["category"],
                "query": q["query"],
                "expected": sorted(expected),
                "retrieval_ok": bool(retrieval_ok),
                "pruning_ok": bool(pruning_ok),
                "retrieved_nodes": result["retrieval"]["node_count"],
                "kept_nodes": result["pruned"]["node_count"],
                "tokens": result["tokens"],
                "latency_ms": result["execution_time_ms"],
                "answer": result["answer"][:200],
            })
            if i % 10 == 0:
                print(f"  {i}/{len(queries)} scored")
    finally:
        driver.close()

    n = len(rows)
    retrieval_acc = 100.0 * sum(1 for r in rows if r["retrieval_ok"]) / n
    pruning_acc = 100.0 * sum(1 for r in rows if r["pruning_ok"]) / n
    report = {
        "queries": n,
        "retrieval_accuracy": round(retrieval_acc, 2),
        "pruning_accuracy": round(pruning_acc, 2),
        "avg_savings_pct": round(statistics.mean(
            r["tokens"]["savings_percent"] for r in rows), 2),
        "avg_latency_ms": round(statistics.mean(
            r["latency_ms"] for r in rows), 2),
        "answer_mode": args.answer_mode,
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nbanking benchmark: {n} queries, "
          f"retrieval {retrieval_acc:.1f}%, pruning {pruning_acc:.1f}%, "
          f"avg savings {report['avg_savings_pct']:.1f}%")
    print(f"  wrote {args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
