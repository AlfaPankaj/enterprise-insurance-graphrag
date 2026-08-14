"""Edge-case benchmark: 5 queries per real dataset across the file's range.

The standard benchmarks anchor queries on the FIRST rows of each CSV (e.g.
CLM-00001). This script instead samples **5 rows spread across each file** —
1 near the beginning, 2 from the middle, 2 near the end — builds a
ground-truth query for each (ids + thresholds computed from that exact CSV
row), runs it through the FULL pipeline (retrieve -> re-rank -> prune ->
answer), and reports per-query:

  * expected node retrieved? survived pruning?  (positional accuracy)
  * tokens before/after + savings %               (token edge cases)
  * latency                                       (boundary-vs-middle cost)
  * the actual answer text

Why "edge cases" matters: rows at the file's head/tail exercise id boundary
values (CLM-00001 vs CLM-15420), numeric extremes (highest/lowest amounts),
and any ordering artifacts in the data — the same positions a production
system would hit first when real data gets appended or shuffled.

Each dataset's queries run against its OWN freshly-ingested graph (the graph
holds one dataset at a time). Ingest is automatic; use ``--skip-ingest`` when
the right dataset is already loaded.

Positions are DETERMINISTIC percentiles (1 head / 2 middle / 2 tail), so runs
are reproducible; ``--positions`` overrides with explicit row indexes.

Usage:
    .venv/Scripts/python.exe scripts/benchmark_edge_cases.py
    .venv/Scripts/python.exe scripts/benchmark_edge_cases.py --dataset insurance_claims
    .venv/Scripts/python.exe scripts/benchmark_edge_cases.py --positions 1,400,800,1200,15420
    .venv/Scripts/python.exe scripts/benchmark_edge_cases.py --head-pcts 2 --middle-pcts 40 60 --tail-pcts 98
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Real_datasets"
sys.path.insert(0, str(PROJECT_ROOT))          # for `scripts.*` imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # for `graphrag.*` imports

from graphrag.config import settings  # noqa: E402
from graphrag.query_pipeline import run_query  # noqa: E402
from scripts.ingest_real_dataset import _num  # noqa: E402

DATASETS = ["fraud_oracle", "insurance_claims", "insurance_dataset", "data_synthetic"]
LABELED = {"fraud_oracle", "insurance_claims"}

# default spread: 1 head / 2 middle / 2 tail (percentile positions in the file)
DEFAULT_HEAD = [0.05]
DEFAULT_MIDDLE = [0.30, 0.55]
DEFAULT_TAIL = [0.80, 0.95]

# per-dataset fraud ground-truth column + the value that means "fraud" —
# must mirror scripts/ingest_real_dataset.py exactly (no ground-truth drift)
FRAUD_COL = {"fraud_oracle": "FraudFound_P", "insurance_claims": "fraud_reported"}
FRAUD_TRUE = {"fraud_oracle": "1", "insurance_claims": "Y"}

# percentile (of n_rows) beyond which a row counts as "tail" (for labeling)
_HEAD_PCT = 0.08
_TAIL_PCT = 0.92


# ---------------------------------------------------------------------------
# sampling + query building (pure, unit-tested)
# ---------------------------------------------------------------------------

def sample_positions(n_rows: int, head=None, middle=None, tail=None) -> list[int]:
    """1-based row indexes at the given percentiles, deduped and sorted.

    Positions clamp to valid indexes so small files still give head/middle/
    tail coverage.
    """
    pcts = (head or DEFAULT_HEAD) + (middle or DEFAULT_MIDDLE) + (tail or DEFAULT_TAIL)
    return sorted({max(1, min(n_rows, round(p * n_rows))) for p in pcts})


def build_query(dataset: str, row: dict, i: int) -> dict:
    """One ground-truth query anchored on CSV row ``i`` (1-based)."""
    cid = f"CLM-{i:05d}"
    if dataset == "fraud_oracle":
        pol = str(row.get("PolicyNumber", "")).strip() or i
        return {"query": f"Is claim {cid} flagged as fraud?",
                "expected": [cid], "fraud_label": _fraud(dataset, row)}
    if dataset == "insurance_claims":
        return {"query": f"Was claim {cid} reported as fraud?",
                "expected": [cid], "fraud_label": _fraud(dataset, row)}
    if dataset == "insurance_dataset":
        amount = _num(row.get("Claim_Amount")) or 0
        return {"query": f"What is the status of claim {cid}?",
                "expected": [cid]}
    # data_synthetic — anchor on the graph's actual node ids (a bare customer
    # number can't seed retrieval: the entity regex matches POL-/PH- ids, so
    # the query must name the node id just like the other datasets)
    cust = str(row.get("Customer ID", "")).strip()
    ph_id = f"PH-{(cust if cust.isdigit() else i):>05}"
    pol_id = f"POL-{(cust if cust.isdigit() else i):>05}"
    cov_id = f"COV-{(cust if cust.isdigit() else i):>05}"
    # the answer names the coverage node (the asked-about entity) — accept
    # either the policy or its coverage id
    return {"query": f"Show me the coverage for policyholder {ph_id}",
            "expected": [pol_id, cov_id]}


def _fraud(dataset: str, row: dict) -> bool:
    if dataset not in FRAUD_COL:
        return False
    return str(row.get(FRAUD_COL[dataset], "")).strip().upper() == FRAUD_TRUE[dataset]


def _rows(name: str) -> list[dict]:
    with (DATA_DIR / f"{name}.csv").open(newline="", encoding="utf-8-sig",
                                         errors="replace") as f:
        return list(csv.DictReader(f))


def _positions_for(dataset: str, n_rows: int, args: argparse.Namespace) -> list[int]:
    if args.positions:
        raw = [int(p) for p in args.positions.split(",")]
        bad = [p for p in raw if not 1 <= p <= n_rows]
        if bad:
            raise ValueError(
                f"--positions out of range for {dataset}: {bad} "
                f"(valid 1..{n_rows})"
            )
        return sorted(set(raw))
    return sample_positions(n_rows,
                            [p / 100.0 for p in args.head_pcts],
                            [p / 100.0 for p in args.middle_pcts],
                            [p / 100.0 for p in args.tail_pcts])


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def _row_position(i: int, n_rows: int) -> str:
    if i <= max(1, round(_HEAD_PCT * n_rows)):
        return "head"
    if i >= round(_TAIL_PCT * n_rows):
        return "tail"
    return "middle"


def run_one(driver, dataset: str, row: dict, i: int, n_rows: int,
            args: argparse.Namespace) -> dict:
    q = build_query(dataset, row, i)
    res = run_query(driver, q["query"], token_budget=args.token_budget,
                    reranker_mode=args.reranker_mode, answer_mode=args.answer_mode)
    expected = set(q["expected"])
    # full retrieved node set lives on the traversal lineage (the retrieval
    # summary only carries seeds/counts)
    sub_ids = set(res["traversal"]["nodes_visited"])
    kept = set(res["pruned"]["kept"])
    answer = res["answer"]

    if "fraud_label" in q:
        from graphrag.fraud_ground_truth import verdict_for_claim
        verdict = verdict_for_claim(answer, f"CLM-{i:05d}")
        if q["fraud_label"]:
            # the system must affirmatively flag real fraud
            answer_ok = verdict == "YES"
        else:
            # no-false-accusation rule: UNKNOWN (refusal) is as correct as a
            # clean NO — the answer must simply NOT claim fraud
            answer_ok = verdict in ("NO", "UNKNOWN")
    else:
        # id must survive into the final context AND be named in the answer
        verdict = "n/a"
        answer_ok = bool(expected & kept) and any(e in answer for e in expected)

    return {
        "dataset": dataset,
        "row": i,
        "position": _row_position(i, n_rows),
        "query": q["query"],
        "expected": sorted(expected),
        "expected_found": len(expected & sub_ids),
        "retrieved": bool(expected & sub_ids),
        "survived_prune": bool(expected & kept),
        "tokens_before": res["tokens"]["before"],
        "tokens_after": res["tokens"]["after"],
        "savings_percent": res["tokens"]["savings_percent"],
        "latency_ms": res["execution_time_ms"],
        "verdict": verdict,
        "answer_ok": bool(answer_ok),
        "answer": answer[:220],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    datasets = [args.dataset] if args.dataset else DATASETS

    if args.answer_mode == "auto":
        from graphrag.answer_generator import ollama_available
        if not ollama_available():
            print("NOTE: Ollama unreachable — answers will be extractive; fraud "
                  "verdicts may read UNKNOWN and be marked FAIL (that is a real "
                  "signal: true fraud must be affirmatively flagged).")

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    results: list[dict] = []
    try:
        for ds in datasets:
            rows = _rows(ds)
            positions = _positions_for(ds, len(rows), args)
            if not args.skip_ingest:
                from scripts.ingest_real_dataset import main as ingest_main
                print(f"\n--- ingesting {ds} ---")
                rc = ingest_main([ds, "--reset"])
                if rc != 0:
                    print(f"  !! ingest failed for {ds} (rc={rc})")
                    continue
            print(f"\n== {ds}: {len(rows)} rows, querying rows {positions} ==")
            for i in positions:
                r = run_one(driver, ds, rows[i - 1], i, len(rows), args)
                results.append(r)
                mark = "OK " if (r["retrieved"] and r["survived_prune"]
                                 and r["answer_ok"]) else "FAIL"
                print(f"  [{mark}] row {i:>6} ({r['position']:<6}) "
                      f"{r['query'][:50]:<50} "
                      f"{r['tokens_before']:>5}->{r['tokens_after']:>4} tok "
                      f"({r['savings_percent']:>5.1f}%) {r['latency_ms']:>6.1f}ms")
                print(f"      ANSWER: {r['answer'][:170]}")
                if not r["answer_ok"]:
                    print(f"      expected {r['expected']} · retrieved "
                          f"{int(r['retrieved'])} · pruned {int(r['survived_prune'])}")
                    print(f"      verdict {r['verdict']}")
    finally:
        driver.close()

    ok = sum(1 for r in results
             if r["retrieved"] and r["survived_prune"] and r["answer_ok"])
    summary = {
        "total_queries": len(results),
        "passed": ok,
        "retrieval_accuracy": round(ok / len(results) * 100, 1) if results else 0,
        "avg_savings_pct": round(sum(r["savings_percent"] for r in results)
                                 / len(results), 2) if results else 0,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results)
                                / len(results), 2) if results else 0,
        "results": results,
    }
    out = args.output or (PROJECT_ROOT / "data" / "benchmarks" / "edge_cases.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 72)
    print(f"  {ok}/{len(results)} edge-case queries passed "
          f"({summary['retrieval_accuracy']}%) · avg savings "
          f"{summary['avg_savings_pct']}% · avg {summary['avg_latency_ms']}ms")
    print(f"  written: {out}")
    return 0 if ok == len(results) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", choices=DATASETS, help="one dataset (default: all 4)")
    p.add_argument("--skip-ingest", action="store_true",
                   help="don't ingest -- run against whatever is loaded in Neo4j")
    p.add_argument("--head-pcts", type=int, nargs="+", default=[5],
                   help="percentile positions from the file's start (default: 5)")
    p.add_argument("--middle-pcts", type=int, nargs="+", default=[30, 55])
    p.add_argument("--tail-pcts", type=int, nargs="+", default=[80, 95])
    p.add_argument("--positions",
                   help="explicit 1-based row positions, comma-separated "
                        "(overrides the head/middle/tail percentiles)")
    p.add_argument("--token-budget", type=int, default=1280)
    p.add_argument("--reranker-mode", default="lexical",
                   choices=["auto", "cross-encoder", "lexical"])
    p.add_argument("--answer-mode", default="auto",
                   choices=["extractive", "auto", "llm"],
                   help="answer generation (default: auto = Ollama LLM with "
                        "extractive fallback; extractive = fast deterministic)")
    p.add_argument("--output", type=Path)
    p.add_argument("--uri", default=settings.NEO4J_URI)
    p.add_argument("--user", default=settings.NEO4J_USER)
    p.add_argument("--password", default=settings.NEO4J_PASSWORD)
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
