"""Fraud-detection accuracy benchmark over real fraud labels.

Runs the FULL pipeline (retrieve -> re-rank -> prune -> answer) for every
fraud claim in the loaded dataset — all 923 of ``fraud_oracle`` by default —
plus a random sample of non-fraud claims, then compares the per-claim answer
verdict against the CSV ground truth and reports precision / recall / F1 /
accuracy.

Design notes (read before quoting the numbers):

  * The graph is built FROM the labels, so a perfect score is the expected
    outcome — this benchmark proves *end-to-end fidelity*: every fraud claim's
    FraudFlag survives retrieval + pruning (recall), no clean claim is ever
    falsely flagged (precision), at 15k+ claim scale. It is a regression test
    of the whole pipeline, not a learned-detector evaluation.
  * ``answer_mode`` is pinned to ``extractive`` — deterministic, ~ms per claim.
    An LLM run over 2,400 claims would take hours on a local CPU model; the
    verdict parser (``fraud_ground_truth.verdict_for_claim``) works on any
    answer text, so the same script works with ``--answer-mode auto`` on a
    small ``--limit`` if you want LLM numbers.
  * A claim is a fraud *prediction* only when the answer affirmatively flags
    it (verdict YES). Refusals / absent mentions ("not determinable", "no
    context") count as NOT fraud — the system must say so to get credit.
  * ``--negatives`` samples non-fraud claims (default 1,500) so precision has
    a denominator; ``--negatives 0`` disables them (recall + the 923 fraud
    TP/FN only). The audit trail is disabled during the run so ~2.4k
    benchmark queries don't flood ``data/audit_trail/``.

Usage:
    .venv/Scripts/python.exe scripts/benchmark_fraud_detection.py
    .venv/Scripts/python.exe scripts/benchmark_fraud_detection.py --dataset insurance_claims --negatives 500
    .venv/Scripts/python.exe scripts/benchmark_fraud_detection.py --limit 100 --negatives 100 --workers 4
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))          # for `scripts.*` imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # for `graphrag.*` imports

from graphrag.config import settings  # noqa: E402
from graphrag.custom_sessions import (  # noqa: E402
    _TRUE_VALUES as CUSTOM_TRUE_VALUES,
)
from graphrag.fraud_ground_truth import (  # noqa: E402
    load_ground_truth,
    verdict_for_claim,
)
from graphrag.query_pipeline import run_query  # noqa: E402

_CLAIM_COL_RE = re.compile(r"claim|fraud|amount|loss|incident|coverage")
_FRAUD_COL_RE = re.compile(r"fraud")
_ID_COL_RE = re.compile(r"(^|_)(id|number|no)$|_id$|^id$")

# per-dataset default (fraud_oracle / insurance_claims); --output overrides
OUT_TEMPLATE = PROJECT_ROOT / "data" / "benchmarks" / "fraud_detection_{dataset}.json"


# ---------------------------------------------------------------------------
# custom-session ground truth (same column heuristics as the CSV adapter)
# ---------------------------------------------------------------------------

def custom_ground_truth(csv_path: Path) -> dict[str, bool]:
    """{claim_id: is_fraud} for a user-uploaded CSV, mirroring
    ``graphrag.custom_sessions.adapt_csv_to_graph`` so the ids match the graph.

    Returns {} when the CSV has no fraud column or no claim-like columns.
    """
    import csv as _csv

    with csv_path.open(newline="", encoding="utf-8-sig",
                       errors="replace") as fh:
        reader = _csv.DictReader(fh)
        headers = [h.strip() for h in (reader.fieldnames or [])]
        rows = list(reader)
    if not headers:
        return {}
    lower = {h: h.lower() for h in headers}
    if not any(_CLAIM_COL_RE.search(lower[h]) for h in headers):
        return {}
    id_col = next((h for h in headers if _ID_COL_RE.search(lower[h])), None)
    fraud_col = next((h for h in headers if _FRAUD_COL_RE.search(lower[h])), None)
    if not fraud_col:
        return {}
    gt: dict[str, bool] = {}
    for i, row in enumerate(rows, start=1):
        cid = str(row.get(id_col) or "").strip() or f"CLM-{i:05d}"
        gt[cid] = str(row.get(fraud_col, "")).strip().lower() in CUSTOM_TRUE_VALUES
    return gt


# ---------------------------------------------------------------------------
# pure metric helpers (unit-tested)
# ---------------------------------------------------------------------------

def predict_fraud(verdict: str) -> bool:
    """A claim is predicted fraud only when the answer affirmatively flags it."""
    return verdict == "YES"


def sample_clean_claims(clean: list[str], negatives: int, rng: random.Random) -> list[str]:
    """Non-fraud claims to evaluate: ``negatives`` of them, 0 = none.

    ``--negatives 0`` must yield an empty list — falling through to "keep all
    14.5k clean claims" would silently turn a quick run into an hour.
    """
    if negatives == 0:
        return []
    return rng.sample(clean, min(negatives, len(clean)))


def compute_confusion(pairs: list[tuple[bool, str]]) -> tuple[int, int, int, int]:
    """(tp, fp, tn, fn) from [(ground_truth, verdict), ...]."""
    tp = fp = tn = fn = 0
    for ground, verdict in pairs:
        if predict_fraud(verdict):
            if ground:
                tp += 1
            else:
                fp += 1
        elif ground:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    """precision / recall / F1 / accuracy (0.0 on zero denominators)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def _evaluate_one(driver, cid: str, ground: bool, reranker_mode: str,
                  token_budget: int, answer_mode: str) -> dict:
    res = run_query(driver, f"Is claim {cid} flagged as fraud?",
                    token_budget=token_budget, reranker_mode=reranker_mode,
                    answer_mode=answer_mode)
    verdict = verdict_for_claim(res["answer"], cid)
    visited = set(res["traversal"]["nodes_visited"])
    kept = set(res["pruned"]["kept"])
    return {
        "claim_id": cid,
        "ground_truth": int(ground),
        "verdict": verdict,
        "predicted_fraud": int(predict_fraud(verdict)),
        "flag_in_subgraph": int(any(n.startswith("FRD-") for n in visited)),
        "flag_in_pruned": int(any(n.startswith("FRD-") for n in kept)),
        "tokens_before": res["tokens"]["before"],
        "tokens_after": res["tokens"]["after"],
        "latency_ms": res["execution_time_ms"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fraud-detection accuracy: precision/recall over real labels")
    p.add_argument("--dataset", default=None,
                   help="built-in dataset: fraud_oracle | insurance_claims | synthetic")
    p.add_argument("--custom-session", dest="custom_session", default=None,
                   help="benchmark a user-uploaded CSV session by name (ground "
                        "truth from its source CSV; output to "
                        "fraud_detection_<name>.json)")
    p.add_argument("--limit", type=int, default=None,
                   help="max fraud claims to evaluate (default: all)")
    p.add_argument("--negatives", type=int, default=1500,
                   help="sample of non-fraud claims to include for precision (0 = none)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--reranker-mode", default="lexical",
                   choices=["auto", "cross-encoder", "lexical"])
    # extractive = deterministic & fast; llm/auto are for small --limit runs
    # (each LLM answer takes seconds on a local CPU model)
    p.add_argument("--answer-mode", default="extractive",
                   choices=["extractive", "auto", "llm"])
    p.add_argument("--token-budget", type=int, default=settings.MAX_TOKENS)
    p.add_argument("--output", type=Path, default=None,
                   help="output path (default: data/benchmarks/fraud_detection_<dataset>.json)")
    p.add_argument("--uri", default=settings.NEO4J_URI)
    p.add_argument("--user", default=settings.NEO4J_USER)
    p.add_argument("--password", default=settings.NEO4J_PASSWORD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit is not None and args.limit < 1:
        print("ERROR: --limit must be a positive integer (or omit it for all)")
        return 2
    if args.negatives < 0:
        print("ERROR: --negatives must be >= 0")
        return 2
    if args.workers < 1:
        print("ERROR: --workers must be >= 1")
        return 2
    if not args.dataset and not args.custom_session:
        print("ERROR: pass --dataset or --custom-session")
        return 2

    if args.custom_session:
        from graphrag.custom_sessions import get_custom_session

        rec = get_custom_session(args.custom_session)
        if not rec:
            print(f"ERROR: no custom session named '{args.custom_session}'")
            return 1
        if rec["kind"] != "csv":
            print(f"ERROR: custom session '{args.custom_session}' is {rec['kind']}, "
                  "not csv — fraud labels only come from CSV uploads")
            return 1
        csv_path = PROJECT_ROOT / rec["sources"][0]
        gt = custom_ground_truth(csv_path)
        dataset = args.custom_session
    else:
        gt = load_ground_truth(args.dataset)
        dataset = args.dataset
    if not gt:
        print(f"ERROR: no fraud ground truth for '{dataset}' "
              "(custom CSV has no fraud column?)")
        return 1
    fraud = [cid for cid, v in gt.items() if v]
    clean = [cid for cid, v in gt.items() if not v]
    if args.limit:
        fraud = fraud[: args.limit]
    # benchmark queries shouldn't pollute the audit trail (they're thousands);
    # set inside main() so importing this module has no global side effects,
    # and restored in the run's finally so an in-process caller keeps audit
    # logging enabled afterwards
    audit_was_enabled = settings.AUDIT_ENABLED
    settings.AUDIT_ENABLED = False
    rng = random.Random(args.seed)
    clean = sample_clean_claims(clean, args.negatives, rng)
    targets = [(cid, True) for cid in fraud] + [(cid, False) for cid in clean]
    print(f"== {args.dataset}: {len(fraud)} fraud + {len(clean)} clean "
          f"claims through the pipeline ({args.workers} workers) ==")

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    results: list[dict] = []
    t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_evaluate_one, driver, cid, ground,
                            args.reranker_mode, args.token_budget,
                            args.answer_mode): cid
                for cid, ground in targets
            }
            done = 0
            for fut in as_completed(futures):
                results.append(fut.result())
                done += 1
                if done % 400 == 0:
                    print(f"  ...{done}/{len(targets)} claims done "
                          f"({time.perf_counter() - t0:.0f}s)")
    finally:
        driver.close()
        settings.AUDIT_ENABLED = audit_was_enabled

    elapsed = time.perf_counter() - t0
    results.sort(key=lambda r: r["claim_id"])
    pairs = [(bool(r["ground_truth"]), r["verdict"]) for r in results]
    tp, fp, tn, fn = compute_confusion(pairs)
    m = metrics(tp, fp, tn, fn)

    summary = {
        "dataset": dataset,
        "fraud_evaluated": len(fraud),
        "clean_evaluated": len(clean),
        "total": len(results),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "accuracy": m["accuracy"],
        "flag_survived_prune_pct": round(
            sum(r["flag_in_pruned"] for r in results if r["ground_truth"]) / max(len(fraud), 1) * 100, 2),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / max(len(results), 1), 2),
        "avg_tokens_before": round(sum(r["tokens_before"] for r in results) / max(len(results), 1), 1),
        "avg_tokens_after": round(sum(r["tokens_after"] for r in results) / max(len(results), 1), 1),
        "elapsed_s": round(elapsed, 1),
        "workers": args.workers,
        "reranker_mode": args.reranker_mode,
        "results": results,
    }
    out = args.output or Path(str(OUT_TEMPLATE).format(dataset=dataset))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"  {dataset}: {len(fraud)} fraud + {len(clean)} clean claims "
          f"in {elapsed:.1f}s ({summary['avg_latency_ms']}ms avg/claim)")
    print(f"  Confusion   : TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Precision   : {m['precision']*100:.2f}%   (TP / (TP+FP))")
    print(f"  Recall      : {m['recall']*100:.2f}%   (TP / (TP+FN)) — all {len(fraud)} fraud labels")
    print(f"  F1          : {m['f1']*100:.2f}%")
    print(f"  Accuracy    : {m['accuracy']*100:.2f}%")
    print(f"  FraudFlag survived pruning: {summary['flag_survived_prune_pct']}% of fraud claims")
    print(f"  Tokens      : avg {summary['avg_tokens_before']:.0f} -> {summary['avg_tokens_after']:.0f}")
    print(f"  Results     : {out}")
    # precision is undefined (not 0) when no clean claims were evaluated
    gate_ok = m["recall"] >= 0.95 and (len(clean) == 0 or m["precision"] >= 0.95)
    print("=" * 72)
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
