"""Ground-truth benchmark against the REAL insurance datasets.

For each dataset, builds natural-language queries whose expected answer nodes
are computed from the source CSV itself (fraud ground truth, id lookups,
amount thresholds), then runs them through the full pipeline
(retrieve -> re-rank -> prune) and reports:

  * retrieval accuracy  — did every expected node enter the sub-graph?
  * pruning accuracy    — did every expected node survive the token budget?
  * token savings       — baseline vs optimized context
  * latency

Usage:
    .venv/Scripts/python.exe scripts/benchmark_real_dataset.py insurance_claims
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Real_datasets"
sys.path.insert(0, str(PROJECT_ROOT))          # for `scripts.*` imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # for `graphrag.*` imports

from src.graphrag.config import settings  # noqa: E402
from src.graphrag.graph_retriever import retrieve_subgraph  # noqa: E402
from src.graphrag.query_pipeline import run_query  # noqa: E402
from scripts.ingest_real_dataset import _num, policy_id  # noqa: E402


def _rows(name: str) -> list[dict]:
    with (DATA_DIR / f"{name}.csv").open(newline="", encoding="utf-8-sig",
                                         errors="replace") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# ground-truth query builders (expected ids computed from the CSV itself)
# ---------------------------------------------------------------------------

def _queries_fraud_oracle(rows: list[dict], limit: int | None = None) -> list[dict]:
    fraud_by_pol: dict[str, set[str]] = defaultdict(set)
    for i, r in enumerate(rows, start=1):
        if str(r.get("FraudFound_P", "0")).strip() == "1":
            fraud_by_pol[policy_id(r["PolicyNumber"], i)].add(f"CLM-{i:05d}")

    queries = [
        {"category": "id-lookup",
         "query": "What is the status of claim CLM-00001?",
         "expected": ["CLM-00001"]},
    ]
    # only add fraud-anchored queries when the dataset actually has fraud labels
    if len(fraud_by_pol) >= 1:
        pol1 = next(iter(fraud_by_pol))
        queries.append({"category": "fraud-list",
                        "query": f"Show me all fraud claims under policy {pol1}",
                        "expected": sorted(fraud_by_pol[pol1])})
    if len(fraud_by_pol) >= 2:
        pol2 = list(fraud_by_pol)[1]
        queries.append({"category": "fraud-check",
                        "query": "Which claims are flagged as fraud under policy "
                                  f"{pol2}?",
                        "expected": sorted(fraud_by_pol[pol2])})
    # a keyword-seeded query from the data (seeding is capped at 5, so the
    # ground truth is recall>=1 + precision rather than an exact id set)
    urban = [f"CLM-{i:05d}" for i, r in enumerate(rows, start=1)
             if str(r.get("AccidentArea", "")).strip().lower() == "urban"]
    if urban:
        queries.append({"category": "keyword",
                        "query": "Show me urban claims",
                        "expected": urban, "mode": "top-k"})
    return queries


def _queries_insurance_claims(rows: list[dict], limit: int | None = None) -> list[dict]:
    fraud_by_pol: dict[str, set[str]] = defaultdict(set)
    high_claims: set[str] = set()
    theft: set[str] = set()
    for i, r in enumerate(rows, start=1):
        pol = policy_id(r["policy_number"], i)
        clm = f"CLM-{i:05d}"
        if str(r.get("fraud_reported", "N")).strip().upper() == "Y":
            fraud_by_pol[pol].add(clm)
        if (_num(r.get("total_claim_amount")) or 0) >= 40000:
            high_claims.add(clm)
        if "theft" in str(r.get("incident_type", "")).lower():
            theft.add(clm)

    queries = [
        {"category": "id-lookup",
         "query": "What is the status of claim CLM-00001?",
         "expected": ["CLM-00001"]},
        {"category": "amount-threshold",
         "query": "Show me all claims over $40,000",
         "expected": sorted(high_claims), "mode": "top-k"},
        {"category": "keyword",
         "query": "Show me vehicle theft claims",
         "expected": sorted(theft), "mode": "top-k"},
    ]
    for pol, clms in list(fraud_by_pol.items())[:2]:
        if clms:
            queries.append({"category": "fraud-list",
                            "query": f"Show me all fraud claims under policy {pol}",
                            "expected": sorted(clms)})
    return queries


def _queries_insurance_dataset(rows: list[dict], limit: int | None = None) -> list[dict]:
    big: set[str] = set()
    doctor: set[str] = set()
    for i, r in enumerate(rows, start=1):
        if (_num(r.get("Claim_Amount")) or 0) >= 20000:
            big.add(f"CLM-{i:05d}")
        if "doctor" in str(r.get("Occupation", "")).lower():
            doctor.add(f"CLM-{i:05d}")
    queries = [
        {"category": "amount-threshold",
         "query": "Show me all claims over $20,000",
         "expected": sorted(big), "mode": "top-k"},
        {"category": "keyword",
         "query": "Show me all claims from doctors",
         "expected": sorted(doctor), "mode": "top-k"},
    ]
    return queries


def _queries_data_synthetic(rows: list[dict], limit: int | None = None) -> list[dict]:
    # mirror the ingest --limit so expected ids never reference rows the graph
    # doesn't have (None = the ingest default of all 53,503 rows)
    rows = rows if limit is None else rows[:limit]
    big_premium: set[str] = set()
    low_deduct: set[str] = set()
    for i, r in enumerate(rows, start=1):
        pol = policy_id(r["Customer ID"], i, pad=5)
        if (_num(r.get("Premium Amount")) or 0) >= 5000:
            big_premium.add(pol)
        if 0 < (_num(r.get("Deductible")) or 0) <= 1000:
            low_deduct.add(pol)
    queries = [
        {"category": "premium-threshold",
         "query": "Show me policies with premium over $5,000",
         "expected": sorted(big_premium), "mode": "top-k"},
        {"category": "deductible-threshold",
         "query": "Show me policies with deductible under $1,000",
         "expected": sorted(low_deduct), "mode": "top-k"},
    ]
    return queries


# ---------------------------------------------------------------------------
# 100-query expansion — distributed ground-truth coverage per dataset.
# ---------------------------------------------------------------------------
# The base builders above produce 2-4 queries per dataset (enough for a smoke
# run). For real validation we expand each dataset to a target (default 100)
# queries: id-lookups spread head/middle/tail, fraud-checks per policy,
# amount-thresholds at several cutoffs and keyword queries over value columns.
# Every expected id is computed from the CSV itself, so the ground truth stays
# exact for id/fraud queries and recall>=1 + precision for top-k ones.

# value columns worth keyword-seeding per dataset (must exist in the graph)
_KW_COL = {
    "fraud_oracle": "AccidentArea",
    "insurance_claims": "incident_type",
    "insurance_dataset": "Occupation",
    "data_synthetic": "Policy Type",
}
# numeric column + owning graph label for amount-threshold queries
_NUM_COL = {
    "fraud_oracle": ("VehiclePrice", "Claim"),
    "insurance_claims": ("total_claim_amount", "Claim"),
    "insurance_dataset": ("Claim_Amount", "Claim"),
    "data_synthetic": ("Premium Amount", "Policy"),
}


def _id_at(dataset: str, i: int, row: dict | None = None) -> str:
    """Graph node id for CSV row ``i`` (1-based)."""
    if dataset == "data_synthetic":
        # policy ids derive from the Customer ID (see policy_id), NOT the row
        # index — the benchmark must mirror the ingest exactly
        return policy_id((row or {}).get("Customer ID", ""), i, pad=5)
    return f"CLM-{i:05d}"


def _policy_at(dataset: str, row: dict, i: int) -> str:
    """The policy a row's claim belongs to (the fraud-check anchor)."""
    if dataset == "fraud_oracle":
        return policy_id(row.get("PolicyNumber", ""), i)
    return policy_id(row.get("policy_number", ""), i)


def _row_fraud(dataset: str, row: dict) -> bool:
    if dataset == "fraud_oracle":
        return str(row.get("FraudFound_P", "0")).strip() == "1"
    if dataset == "insurance_claims":
        return str(row.get("fraud_reported", "N")).strip().upper() == "Y"
    return False


def _sample_idxs(n_rows: int, n: int, rng=None) -> list[int]:
    """Deterministic head/middle/tail-spread row indexes for id-lookups.

    Returns ``n`` unique 1-based indexes: first/last row, plus evenly-spaced
    percentiles in between, so coverage spans the whole file (edge values
    included).
    """
    if n_rows == 0 or n <= 0:
        return []
    idxs: set[int] = {1, n_rows}
    for k in range(1, n - 1):
        idxs.add(max(1, min(n_rows, 1 + round(k * (n_rows - 1) / max(n - 1, 1)))))
    return sorted(idxs)


def _threshold_midpoint(vals: list[float], pct: float) -> float | None:
    """A threshold strictly between two distinct data values at percentile pct."""
    distinct = sorted(set(vals))
    if len(distinct) < 2:
        return None
    pos = min(int(pct * len(distinct)), len(distinct) - 1)
    if pos >= len(distinct) - 1:
        pos = len(distinct) - 2
    return (distinct[pos] + distinct[pos + 1]) / 2.0


def _expand_to_target(dataset: str, rows: list[dict],
                      target: int = 100) -> list[dict]:
    """Build ``target`` deterministic ground-truth queries for a real dataset.

    Fixed mix (deduped by category+query, then truncated to target):
      * ~40% id-lookups spread head/middle/tail (edge ids included),
      * ~25% fraud-checks (one per fraud-bearing policy),
      * ~20% amount-thresholds at several cutoffs,
      * ~15% keyword queries over the dataset's value column.
    """
    n = len(rows)
    queries: list[dict] = []
    seen: set[tuple] = set()

    def add(q: dict) -> None:
        key = (q["category"], q["query"])
        if key not in seen:
            seen.add(key)
            queries.append(q)

    # 1) id-lookups — ~55% of the target, spread across the file (edge ids
    # included, so the exact-match metric exercises boundary values too)
    n_id = max(1, int(target * 0.55))
    for i in _sample_idxs(n, n_id):
        eid = _id_at(dataset, i, rows[i - 1])
        if dataset == "data_synthetic":
            cust = str(rows[i - 1].get("Customer ID", "")).strip()
            ph = f"PH-{(cust if cust.isdigit() else i):>05}"
            add({"category": "id-lookup",
                 "query": f"Show me the coverage for policyholder {ph}",
                 "expected": [eid, f"COV-{(cust if cust.isdigit() else i):>05}"]})
        else:
            add({"category": "id-lookup",
                 "query": f"What is the status of claim {eid}?",
                 "expected": [eid]})

    # 2) fraud-checks — one per fraud-bearing policy (only labeled datasets)
    fraud_by_pol: dict[str, set[str]] = defaultdict(set)
    for i, r in enumerate(rows, start=1):
        if _row_fraud(dataset, r):
            fraud_by_pol[_policy_at(dataset, r, i)].add(_id_at(dataset, i))
    n_fraud = min(len(fraud_by_pol), max(0, int(target * 0.25)))
    for pol in list(fraud_by_pol)[:n_fraud]:
        clms = sorted(fraud_by_pol[pol])
        add({"category": "fraud-list",
             "query": f"Show me all fraud claims under policy {pol}",
             "expected": clms})

    # 3) amount-thresholds — several cutoffs over the numeric column
    num_col, label = _NUM_COL.get(dataset, (None, None))
    if num_col:
        vals = sorted(_num(r.get(num_col)) for r in rows
                      if _num(r.get(num_col)) is not None)
        if len(vals) >= 2:
            colname = num_col.replace("_", " ").title()
            noun = "policies" if label == "Policy" else "claims"
            for pct in (0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95, 0.05,
                        0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.15, 0.45):
                thr = _threshold_midpoint(vals, pct)
                if thr is None:
                    continue
                over = sorted(_id_at(dataset, i, r)
                              for i, r in enumerate(rows, start=1)
                              if (_num(r.get(num_col)) or 0) > thr)
                if over:
                    add({"category": "amount-threshold",
                         "query": f"Show me all {noun} with {colname} over {thr:g}",
                         "expected": over, "mode": "top-k"})

    # 4) keyword queries — over the dataset's value column
    kw_col = _KW_COL.get(dataset)
    if kw_col:
        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            v = str(r.get(kw_col, "")).strip()
            if v:
                counts[v.lower()] += 1
        for term, cnt in sorted(counts.items(), key=lambda t: -t[1])[:15]:
            if cnt < 3:
                continue
            expected = sorted(_id_at(dataset, i, r)
                              for i, r in enumerate(rows, start=1)
                              if str(r.get(kw_col, "")).strip().lower() == term)
            add({"category": "keyword",
                 "query": f"Show me all {_KW_NOUN.get(dataset, 'claims')} "
                          f"with {kw_col.lower()} {term}",
                 "expected": expected, "mode": "top-k"})

    # 5) fill to target — extra id-lookups at ever-denser spread positions
    # (dedup means each additional query is a genuinely new boundary id)
    extra = max(0, target - len(queries))
    for i in _sample_idxs(n, n_id + extra):
        if len(queries) >= target:
            break
        eid = _id_at(dataset, i, rows[i - 1])
        if dataset == "data_synthetic":
            cust = str(rows[i - 1].get("Customer ID", "")).strip()
            ph = f"PH-{(cust if cust.isdigit() else i):>05}"
            add({"category": "id-lookup",
                 "query": f"Show me the coverage for policyholder {ph}",
                 "expected": [eid, f"COV-{(cust if cust.isdigit() else i):>05}"]})
        else:
            add({"category": "id-lookup",
                 "query": f"What is the status of claim {eid}?",
                 "expected": [eid]})
    return queries[:target]


_KW_NOUN = {
    "fraud_oracle": "claims", "insurance_claims": "claims",
    "insurance_dataset": "claims", "data_synthetic": "policies",
}


def _queries_synthetic(rows: list[dict], limit: int | None = None) -> list[dict]:
    """Ground-truth queries for the PDF demo graph (data/samples/claims.json)."""
    fraud_by_pol: dict[str, set[str]] = defaultdict(set)
    big: set[str] = set()
    for i, r in enumerate(rows, start=1):
        cl = r.get("claim") or {}
        cid = cl.get("id") or r.get("doc_id") or f"CLM-{i:03d}"
        pol = cl.get("policy_id") or f"POL-{i:03d}"
        # fraud_flag is the FraudFlag node dict when present (None otherwise)
        flag = r.get("fraud_flag")
        if isinstance(flag, dict) or str(flag).strip().lower() in ("1", "true", "yes", "y"):
            fraud_by_pol[pol].add(cid)
        if (_num(cl.get("amount")) or 0) >= 20000:
            big.add(cid)
    queries = []
    for pol, clms in list(fraud_by_pol.items())[:25]:
        queries.append({"category": "fraud-list",
                        "query": f"Show me all fraud claims under policy {pol}",
                        "expected": sorted(clms)})
    if big:
        queries.append({"category": "amount-threshold",
                        "query": "Show me all claims over $20,000",
                        "expected": sorted(big), "mode": "top-k"})
    # fill with id-lookups spread across the demo graph
    n_rows = len(rows)
    for i in _sample_idxs(n_rows, max(0, 100 - len(queries)), None):
        cl = rows[i - 1].get("claim") or {}
        cid = cl.get("id") or rows[i - 1].get("doc_id") or f"CLM-{i:03d}"
        queries.append({"category": "id-lookup",
                        "query": f"What is the status of claim {cid}?",
                        "expected": [cid]})
    return queries[:100]


# --- generic builder for user-uploaded (custom) CSVs ----------------------
# Mirrors the column heuristics of graphrag.custom_sessions.adapt_csv_to_graph
# so the expected ids below match what the ingest actually built.

_CLAIM_COL_RE = re.compile(r"claim|fraud|amount|loss|incident|coverage")
_FRAUD_COL_RE = re.compile(r"fraud")
_ID_COL_RE = re.compile(r"(^|_)(id|number|no)$|_id$|^id$")
_TRUE_VALUES = {"1", "y", "yes", "true", "fraud", "fraudulent", "flagged"}


def _queries_custom_session(rows: list[dict], limit: int | None = None) -> list[dict]:
    """Ground-truth queries for an arbitrary user-uploaded CSV.

    Reads the same column heuristics as the custom-CSV ingest adapter:
    claim-like CSVs -> ``(:Claim)`` nodes, ``*_id`` columns -> node ids, a
    fraud column -> ``(:FraudFlag)`` + ``FRAUD_DETECTED`` edges. Builds:

      * an id-lookup (exact node id from the CSV),
      * a fraud query when the CSV has a fraud column (expected = flagged
        claim ids),
      * a numeric-threshold query over the first numeric column that has
        real variance (expected = ids above the 75th percentile).
    """
    if not rows:
        return []
    headers = list(rows[0].keys())
    lower = {h: h.lower() for h in headers}
    is_claims = any(_CLAIM_COL_RE.search(lower[h]) for h in headers)
    id_col = next((h for h in headers if _ID_COL_RE.search(lower[h])), None)
    fraud_col = next((h for h in headers if _FRAUD_COL_RE.search(lower[h])), None)

    prefix = "CLM" if is_claims else "REC"  # matches adapt_csv_to_graph ids
    ids = []
    flagged: list[str] = []
    for i, row in enumerate(rows, start=1):
        nid = str(row.get(id_col) or "").strip() or f"{prefix}-{i:05d}"
        ids.append(nid)
        if fraud_col and str(row.get(fraud_col, "")).strip().lower() in _TRUE_VALUES:
            flagged.append(nid)

    queries = []
    if is_claims and ids:
        queries.append({"category": "id-lookup",
                        "query": f"What is the status of claim {ids[0]}?",
                        "expected": [ids[0]]})
    elif ids:
        queries.append({"category": "id-lookup",
                        "query": f"What is the status of record {ids[0]}?",
                        "expected": [ids[0]]})
    if flagged:
        queries.append({"category": "fraud-list",
                        "query": "Which claims are flagged as fraud?",
                        "expected": sorted(flagged)})
    # numeric threshold over the first varying numeric column
    num_col = None
    for h in headers:
        if h == id_col or _FRAUD_COL_RE.search(lower[h]):
            continue
        vals = [_num(row.get(h)) for row in rows]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2 and len(set(vals)) > 1:
            num_col = h
            break
    if num_col and ids:
        raw = [(nid, _num(row.get(num_col)) or 0.0) for nid, row in zip(ids, rows)]
        vals = sorted(v for _, v in raw)
        # pick a threshold strictly BETWEEN data values: the retriever treats
        # "over" as >=, so a threshold that equals a data point (e.g. 9000 on a
        # row with amount 9000) would pull that row in off-target. The midpoint
        # of the value at the 75th percentile and the next higher distinct
        # value never coincides with a data point.
        idx = int(len(vals) * 0.75)
        if idx >= len(vals) - 1:
            idx = max(0, len(vals) - 2)
        distinct = sorted(set(vals))
        pos = distinct.index(vals[idx])
        if pos < len(distinct) - 1:
            thr = (distinct[pos] + distinct[pos + 1]) / 2.0
        else:
            thr = vals[idx] + 1.0  # all values equal — still a usable cut
        over = sorted(nid for nid, v in raw if v > thr)
        if over:
            noun = "claim" if is_claims else "record"
            colname = num_col.replace("_", " ")
            queries.append({"category": "amount-threshold",
                            "query": f"Show me all {noun}s with {colname} over {thr:g}",
                            "expected": over, "mode": "top-k"})
    return queries


_BUILDERS = {
    "fraud_oracle": _queries_fraud_oracle,
    "insurance_claims": _queries_insurance_claims,
    "insurance_dataset": _queries_insurance_dataset,
    "data_synthetic": _queries_data_synthetic,
    "synthetic": _queries_synthetic,
}


def build_custom_session_queries(name: str) -> list[dict]:
    """Ground-truth queries for a registered custom session (registry CSV)."""
    from graphrag.custom_sessions import get_custom_session

    rec = get_custom_session(name)
    if not rec:
        raise ValueError(f"no custom session named {name!r}")
    if rec["kind"] != "csv":
        raise ValueError(
            f"custom session {name!r} is {rec['kind']}, not csv — the generic "
            "benchmark supports CSV uploads (PDF uploads run the edge-case "
            "benchmark against the demo graph).")
    src = PROJECT_ROOT / rec["sources"][0]
    with src.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        rows = list(csv.DictReader(f))
    return _queries_custom_session(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark the pipeline on a real dataset")
    p.add_argument("dataset", choices=list(_BUILDERS), nargs="?",
                   help="built-in dataset to benchmark")
    p.add_argument("--token-budget", type=int, default=1280)
    p.add_argument("--queries", type=int, default=100,
                   help="expand ground-truth queries to this many per dataset "
                        "(default: 100; real datasets get a distributed mix)")
    p.add_argument("--limit", type=int, default=None,
                   help="match an ingest --limit for data_synthetic "
                        "(default: all rows)")
    p.add_argument("--output", type=Path)
    p.add_argument("--custom-session", dest="custom_session", default=None,
                   help="benchmark a user-uploaded CSV session by name "
                        "(reads its source from data/custom_sessions.json)")
    p.add_argument("--uri", default=settings.NEO4J_URI)
    p.add_argument("--user", default=settings.NEO4J_USER)
    p.add_argument("--password", default=settings.NEO4J_PASSWORD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit is not None and args.limit < 1:
        print("ERROR: --limit must be a positive integer (or omit it for all rows)")
        return 2
    if args.custom_session:
        dataset = args.custom_session
        rows = []
        queries = build_custom_session_queries(dataset)
    else:
        if not args.dataset:
            print("ERROR: pass a dataset name or --custom-session <name>")
            return 2
        dataset = args.dataset
        if dataset == "synthetic":
            import json as _json

            samples = PROJECT_ROOT / "data" / "samples" / "claims.json"
            raw = _json.loads(samples.read_text(encoding="utf-8"))
            rows = raw if isinstance(raw, list) else raw.get("claims", raw.get("data", []))
        else:
            rows = _rows(dataset)
        queries = _BUILDERS[dataset](rows, args.limit)
        if dataset in _KW_COL and args.queries > len(queries):
            queries = _expand_to_target(dataset, rows, args.queries)
    print(f"== {dataset}: {len(queries)} ground-truth queries ==")

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    results = []
    try:
        for q in queries:
            res = run_query(driver, q["query"], token_budget=args.token_budget,
                            reranker_mode="lexical", answer_mode="extractive")
            expected = set(q["expected"])
            mode = q.get("mode", "exact")
            sub = retrieve_subgraph(driver, q["query"], settings.MAX_HOPS)
            sub_ids = {n["id"] for n in sub["nodes"]}
            kept = set(res["pruned"]["kept"])
            if mode == "exact":
                retrieval_hit = expected <= sub_ids
                prune_hit = expected <= kept
                bad_seeds: set[str] = set()
            else:  # top-k: seeding is capped, so check recall>=1 + precision
                seed_ids = set(res["retrieval"]["seeds"])
                bad_seeds = {s for s in seed_ids if s.startswith(("CLM", "POL"))} - expected
                retrieval_hit = bool(expected & sub_ids) and not bad_seeds
                prune_hit = bool(expected & kept) and not bad_seeds
            expected_found = len(expected & sub_ids)
            results.append({
                "category": q["category"], "query": q["query"], "mode": mode,
                "expected": sorted(expected),
                "expected_found": expected_found, "expected_total": len(expected),
                "missing_from_graph": sorted(expected - sub_ids)[:5],
                "missing_after_prune": sorted(expected - kept)[:5],
                "bad_seeds": sorted(bad_seeds),
                "retrieval_hit": retrieval_hit,
                "prune_hit": prune_hit,
                "tokens_before": res["tokens"]["before"],
                "tokens_after": res["tokens"]["after"],
                "savings_percent": res["tokens"]["savings_percent"],
                "execution_time_ms": res["execution_time_ms"],
            })
            mark = "OK " if retrieval_hit and prune_hit else "FAIL"
            where = f" ({expected_found}/{len(expected)} found)" if mode == "top-k" else ""
            print(f"  [{mark}] {q['category']:<18} {q['query'][:46]:<46} "
                  f"{res['tokens']['before']:>5} -> {res['tokens']['after']:>4} tok "
                  f"({res['tokens']['savings_percent']:>5.1f}%) "
                  f"{res['execution_time_ms']:>6.1f}ms{where}")
            if not (retrieval_hit and prune_hit):
                if bad_seeds:
                    print(f"         off-target seeds: {sorted(bad_seeds)[:5]}")
                if not retrieval_hit:
                    print(f"         not found: {sorted(expected - sub_ids)[:5]}")
                if not prune_hit:
                    print(f"         dropped in prune: {sorted(expected - kept)[:5]}")
    finally:
        driver.close()

    n = len(results)
    ret_acc = sum(r["retrieval_hit"] for r in results)
    prn_acc = sum(r["prune_hit"] for r in results)
    savings = [r["savings_percent"] for r in results]
    summary = {
        "dataset": dataset,
        "queries": n,
        "retrieval_accuracy": round(ret_acc / n * 100, 2),
        "pruning_accuracy": round(prn_acc / n * 100, 2),
        "avg_savings_pct": round(sum(savings) / n, 2),
        "avg_latency_ms": round(sum(r["execution_time_ms"] for r in results) / n, 2),
        "results": results,
    }
    print("=" * 72)
    print(f"  {dataset}: retrieval {ret_acc}/{n} = {summary['retrieval_accuracy']}% | "
          f"pruning {prn_acc}/{n} = {summary['pruning_accuracy']}% | "
          f"avg savings {summary['avg_savings_pct']}% | "
          f"avg {summary['avg_latency_ms']}ms")
    # persist per-dataset by default so the dashboard can show all 4 files
    out = args.output or PROJECT_ROOT / "data" / "benchmarks" / f"real_{dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  written: {out}")
    print("=" * 72)
    return 0 if prn_acc == n else 1


if __name__ == "__main__":
    sys.exit(main())
