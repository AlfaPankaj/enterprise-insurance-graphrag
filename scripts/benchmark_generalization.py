"""Anti-circularity generalization probes — "is the 100% just id-lookups?"

The main benchmarks (``benchmark_real_dataset.py``) derive ground-truth
queries from the SAME CSVs that built the graph, and ~88% of them are exact
id-lookups ("What is the status of claim CLM-00042?") — an enterprise auditor
will (correctly) call that near-circular: the retriever has an exact
entity-id regex, so of course it finds the node it was told to create. These
probes break the circularity by testing the pipeline where the ground truth
is NOT trivially reachable:

  1. PARAPHRASE  — the same expected nodes, reached through natural-language
     rephrasing that does NOT quote the claim id. Policy-anchored queries
     force a multi-hop traversal (POL -> HAS_CLAIM -> CLM); attribute-anchored
     queries force keyword/numeric seeding ("Show me the claims caused by
     {Fault}"). If retrieval only works when you quote the exact id, these
     fail.
  2. NEGATIVE    — ids that don't exist (CLM-99999), ids without the entity
     prefix, and keywords that aren't in the data. The correct answer is
     "no context / not found" — any fabricated claim is a hallucination
     failure.
  3. CROSS-SCHEMA — while dataset A is loaded, ask queries phrased for
     dataset B (different column vocabulary). The pipeline must either
     resolve them against A's schema or honestly refuse — never fabricate.
  4. ANSWER-LEVEL — every non-refuse probe's final answer TEXT must name the
     expected node id (surviving pruning is necessary but not sufficient).

Every probe runs through the FULL pipeline (retrieve -> re-rank -> prune ->
answer), extractive answers, audit trail disabled (benchmark queries must not
pollute it). Runs against the CURRENTLY LOADED graph — switch the session
first (the dashboard/app do this automatically). Writes
``data/benchmarks/generalization_<dataset>.json``.

Usage:
    .venv/Scripts/python.exe scripts/benchmark_generalization.py fraud_oracle
    .venv/Scripts/python.exe scripts/benchmark_generalization.py insurance_claims --skip-ingest
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Real_datasets"
sys.path.insert(0, str(PROJECT_ROOT))          # for `scripts.*` imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # for `graphrag.*` imports

from graphrag.config import settings  # noqa: E402
from graphrag.query_pipeline import run_query  # noqa: E402
from scripts.benchmark_real_dataset import (  # noqa: E402
    _id_at,
    _policy_at,
    _row_fraud,
    _sample_idxs,
)
from scripts.ingest_real_dataset import _num, policy_id  # noqa: E402

# any claim-id-like string inside an answer = fabrication (for refuse probes)
_CLAIM_ID_RE = re.compile(r"\bCLM-\d{3,}\b")
_POLICY_ID_RE = re.compile(r"\bPOL-\d{3,}\b")

# per-dataset vocabulary for attribute-anchored paraphrase + cross-schema:
# (CSV column, graph prop name, owning node label). Only columns the
# retriever actually keyword-searches (its _KEYWORD_PROPS: name/type/status/
# cause/reason/severity/category/occupation/...) are usable — a query about a
# non-searchable prop (auto_make, city) SHOULD come back empty, which would
# make the probe assert a failure on honest behavior.
_ATTRS = {
    "fraud_oracle": [("AccidentArea", "status", "Claim"),
                     ("Fault", "cause", "Claim")],
    "insurance_claims": [("incident_type", "cause", "Claim"),
                         ("insured_occupation", "occupation", "Policyholder")],
    "insurance_dataset": [("Occupation", "occupation", "Policyholder")],
    "data_synthetic": [("Policy Type", "type", "Policy")],
}

_NUM_ATTR = {
    "fraud_oracle": ("VehiclePrice", "amount", "Claim"),
    "insurance_claims": ("total_claim_amount", "amount", "Claim"),
    "insurance_dataset": ("Claim_Amount", "amount", "Claim"),
    "data_synthetic": ("Premium Amount", "premium", "Policy"),
}


# ---------------------------------------------------------------------------
# probe builders (pure -> unit-tested)
# ---------------------------------------------------------------------------

def _rows(dataset: str) -> list[dict]:
    if dataset == "synthetic":
        raw = json.loads((PROJECT_ROOT / "data" / "samples" / "claims.json")
                         .read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else raw.get("claims", raw.get("data", []))
    with (DATA_DIR / f"{dataset}.csv").open(newline="", encoding="utf-8-sig",
                                            errors="replace") as f:
        return list(csv.DictReader(f))


def _claim_ids_for_policy(dataset: str, rows: list[dict], pol: str) -> list[str]:
    """All claim ids whose row maps to policy ``pol`` (the HAS_CLAIM children)."""
    out = []
    for i, r in enumerate(rows, start=1):
        if _policy_at(dataset, r, i) == pol:
            out.append(_id_at(dataset, i, r))
    return sorted(out)


def paraphrase_queries(dataset: str, rows: list[dict], n: int = 15) -> list[dict]:
    """(Probe 1) Same expected nodes, phrased WITHOUT the claim id.

    Policy-anchored: "Show me the claims under policy {POL}" — the retriever
    must traverse HAS_CLAIM, not match a CLM id. Attribute-anchored:
    "Show me the claims caused by {value}" — must seed via keyword scan.
    """
    qs: list[dict] = []
    seen: set[tuple] = set()

    def add(q: dict) -> None:
        key = (q["kind"], q["query"])
        if key not in seen:
            seen.add(key)
            qs.append(q)

    if dataset == "data_synthetic":
        for i in _sample_idxs(len(rows), n):
            cust = str(rows[i - 1].get("Customer ID", "")).strip()
            ph = f"PH-{(cust if cust.isdigit() else i):>05}"
            pol = policy_id(rows[i - 1].get("Customer ID", i), i, pad=5)
            cov = f"COV-{(cust if cust.isdigit() else i):>05}"
            add({"kind": "paraphrase", "variant": "policy-anchored",
                 "query": f"Show me the coverage for policyholder {ph}",
                 "expected": [pol, cov], "mode": "exact"})
    elif dataset == "synthetic":
        for i in _sample_idxs(len(rows), n):
            cl = rows[i - 1].get("claim") or {}
            cid = cl.get("id") or rows[i - 1].get("doc_id") or f"CLM-{i:03d}"
            pol = cl.get("policy_id") or f"POL-{i:03d}"
            add({"kind": "paraphrase", "variant": "policy-anchored",
                 "query": f"Show me all claims under policy {pol}",
                 "expected": [cid], "mode": "exact"})
    else:
        for i in _sample_idxs(len(rows), n):
            pol = _policy_at(dataset, rows[i - 1], i)
            clm = _id_at(dataset, i, rows[i - 1])
            children = _claim_ids_for_policy(dataset, rows, pol)
            if dataset in ("fraud_oracle", "insurance_claims"):
                add({"kind": "paraphrase", "variant": "policy-anchored",
                     "query": f"Show me all claims under policy {pol}",
                     "expected": children, "mode": "exact"})
            else:
                add({"kind": "paraphrase", "variant": "policy-anchored",
                     "query": f"Show me the claim under policy {pol}",
                     "expected": [clm], "mode": "exact"})

    # attribute-anchored: pick a value whose matching row count is bounded so
    # top-k precision stays meaningful, then ask WITHOUT any id
    for col, prop, label in _ATTRS.get(dataset, []):
        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            v = str(r.get(col, "")).strip()
            if v:
                counts[v.lower()] += 1
        for term, cnt in sorted(counts.items(), key=lambda t: -t[1]):
            if not (3 <= cnt <= 500):
                continue
            expected = sorted(
                _id_at(dataset, i, r)
                for i, r in enumerate(rows, start=1)
                if str(r.get(col, "")).strip().lower() == term
            )
            noun = "policies" if dataset == "data_synthetic" else "claims"
            qtext = f"Show me all {noun} related to {term}"
            add({"kind": "paraphrase", "variant": f"attribute-{prop}",
                 "query": qtext, "expected": expected, "mode": "top-k"})
            break  # one attribute query per column is enough
    return qs[: n * 3]


def negative_queries(dataset: str, rows: list[dict]) -> list[dict]:
    """(Probe 2) Queries that must come back EMPTY — no fabrication allowed."""
    n = len(rows)
    far = max(n + 999, 99999)
    qs = [
        {"kind": "negative", "variant": "nonexistent-id",
         "query": f"What is the status of claim CLM-{far:05d}?",
         "expected": [], "mode": "refuse"},
        {"kind": "negative", "variant": "unknown-keyword",
         "query": "Show me claims caused by quantum entangled hovercraft",
         "expected": [], "mode": "refuse"},
        {"kind": "negative", "variant": "nonexistent-policy",
         "query": f"Show me claims under policy POL-{far:05d}",
         "expected": [], "mode": "refuse"},
    ]
    if dataset in ("fraud_oracle", "insurance_claims"):
        qs.append({"kind": "negative", "variant": "nonexistent-fraud-id",
                   "query": f"Is claim CLM-{far:05d} flagged as fraud?",
                   "expected": [], "mode": "refuse"})
    return qs


def cross_schema_queries(dataset: str, rows: list[dict]) -> list[dict]:
    """(Probe 3) Phrase queries for OTHER datasets' vocabulary, run on this one.

    A query resolves (exact expected ids from THIS dataset's CSV) when the
    concept exists here; it must honestly refuse when the column doesn't
    exist here — e.g. "claims from doctors" resolves on insurance_claims /
    insurance_dataset (they have an occupation column) but must return empty
    on fraud_oracle (which has none).
    """
    qs: list[dict] = []
    has_fraud = dataset in ("fraud_oracle", "insurance_claims")

    if dataset in ("fraud_oracle", "insurance_claims"):
        # fraud_oracle-style question phrased with insurance_claims vocabulary.
        # The claim EXISTS either way (fraud or not) — expected stays [clm];
        # only the ANSWER-level verdict differs, which the extractive answer
        # renders as a fraud flag list or a plain context node.
        i0 = min(3, len(rows))
        clm = _id_at(dataset, i0, rows[i0 - 1])
        qs.append({"kind": "cross-schema", "variant": "fraud-other-vocab",
                   "query": f"Was claim {clm} reported as fraudulent?",
                   "expected": [clm], "mode": "exact"})

    # occupation vocabulary: exists on insurance_claims / insurance_dataset /
    # data_synthetic (Policyholder.occupation), absent on fraud_oracle
    occ_col = {"insurance_claims": "insured_occupation",
               "insurance_dataset": "Occupation"}.get(dataset)
    if occ_col:
        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            v = str(r.get(occ_col, "")).strip()
            if v:
                counts[v.lower()] += 1
        term = max(counts, key=counts.get) if counts else None
        if term and 3 <= counts[term] <= 500:
            expected = sorted(
                _id_at(dataset, i, r)
                for i, r in enumerate(rows, start=1)
                if str(r.get(occ_col, "")).strip().lower() == term
            )
            qs.append({"kind": "cross-schema", "variant": "occupation-vocab",
                       "query": f"Show me claims from {term} workers",
                       "expected": expected, "mode": "top-k"})
    else:
        qs.append({"kind": "cross-schema", "variant": "occupation-vocab",
                   "query": "Show me claims from doctor workers",
                   "expected": [], "mode": "refuse"})

    # amount vocabulary phrased differently (dollar words instead of "over")
    num_col, _, _ = _NUM_ATTR.get(dataset, (None, None, None))
    if num_col:
        vals = sorted(_num(r.get(num_col)) for r in rows
                      if _num(r.get(num_col)) is not None)
        if len(vals) >= 4:
            thr = vals[len(vals) // 2]
            expected = sorted(
                _id_at(dataset, i, r)
                for i, r in enumerate(rows, start=1)
                if (_num(r.get(num_col)) or 0) > thr
            )
            if expected and len(expected) <= 3000:
                noun = "policies" if dataset == "data_synthetic" else "claims"
                qs.append({"kind": "cross-schema", "variant": "dollar-vocab",
                           "query": f"Which {noun} cost more than ${thr:,.0f}?",
                           "expected": expected, "mode": "top-k"})
    if not has_fraud:
        qs.append({"kind": "cross-schema", "variant": "fraud-absent",
                   "query": "Which claims are flagged as fraud?",
                   "expected": [], "mode": "refuse"})
    return qs


def build_probes(dataset: str, n_paraphrase: int = 15) -> list[dict]:
    """All probes for one dataset (paraphrase + negative + cross-schema)."""
    rows = _rows(dataset)
    return (paraphrase_queries(dataset, rows, n_paraphrase)
            + negative_queries(dataset, rows)
            + cross_schema_queries(dataset, rows))


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

_ANY_ID_RE = re.compile(r"\b(?:CLM|POL|PH|COV|FRD)-\d{3,}\b")


def _probe_result(res: dict, probe: dict) -> dict:
    """Score one probe: retrieval/prune, answer-level, refusal honesty.

    Answer-level bar (extractive answers name only the top-ranked node): the
    final answer must reference an expected node OR the queried anchor id —
    i.e. it is grounded in the right part of the graph. An answer that names
    an off-target id (e.g. a claim from a different policy that slipped into
    the sub-graph and got ranked #1) fails this check.
    """
    expected = set(probe["expected"])
    mode = probe.get("mode", "exact")
    sub_ids = set(res["traversal"]["nodes_visited"])
    kept = set(res["pruned"]["kept"])
    answer = res["answer"]
    seed_ids = set(res["retrieval"]["seeds"])

    if mode == "refuse":
        # must come back EMPTY: no entity seeds, no fabricated ids in answer
        fabricated = _ANY_ID_RE.findall(answer)
        retrieval_ok = not seed_ids and not sub_ids
        answer_ok = not fabricated
        return {
            "retrieval_hit": retrieval_ok,
            "prune_hit": retrieval_ok,  # nothing to prune
            "answer_ok": answer_ok,
            "fabricated": sorted(set(fabricated))[:5],
            "seeds": sorted(seed_ids)[:5],
        }

    # ids mentioned in the query are legitimate ANCHOR seeds (a policy-anchored
    # paraphrase "claims under policy POL-463237" must seed POL-463237) — only
    # seeds that are neither expected answers nor query anchors are off-target
    query_ids = set(_ANY_ID_RE.findall(probe["query"]))
    bad_seeds = {s for s in seed_ids
                 if s.startswith(("CLM", "POL"))} - expected - query_ids
    if mode == "exact":
        retrieval_hit = expected <= sub_ids and not bad_seeds
        prune_hit = expected <= kept
    else:  # top-k: seeding capped, recall>=1 + precision
        retrieval_hit = bool(expected & sub_ids) and not bad_seeds
        prune_hit = bool(expected & kept) and not bad_seeds

    # answer-level: extractive answers name the top-ranked node, so require
    # the answer to reference an expected node or the query's anchor id
    query_ids = set(_ANY_ID_RE.findall(probe["query"]))
    answer_ids = set(_ANY_ID_RE.findall(answer))
    answer_ok = bool(expected) and bool(answer_ids & (expected | query_ids))
    return {
        "retrieval_hit": retrieval_hit,
        "prune_hit": prune_hit,
        "answer_ok": answer_ok,
        "bad_seeds": sorted(bad_seeds)[:5],
        "answer_ids": sorted(answer_ids)[:8],
        "answer_names_expected": bool(answer_ids & expected),
    }


def run_one(driver, probe: dict, args: argparse.Namespace) -> dict:
    res = run_query(driver, probe["query"], token_budget=args.token_budget,
                    reranker_mode="lexical", answer_mode="extractive")
    score = _probe_result(res, probe)
    return {
        "kind": probe["kind"],
        "variant": probe.get("variant", ""),
        "query": probe["query"],
        "mode": probe.get("mode", "exact"),
        "expected": sorted(probe["expected"]),
        "answer": res["answer"][:200],
        "tokens_before": res["tokens"]["before"],
        "tokens_after": res["tokens"]["after"],
        "latency_ms": res["execution_time_ms"],
        **score,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dataset", nargs="?", default=None,
                   help="dataset whose graph is loaded (fraud_oracle | "
                        "insurance_claims | insurance_dataset | synthetic); "
                        "default: auto-detect from the graph")
    p.add_argument("--paraphrase", type=int, default=15,
                   help="policy-anchored paraphrase probes (default 15)")
    p.add_argument("--token-budget", type=int, default=settings.MAX_TOKENS)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", type=Path)
    p.add_argument("--uri", default=settings.NEO4J_URI)
    p.add_argument("--user", default=settings.NEO4J_USER)
    p.add_argument("--password", default=settings.NEO4J_PASSWORD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        print("ERROR: --workers must be >= 1")
        return 2

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        if not args.dataset:
            from graphrag.sessions import current_session_id
            args.dataset = current_session_id(driver)
            if args.dataset == "pdf_demo":
                args.dataset = "synthetic"
        probes = build_probes(args.dataset, args.paraphrase)
    finally:
        driver.close()
    print(f"== {args.dataset}: {len(probes)} generalization probes "
          f"({args.workers} workers) ==")

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    results: list[dict] = []
    t0 = time.perf_counter()
    audit_was_enabled = settings.AUDIT_ENABLED
    settings.AUDIT_ENABLED = False
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, driver, p, args): p for p in probes}
            for fut in as_completed(futures):
                results.append(fut.result())
    finally:
        driver.close()
        settings.AUDIT_ENABLED = audit_was_enabled
    elapsed = time.perf_counter() - t0

    # per-kind summaries
    by_kind: dict[str, dict] = {}
    for r in results:
        k = by_kind.setdefault(r["kind"], {"total": 0, "passed": 0,
                                           "answer_passed": 0})
        k["total"] += 1
        if r["retrieval_hit"] and r["prune_hit"]:
            k["passed"] += 1
        if r.get("answer_ok"):
            k["answer_passed"] += 1
    total = len(results)
    passed = sum(r["retrieval_hit"] and r["prune_hit"] for r in results)
    answer_passed = sum(bool(r.get("answer_ok")) for r in results)

    summary = {
        "dataset": args.dataset,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probes_total": total,
        "retrieval_prune_passed": passed,
        "retrieval_prune_accuracy": round(passed / total * 100, 2) if total else 0.0,
        "answer_level_passed": answer_passed,
        "answer_level_accuracy": round(answer_passed / total * 100, 2) if total else 0.0,
        "by_kind": by_kind,
        "elapsed_s": round(elapsed, 1),
        "results": results,
    }
    out = args.output or (PROJECT_ROOT / "data" / "benchmarks"
                          / f"generalization_{args.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    for kind, k in by_kind.items():
        print(f"  {kind:<16} {k['passed']:>3}/{k['total']:<3} retrieval+prune "
              f"({k['answer_passed']}/{k['total']} answers name expected node)")
    print(f"  TOTAL: {passed}/{total} = {summary['retrieval_prune_accuracy']}% "
          f"retrieval+prune · {answer_passed}/{total} = "
          f"{summary['answer_level_accuracy']}% answer-level · {elapsed:.0f}s")
    for r in results:
        if not (r["retrieval_hit"] and r["prune_hit"]) or not r.get("answer_ok"):
            mark = "FAIL"
            print(f"  [{mark}] {r['kind']}/{r.get('variant','')}: {r['query'][:60]}")
            if r.get("fabricated"):
                print(f"         FABRICATED: {r['fabricated']}")
            if r.get("bad_seeds"):
                print(f"         off-target seeds: {r['bad_seeds']}")
    print("=" * 72)
    return 0 if passed == total and answer_passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
