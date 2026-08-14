"""Generate the 20 benchmark queries with ground-truth expected node ids.

Expectations are derived *from the sample data itself* (the same data the PDFs
were generated from), so every query's correct answer is known by construction —
this is what makes Phase 3 accuracy measurable.

Output: data/benchmarks/benchmark_queries.json  (deterministic, seed 42)

Usage: .venv/Scripts/python.exe scripts/build_benchmark_queries.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = PROJECT_ROOT / "data" / "samples"
OUT = PROJECT_ROOT / "data" / "benchmarks" / "benchmark_queries.json"


def _load(name: str) -> list[dict]:
    return json.loads((SAMPLES / name).read_text(encoding="utf-8"))


def build() -> list[dict]:
    policies = _load("policies.json")
    claims = _load("claims.json")
    endorsements = _load("endorsements.json")
    pol_by_id = {p["id"]: p for p in policies}
    claim_records = {r["claim"]["id"]: r for r in claims}

    def pol(n: int) -> dict:
        return policies[n - 1]

    def clm_of_pol(p: dict) -> list[dict]:
        return [c for c in claims if c["claim"]["policy_id"] == p["id"]]

    queries: list[dict] = []

    # --- policy facts ---
    p1, p5, p10, p20, p30, p40 = pol(1), pol(5), pol(10), pol(20), pol(30), pol(40)
    queries += [
        {"id": "q01", "category": "policy_fact",
         "query": f"What is the status of policy {p1['id']}?", "expected": [p1["id"]]},
        {"id": "q02", "category": "policy_fact",
         "query": f"What is the annual premium of policy {p10['id']}?", "expected": [p10["id"]]},
        {"id": "q03", "category": "policy_fact",
         "query": f"What type is policy {p20['id']}?", "expected": [p20["id"]]},
        {"id": "q04", "category": "policy_fact",
         "query": f"What is the deductible of policy {p30['id']}?", "expected": [p30["id"]]},
    ]

    # --- policyholder & coverages ---
    ph5 = p5["policyholder"]
    queries += [
        {"id": "q05", "category": "policyholder",
         "query": f"Who is the policyholder of policy {p5['id']}?", "expected": [ph5["id"], p5["id"]]},
        {"id": "q06", "category": "coverage",
         "query": f"List the coverages of policy {p5['id']}",
         "expected": [p5["id"]] + [c["id"] for c in p5["coverages"]]},
        {"id": "q07", "category": "keyword",
         "query": f"Find the policies held by {ph5['name']}",
         "expected": [ph5["id"], p5["id"]]},
    ]

    # --- claims & fraud ---
    flagged = next(r for r in claims if r.get("fraud_flag"))
    fc = flagged["claim"]
    fpol = pol_by_id[fc["policy_id"]]
    inv_claim = next((r for r in claims if r.get("investigator")), None)
    ic = inv_claim["claim"]
    queries += [
        {"id": "q08", "category": "fraud",
         "query": f"Does claim {fc['id']} have a fraud flag?",
         "expected": [fc["id"], flagged["fraud_flag"]["id"]]},
        {"id": "q09", "category": "investigator",
         "query": f"Who investigates claim {ic['id']}?",
         "expected": [ic["id"], inv_claim["investigator"]["id"]]},
        {"id": "q10", "category": "claim_fact",
         "query": f"Find the policy for claim {fc['id']}",
         "expected": [fc["id"], fpol["id"]]},
        {"id": "q11", "category": "claim_fact",
         "query": f"What is the status of claim {ic['id']}?", "expected": [ic["id"]]},
    ]

    # --- multi-hop: claim -> policy -> coverages ---
    c5 = clm_of_pol(p5)[0]["claim"]  # every policy has at least one claim
    cov_pol = pol_by_id[c5["policy_id"]]
    queries.append({
        "id": "q12", "category": "multi_hop",
        "query": f"Which coverages apply to claim {c5['id']}?",
        "expected": [c5["id"], cov_pol["id"]] + [c["id"] for c in cov_pol["coverages"]],
    })

    # --- endorsements ---
    end_pol = next((p for p in policies if p["endorsements"]), None)
    end = end_pol["endorsements"][0]
    queries += [
        {"id": "q13", "category": "endorsement",
         "query": f"What endorsements amend policy {end_pol['id']}?",
         "expected": [end_pol["id"]] + [e["id"] for e in end_pol["endorsements"]]},
        {"id": "q14", "category": "endorsement",
         "query": f"Which {end['type'].replace('_', ' ').lower()} endorsements are on policy {end_pol['id']}?",
         "expected": [end_pol["id"], end["id"]]},
    ]

    # --- cause / status / amount filters ---
    cause = "Fire damage"
    fire_claims = [r for r in claims if r["claim"]["cause"] == cause][:5]
    fire_pol = pol_by_id[fire_claims[0]["claim"]["policy_id"]]
    queries.append({
        "id": "q15", "category": "keyword_filter",
        "query": f"Show me claims on policy {fire_pol['id']} caused by {cause.lower()}",
        "expected": [fire_pol["id"]] + [r["claim"]["id"] for r in fire_claims
                                        if r["claim"]["policy_id"] == fire_pol["id"]],
    })

    paid = [r for r in claims if r["claim"]["status"] == "PAID"]
    paid_pol = pol_by_id[paid[0]["claim"]["policy_id"]]
    paid_of = [r for r in paid if r["claim"]["policy_id"] == paid_pol["id"]]
    queries.append({
        "id": "q16", "category": "status_filter",
        "query": f"Which claims under policy {paid_pol['id']} are paid?",
        "expected": [paid_pol["id"]] + [r["claim"]["id"] for r in paid_of],
    })

    big = next(r for r in claims if r["claim"]["amount"] > 100_000)
    big_pol = pol_by_id[big["claim"]["policy_id"]]
    queries.append({
        "id": "q17", "category": "amount_filter",
        "query": f"Show claims over $100,000 on policy {big_pol['id']}",
        "expected": [big_pol["id"], big["claim"]["id"]],
    })

    # --- investigator by role ---
    inv_by_role = next((r for r in claims if r.get("investigator") and r["investigator"]["role"] == "FRAUD_SPECIALIST"), None)
    if inv_by_role:
        queries.append({
            "id": "q18", "category": "investigator",
            "query": "Which claims are handled by a FRAUD_SPECIALIST?",
            "expected": [inv_by_role["investigator"]["id"], inv_by_role["claim"]["id"]],
        })
    else:
        queries.append({
            "id": "q18", "category": "investigator",
            "query": f"Which claims does {inv_claim['investigator']['id']} investigate?",
            "expected": [inv_claim["investigator"]["id"], ic["id"]],
        })

    # --- fraud severity ---
    queries.append({
        "id": "q19", "category": "fraud",
        "query": f"What is the severity of the fraud flag on claim {fc['id']}?",
        "expected": [fc["id"], flagged["fraud_flag"]["id"]],
    })

    # --- high-risk policyholder ---
    high_risk = max(policies, key=lambda p: p["policyholder"]["risk_score"])
    queries.append({
        "id": "q20", "category": "policyholder",
        "query": f"What is the risk score of {high_risk['policyholder']['name']}?",
        "expected": [high_risk["policyholder"]["id"], high_risk["id"]],
    })

    assert len(queries) == 20, f"expected 20 queries, got {len(queries)}"
    return queries


def main() -> int:
    queries = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(queries, indent=2), encoding="utf-8")
    print(f"wrote {len(queries)} benchmark queries -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
