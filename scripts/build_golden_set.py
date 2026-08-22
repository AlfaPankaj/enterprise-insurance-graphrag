#!/usr/bin/env python3
"""Build the golden question set for answer-quality evaluation (v2 — WS-C, G15).

Ground truth by construction — every question and its expected behavior is
computed from ``data/samples/*.json`` (the exact records the demo graph is
seeded from), so no expected answer is hand-typed:

  * fraud yes/no            — claims.json fraud_flag labels (mixed)
  * status of claim         — claim ids (+ their status string)
  * coverages of policy     — coverage ids nested in policies.json
  * policyholder of policy  — policyholder id per policy
  * investigator of claim   — investigator id per claim
  * premium thresholds      — policies with premium ≥ X (computed)
  * paraphrase              — same question re-worded WITHOUT the id quote
  * negative probes         — non-existent ids that must come back empty/refused

Output: data/benchmarks/golden_questions.json (committed; regenerable).

Usage:  python scripts/build_golden_set.py [--size 60]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = PROJECT_ROOT / "data" / "samples"
OUT = PROJECT_ROOT / "data" / "benchmarks" / "golden_questions.json"


def load(name: str) -> list:
    return json.loads((SAMPLES / name).read_text(encoding="utf-8"))


def build(size: int = 60, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    policies = load("policies.json")
    claims = load("claims.json")

    flagged = [c for c in claims if c.get("fraud_flag")]
    clean = [c for c in claims if not c.get("fraud_flag")]
    with_inv = [c for c in claims if c.get("investigator")]
    with_cov = [p for p in policies if p["coverages"]]

    out: list[dict] = []

    # 1) fraud yes/no — mix of flagged and clean claims
    for rec in rng.sample(flagged, min(6, len(flagged))) + \
            rng.sample(clean, min(6, len(clean))):
        cid = rec["claim"]["id"]
        out.append({
            "query": f"Does claim {cid} have a fraud flag?",
            "category": "fraud",
            "answerable": True,
            "expected_ids": [cid],
            "ground_truth": "fraud" if rec.get("fraud_flag") else "clean",
        })

    # 2) claim status
    for rec in rng.sample(claims, min(8, len(claims))):
        cid = rec["claim"]["id"]
        out.append({
            "query": f"What is the status of claim {cid}?",
            "category": "status",
            "answerable": True,
            "expected_ids": [cid],
            "expected_token": str(rec["claim"].get("status", "")).lower(),
        })

    # 3) coverages per policy
    for pol in rng.sample(with_cov, min(8, len(with_cov))):
        cov_ids = [c["id"] for c in pol["coverages"]]
        out.append({
            "query": f"Which coverages apply to policy {pol['id']}?",
            "category": "coverage",
            "answerable": True,
            "expected_ids": cov_ids,
        })

    # 4) policyholder per policy
    for pol in rng.sample(policies, min(6, len(policies))):
        out.append({
            "query": f"Who holds policy {pol['id']}?",
            "category": "policyholder",
            "answerable": True,
            "expected_ids": [pol["policyholder"]["id"]],
        })

    # 5) investigator per claim
    for rec in rng.sample(with_inv, min(6, len(with_inv))):
        cid = rec["claim"]["id"]
        out.append({
            "query": f"Who investigates claim {cid}?",
            "category": "investigator",
            "answerable": True,
            "expected_ids": [rec["investigator"]["id"]],
        })

    # 6) premium thresholds (computed expected set)
    premiums = sorted({p["premium"] for p in policies})
    threshold = premiums[len(premiums) // 3]  # a value ~1/3 of policies exceed
    over = [p["id"] for p in policies if p["premium"] >= threshold]
    out.append({
        "query": f"Which policies have a premium over ${threshold:,.0f}?",
        "category": "threshold",
        "answerable": True,
        "expected_ids": over,
    })

    # 7) paraphrases — same intent, NO entity id quoted
    pol = rng.choice(with_cov)
    out.append({
        "query": "What protections does this commercial policy offer to its holder?",
        "category": "paraphrase",
        "answerable": True,
        "expected_ids": [c["id"] for c in pol["coverages"]],
        "paraphrase_of": f"Which coverages apply to policy {pol['id']}?",
    })
    flagged_claim = rng.choice(flagged)["claim"]["id"]
    out.append({
        "query": "Is there any sign of fraudulent activity on this claim file?",
        "category": "paraphrase",
        "answerable": True,
        "expected_ids": [flagged_claim],
        "paraphrase_of": f"Does claim {flagged_claim} have a fraud flag?",
    })

    # 8) negative probes — non-existent ids must come back empty / refuse
    for q in (
        "Does claim CLM-99999 have a fraud flag?",
        "What is the status of policy POL-99999?",
        "Which coverages apply to policy POL-99999?",
    ):
        out.append({
            "query": q,
            "category": "negative",
            "answerable": False,
            "expected_ids": [],
        })

    out = out[:size]
    rng.shuffle(out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=60,
                        help="number of questions in the set (cap)")
    args = parser.parse_args(argv)

    questions = build(size=args.size, seed=42)
    OUT.write_text(json.dumps({
        "built_from": ["data/samples/policies.json", "data/samples/claims.json"],
        "count": len(questions),
        "questions": questions,
    }, indent=2), encoding="utf-8")
    by_cat: dict[str, int] = {}
    for q in questions:
        by_cat[q["category"]] = by_cat.get(q["category"], 0) + 1
    print(f"wrote {len(questions)} golden questions to {OUT.relative_to(PROJECT_ROOT)}")
    print("categories:", ", ".join(f"{k}={v}" for k, v in sorted(by_cat.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
