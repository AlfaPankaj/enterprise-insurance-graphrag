"""Full-scale V2 benchmark — the 10,200 ground-truth query run, in CI.

Replicates the documented V1 auditor-scale run (real_dataset_results.md) on
the V2 codebase to prove the v2 platform (opt-in features all default-off)
preserves v1 fidelity — then pushes the regenerated ``data/benchmarks/*.json``
back to the branch so the app/dashboard/README show V2 numbers.

Where it runs
-------------
* ONLY in GitHub Actions on an ``arena/*`` branch (or its PR) with a live
  Neo4j — the env gate below self-skips everywhere else, so local ``pytest``
  and post-merge runs on ``main`` never trigger an hour-long benchmark.
* The existing CI job (``ci.yml``) provides exactly what V1 used: Neo4j
  5.26-community at bolt://localhost:7687 + full requirements.txt.

Chain (identical commands/shapes to the V1 run)
-----------------------------------------------
seed demo graph -> synthetic 100 q + fraud (42 labels)
fraud_oracle ingest -> 5,500 q + fraud (923 + 1,500)
insurance_claims ingest -> 600 q + fraud (247 + 753)
insurance_dataset ingest -> 3,900 q
data_synthetic full ingest (53,503 rows) -> 100 q
edge cases (all datasets) + generalization probes
restore demo graph -> export_benchmark_proof.py (consolidated proof)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "benchmarks"

# ---- CI-only gate ----------------------------------------------------------

_HEAD_REF = os.environ.get("GITHUB_HEAD_REF", "")
_REF = os.environ.get("GITHUB_REF", "")
_IN_CI = os.environ.get("GITHUB_ACTIONS") == "true"
_ON_ARENA = _HEAD_REF.startswith("arena/") or "/arena/" in _REF or _REF.startswith("arena/")


def _neo4j_up() -> bool:
    try:
        from neo4j import GraphDatabase
        d = GraphDatabase.driver("bolt://localhost:7687",
                                 auth=("neo4j", "graphrag-demo"),
                                 connection_timeout=3)
        with d.session() as s:
            s.run("RETURN 1").consume()
        d.close()
        return True
    except Exception:  # noqa: BLE001 - gate probe must never raise
        return False


pytestmark = pytest.mark.skipif(
    not (_IN_CI and _ON_ARENA and _neo4j_up()),
    reason="full-scale benchmark: GitHub Actions + arena/* branch + live Neo4j only",
)

PY = sys.executable

CHAIN: list[str] = [
    # 1. PDF demo graph (synthetic session): 100 q + fraud
    f"{PY} scripts/seed_graph.py --reset --apply-schema",
    f"{PY} scripts/benchmark_real_dataset.py synthetic --queries 100",
    f"{PY} scripts/benchmark_fraud_detection.py --dataset synthetic",
    # 2. fraud_oracle: 5,500 q + fraud (923 fraud + 1,500 clean)
    f"{PY} scripts/ingest_real_dataset.py fraud_oracle --reset",
    f"{PY} scripts/benchmark_real_dataset.py fraud_oracle --queries 5500",
    f"{PY} scripts/benchmark_fraud_detection.py --dataset fraud_oracle",
    # 3. insurance_claims: 600 q + fraud (247 fraud + 753 clean)
    f"{PY} scripts/ingest_real_dataset.py insurance_claims --reset",
    f"{PY} scripts/benchmark_real_dataset.py insurance_claims --queries 600",
    f"{PY} scripts/benchmark_fraud_detection.py --dataset insurance_claims",
    # 4. insurance_dataset: 3,900 q
    f"{PY} scripts/ingest_real_dataset.py insurance_dataset --reset",
    f"{PY} scripts/benchmark_real_dataset.py insurance_dataset --queries 3900",
    # 5. data_synthetic: full 53,503-row ingest, 100 q
    f"{PY} scripts/ingest_real_dataset.py data_synthetic --reset",
    f"{PY} scripts/benchmark_real_dataset.py data_synthetic --queries 100",
    # 6. edge cases (all datasets, head/middle/tail) + generalization probes
    f"{PY} scripts/benchmark_edge_cases.py",
    f"{PY} scripts/benchmark_generalization.py insurance_claims",
    # 7. restore the demo graph + consolidate the proof
    f"{PY} scripts/seed_graph.py --reset --apply-schema",
    f"{PY} scripts/export_benchmark_proof.py",
]


def _sh(cmd: str) -> None:
    print(f"\n=== $ {cmd}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    # stream trimmed output into the test log (full stdout is huge)
    lines = proc.stdout.splitlines()
    for line in lines[:40]:
        print("   ", line)
    if len(lines) > 80:
        print(f"    [... {len(lines) - 80} lines ...]")
    for line in lines[-40:]:
        print("   ", line)
    assert proc.returncode == 0, f"benchmark step failed ({proc.returncode}): {cmd}"


def _digest() -> None:
    """Print every regenerated JSON's summary (CI-log = recovery fallback)."""
    proof = json.loads((BENCH / "benchmark_results.json").read_text())
    agg = proof["aggregate_metrics"]
    print("\n===== V2 BENCHMARK DIGEST (10,200 ground-truth queries) =====")
    print(json.dumps(agg, indent=2))
    print("--- fraud_detection ---")
    print(json.dumps(proof["fraud_detection"], indent=2))
    print("--- backend_performance ---")
    print(json.dumps(proof["backend_performance"], indent=2))
    print("--- query_mix ---")
    print(json.dumps(proof["query_mix"], indent=2))
    print("--- per_dataset ---")
    print(json.dumps(proof["per_dataset"], indent=2))
    for name in sorted(p.name for p in BENCH.glob("real_*.json")):
        d = json.loads((BENCH / name).read_text())
        d.pop("results", None)
        print(f"--- {name} ---")
        print(json.dumps(d, indent=2))
    for name in ("edge_cases.json", "generalization_insurance_claims.json"):
        p = BENCH / name
        if p.exists():
            d = json.loads(p.read_text())
            if isinstance(d, dict):
                for k in list(d):
                    if isinstance(d[k], list) and len(d[k]) > 20:
                        d[k] = f"<{len(d[k])} entries>"
            print(f"--- {name} ---")
            print(json.dumps(d, indent=2)[:4000])


def _push_back() -> None:
    """Commit the regenerated JSONs to the branch (GITHUB_TOKEN push never
    re-triggers workflows, so no CI loop)."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    token = os.environ.get("GITHUB_TOKEN", "")
    branch = _HEAD_REF or _REF.rsplit("/", 1)[-1]
    repo = "AlfaPankaj/enterprise-insurance-graphrag"
    steps = [
        "git config user.name 'arena-ai-coding-agent[bot]'",
        "git config user.email 'noreply@github.com'",
        "git add data/benchmarks",
        "git diff --cached --quiet || git commit -q -m "
        "'benchmark(v2): regenerate 10,200-query ground-truth results on V2 "
        "[skip ci]'",
    ]
    for s in steps:
        r = subprocess.run(s, cwd=ROOT, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"push-back step failed: {s}\n{r.stdout}\n{r.stderr}")
            return
    if token:
        url = f"https://x-access-token:{token}@github.com/{repo}.git"
        r = subprocess.run(f"git push {url} HEAD:{branch}", cwd=ROOT, shell=True,
                           capture_output=True, text=True)
        print("push-back:", "ok" if r.returncode == 0
              else f"FAILED (use the digest above)\n{r.stderr[-500:]}")
    else:
        print("no GITHUB_TOKEN — results only in this log (digest above)")


def test_v2_full_scale_benchmark_10_200_queries():
    for cmd in CHAIN:
        _sh(cmd)

    proof = json.loads((BENCH / "benchmark_results.json").read_text())
    agg = proof["aggregate_metrics"]

    # --- the V1 guarantees must hold on V2, at the same scale --------------
    n = agg["wilson_95ci_accuracy"]["n"]
    ok = agg["wilson_95ci_accuracy"]["ok"]
    assert n == 10_200, f"expected 10,200 benchmark queries, ran {n}"
    assert ok == n, f"retrieval+pruning misses on V2: {n - ok}"
    assert agg["retrieval_accuracy"] == 100.0
    assert agg["pruning_accuracy"] == 100.0
    assert agg["average_token_savings_pct"] > 0  # pruning still saves tokens

    fraud = proof["fraud_detection"]
    assert fraud["fraud_evaluated"] == 1_212, fraud["fraud_evaluated"]
    assert fraud["confusion"]["tp"] == 1_212 and fraud["confusion"]["fp"] == 0
    assert fraud["precision"] == 1.0 and fraud["recall"] == 1.0
    assert fraud["f1"] == 1.0

    _digest()
    _push_back()
