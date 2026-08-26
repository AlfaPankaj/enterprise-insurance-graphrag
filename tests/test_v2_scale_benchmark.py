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

Visibility
----------
The driving sandbox cannot read Actions logs, so progress + results are
mirrored into a PR comment (runners have network). The comment is updated
as each chain step completes.

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
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from tests.conftest import annotate

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "benchmarks"
REPO = "AlfaPankaj/enterprise-insurance-graphrag"

# ---- CI-only gate ----------------------------------------------------------

_HEAD_REF = os.environ.get("GITHUB_HEAD_REF", "")
_REF = os.environ.get("GITHUB_REF", "")
_IN_CI = os.environ.get("GITHUB_ACTIONS") == "true"
_ON_ARENA = _HEAD_REF.startswith("arena/") or "/arena/" in _REF \
    or _REF.startswith("arena/")


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

# steps whose exit code must NOT gate the chain (their JSON is validated
# separately below); benchmark_generalization returns 1 unless answer-level
# accuracy is ALSO 100% — but V1's accepted result is 21/24 = 87.5%
_TOLERANT = {"generalization probes"}


def _clean(out: str) -> str:
    """Drop Neo4j notification spam (tenant_id property warnings) so the
    real traceback stays visible in annotations."""
    keep = [l for l in out.splitlines()
            if "Received notification from DBMS" not in l]
    return "\n".join(keep)


CHAIN: list[tuple[str, str]] = [
    ("seed demo graph", f"{PY} scripts/seed_graph.py --reset --apply-schema"),
    ("synthetic 100 q", f"{PY} scripts/benchmark_real_dataset.py synthetic --queries 100"),
    ("fraud benchmark — synthetic (42 labels)",
     f"{PY} scripts/benchmark_fraud_detection.py --dataset synthetic"),
    ("fraud_oracle — ingest 15,420 claims",
     f"{PY} scripts/ingest_real_dataset.py fraud_oracle --reset"),
    ("fraud_oracle — 5,500 ground-truth queries",
     f"{PY} scripts/benchmark_real_dataset.py fraud_oracle --queries 5500"),
    ("fraud benchmark — fraud_oracle (923 + 1,500)",
     f"{PY} scripts/benchmark_fraud_detection.py --dataset fraud_oracle"),
    ("insurance_claims — ingest 1,000 claims",
     f"{PY} scripts/ingest_real_dataset.py insurance_claims --reset"),
    ("insurance_claims — 600 ground-truth queries",
     f"{PY} scripts/benchmark_real_dataset.py insurance_claims --queries 600"),
    ("fraud benchmark — insurance_claims (247 + 753)",
     f"{PY} scripts/benchmark_fraud_detection.py --dataset insurance_claims"),
    ("generalization probes", f"{PY} scripts/benchmark_generalization.py insurance_claims"),
    ("insurance_dataset — ingest 13,000 customers",
     f"{PY} scripts/ingest_real_dataset.py insurance_dataset --reset"),
    ("insurance_dataset — 3,900 ground-truth queries",
     f"{PY} scripts/benchmark_real_dataset.py insurance_dataset --queries 3900"),
    ("data_synthetic — full 53,503-row ingest",
     f"{PY} scripts/ingest_real_dataset.py data_synthetic --reset"),
    ("data_synthetic — 100 ground-truth queries",
     f"{PY} scripts/benchmark_real_dataset.py data_synthetic --queries 100"),
    ("edge cases (all datasets)", f"{PY} scripts/benchmark_edge_cases.py"),
    ("restore demo graph", f"{PY} scripts/seed_graph.py --reset --apply-schema"),
    ("export consolidated proof", f"{PY} scripts/export_benchmark_proof.py"),
]


# ---- PR-comment progress mirror (runners have network; the driver sandbox
#      cannot read Actions logs, so this is the log pipeline) ----------------

_MARKER = f"<!-- v2-bench:{os.environ.get('GITHUB_RUN_ID', 'local')} -->"


def _api(path: str, token: str, method: str = "GET", payload: dict | None = None):
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/{path}",
        data=json.dumps(payload).encode("utf-8") if payload else None,
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json"},
        method=method)
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def _pr(token: str) -> str | None:
    m = re.match(r"refs/pull/(\d+)/", os.environ.get("GITHUB_REF", "") or "")
    return m.group(1) if m else None


class _Progress:
    """One comment per run; edited as the chain advances."""

    def __init__(self):
        self.lines: list[str] = ["**V2 full-scale benchmark** — chain starting…", ""]
        self.done = 0
        self.comment_id: int | None = None
        self.token = os.environ.get("GITHUB_TOKEN", "")
        self.pr = _pr(self.token) if self.token else None

    def _flush(self):
        if not (self.token and self.pr):
            return
        body = _MARKER + "\n" + "\n".join(self.lines)
        try:
            if self.comment_id is None:
                # find or create this run's comment
                comments = _api(f"issues/{self.pr}/comments", self.token)
                for c in comments:
                    if _MARKER in c.get("body", ""):
                        self.comment_id = c["id"]
                        break
                if self.comment_id is None:
                    self.comment_id = _api(
                        f"issues/{self.pr}/comments", self.token, "POST",
                        {"body": body}).get("id")
                return
            _api(f"issues/comments/{self.comment_id}", self.token,
                 "PATCH", {"body": body})
        except Exception as exc:  # noqa: BLE001 - mirror must never fail CI
            print(f"[bench-mirror] {exc}")

    def step_done(self, label: str, tail: str = ""):
        self.done += 1
        self.lines.append(f"✅ {label}" + (f" — `{tail}`" if tail else ""))
        self._flush()
        if self.done % 5 == 0 or self.done == len(CHAIN):
            annotate("notice", f"bench progress {self.done}/{len(CHAIN)}", label)

    def step_failed(self, label: str, out: str):
        self.lines.append(f"❌ **{label}** failed — last output:")
        self.lines.append("```\n" + "\n".join(out.splitlines()[-60:]) + "\n```")
        self._flush()
        annotate("error", f"BENCH STEP FAILED: {label}",
                 "$ " + next(c for l, c in CHAIN if l == label) +
                 "\n--- last output ---\n" + "\n".join(out.splitlines()[-40:]))

    def raw(self, text: str):
        self.lines.append(text)
        self._flush()


def _run_step(cmd: str, label: str, progress: _Progress) -> None:
    print(f"\n=== $ {cmd}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    out = _clean(proc.stdout or "")
    lines = out.splitlines()
    for line in lines[:30]:
        print("   ", line)
    if len(lines) > 60:
        print(f"    [... {len(lines) - 60} lines ...]")
    for line in lines[-30:]:
        print("   ", line)
    if proc.returncode != 0 and label not in _TOLERANT:
        progress.step_failed(label, out)
        raise AssertionError(f"benchmark step failed ({proc.returncode}): {cmd}")
    progress.step_done(label)


def _digest_lines() -> list[str]:
    proof = json.loads((BENCH / "benchmark_results.json").read_text())
    agg = proof["aggregate_metrics"]
    out = ["", "===== V2 BENCHMARK DIGEST (10,200 ground-truth queries) ====="]
    out.append("```json")
    out.append(json.dumps(agg, indent=2))
    out.append("--- fraud ---")
    out.append(json.dumps(proof["fraud_detection"], indent=2))
    out.append("--- backend_performance ---")
    out.append(json.dumps(proof["backend_performance"], indent=2))
    out.append("--- per_dataset ---")
    out.append(json.dumps(proof["per_dataset"], indent=2))
    out.append("--- query_mix ---")
    out.append(json.dumps(proof["query_mix"], indent=2))
    out.append("```")
    return out


def _annotate_digest(proof: dict) -> None:
    """Deliver the headline numbers via annotations (read-only-token safe).

    The step's notice budget is 10 — one notice per payload, single-line,
    compact JSON. These payloads are what updates README/app/dashboard and
    the data/benchmarks/*.json summaries, so they must always survive.
    """
    def compact(obj) -> str:
        return json.dumps(obj, separators=(",", ":"))

    per_file = {}
    for f in sorted(BENCH.glob("real_*.json")):
        b = json.loads(f.read_text())
        per_file[f.stem] = {k: b.get(k) for k in
                            ("queries", "retrieval_accuracy", "pruning_accuracy",
                             "avg_savings_pct", "avg_latency_ms", "timestamp",
                             "dataset")}
    fraud_files = {}
    for f in sorted(BENCH.glob("fraud_detection_*.json")):
        b = json.loads(f.read_text())
        fraud_files[f.stem] = {k: b.get(k) for k in
                               ("fraud_evaluated", "clean_evaluated",
                                "confusion", "precision", "recall", "f1")}
    gen = json.loads((BENCH / "generalization_insurance_claims.json").read_text())
    gen_summary = {k: gen.get(k) for k in
                   ("probes_total", "retrieval_prune_passed",
                    "retrieval_prune_accuracy", "answer_level_passed",
                    "answer_level_accuracy", "by_kind")}
    edge = {}
    ep = BENCH / "edge_cases.json"
    if ep.exists():
        eb = json.loads(ep.read_text())
        edge = {k: eb.get(k) for k in
                ("total_queries", "retrieval_accuracy", "avg_savings_pct",
                 "avg_latency_ms", "timestamp")}

    parts = [
        ("V2R aggregate+fraud",
         compact({"aggregate_metrics": proof["aggregate_metrics"],
                  "fraud_detection": proof["fraud_detection"]})),
        ("V2R backend+mix",
         compact({"backend_performance": proof["backend_performance"],
                  "query_mix": proof["query_mix"]})),
        ("V2R per_dataset",
         compact({"per_dataset": proof["per_dataset"]})),
        ("V2R per-file real",
         compact(per_file)),
        ("V2R fraud files",
         compact(fraud_files)),
        ("V2R gen+edge",
         compact({"generalization": gen_summary, "edge_cases": edge})),
    ]
    # annotate() now emits the titleless ::kind::message form; keep the
    # total at/under the runner's 10-annotations-per-step budget
    for i, (title, payload) in enumerate(parts, 1):
        annotate("error", title, payload[:2000])
    annotate("notice", "canary", "V2R-CANARY-NOTICE-OK")
    annotate("warning", "canary", "V2R-CANARY-WARNING-OK")
    annotate("error", "canary", "V2R-CANARY-ERROR-OK")


def _push_back(progress) -> None:
    """Commit the regenerated JSONs onto the branch tip.

    PR-event checkouts sit on a MERGE commit; committing and pushing HEAD
    directly can be rejected. Rebuild the commit on the real branch tip
    instead (fast-forward, and GITHUB_TOKEN pushes never re-trigger CI).
    """
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    token = os.environ.get("GITHUB_TOKEN", "")
    branch = _HEAD_REF or _REF.rsplit("/", 1)[-1]
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "benchmarks"
    shutil.copytree(BENCH, tmp)
    steps = [
        "git config user.name 'arena-ai-coding-agent[bot]'",
        "git config user.email 'noreply@github.com'",
        f"git fetch origin {branch}",
        "git reset --hard FETCH_HEAD",           # move to the branch tip
        "rm -rf data/benchmarks",
        f"cp -r {tmp} data/benchmarks",
        "git add data/benchmarks",
        "git diff --cached --quiet || git commit -q -m "
        "'benchmark(v2): regenerate 10,200-query ground-truth results on V2 "
        "[skip ci]'",
    ]
    for s in steps:
        r = subprocess.run(s, cwd=ROOT, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            progress.raw(f"push-back prep failed: `{s}` — {r.stderr[-300:]}")
            return
    if token:
        url = f"https://x-access-token:{token}@github.com/{REPO}.git"
        r = subprocess.run(f"git push {url} HEAD:{branch}", cwd=ROOT, shell=True,
                           capture_output=True, text=True)
        progress.raw("results pushed back to the branch" if r.returncode == 0
                     else f"push-back FAILED: {r.stderr[-400:]}")
    else:
        progress.raw("no GITHUB_TOKEN — results only in run output")


def test_v2_full_scale_benchmark_10_200_queries():
    progress = _Progress()
    print(f"[bench] chain: {len(CHAIN)} steps", flush=True)
    for label, cmd in CHAIN:
        _run_step(cmd, label, progress)

    proof = json.loads((BENCH / "benchmark_results.json").read_text())
    agg = proof["aggregate_metrics"]

    # ---- deliver results FIRST — the JSONs must land on the branch even if
    # ---- a quality gate below trips (they are the honest V2 measurement)
    for line in _digest_lines():
        progress.raw(line)
    _annotate_digest(proof)
    _push_back(progress)

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

    # --- generalization probes: report + diagnose, gate at a floor ---------
    # V1 baseline: retrieval+prune 24/24, answer-level 21/24 (87.5%).
    gen_p = BENCH / "generalization_insurance_claims.json"
    gen = json.loads(gen_p.read_text())
    assert gen["probes_total"] >= 24, gen["probes_total"]
    failed = [p for p in gen["results"]
              if not (p.get("retrieval_hit") and p.get("prune_hit"))]
    if failed:
        detail = "\n".join(
            f"[{p['kind']}/{p['variant']}] {p['query']}\n"
            f"  expected={p.get('expected')} retrieval={p.get('retrieval_hit')} "
            f"prune={p.get('prune_hit')} bad_seeds={p.get('bad_seeds')}"
            for p in failed)
        annotate("error", f"V2 generalization: {len(failed)} probe(s) below "
                          f"V1 baseline (V1: 24/24 retrieval+prune)", detail[:6000])
    annotate("notice", "V2 generalization probes",
             json.dumps({k: gen[k] for k in
                         ("probes_total", "retrieval_prune_passed",
                          "retrieval_prune_accuracy", "answer_level_passed",
                          "answer_level_accuracy", "by_kind")}))
    progress.raw(f"generalization: retrieval+prune "
                 f"{gen['retrieval_prune_passed']}/{gen['probes_total']}, "
                 f"answer-level {gen['answer_level_passed']}/"
                 f"{gen['probes_total']} (V1 accepted: 21/24)")
    # strict again: the id-digit numeric-seed leak that failed the three
    # negative probes is fixed (_threshold_numbers) — 24/24 is expected
    assert gen["retrieval_prune_passed"] == gen["probes_total"], gen

    progress.raw("benchmarks delivered; 10,200/10,200 + fraud 100% asserted")
