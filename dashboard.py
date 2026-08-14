"""GraphRAG Token Optimization Dashboard (Shot 2) — Streamlit app.

Run:  streamlit run dashboard.py
     (.venv/Scripts/python.exe -m streamlit run dashboard.py)

Shows the token-optimization story end to end:
  * live query runner  -> baseline vs optimized tokens + savings %
  * benchmark summary  -> accuracy, avg savings, avg latency (from results JSON)
  * per-query detail   -> which nodes survived / were pruned
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from neo4j import GraphDatabase  # noqa: E402

from src.graphrag.config import settings  # noqa: E402
from src.graphrag.fraud_ground_truth import (  # noqa: E402
    build_comparison,
    detect_dataset,
    load_ground_truth,
)
from src.graphrag.query_pipeline import run_query  # noqa: E402
from src.graphrag.reranker import make_reranker  # noqa: E402
from src.graphrag.custom_sessions import list_custom_sessions  # noqa: E402
from src.graphrag.sessions import ensure_pdf_demo_fraud_benchmark  # noqa: E402

st.set_page_config(page_title="GraphRAG Token Optimization Dashboard", layout="wide")

# Generate the PDF demo graph's fraud benchmark once in the background, so the
# Pipeline Validation table shows real Fraud P/R/F1 for the pdf_demo row.
ensure_pdf_demo_fraud_benchmark()

ACCENT = "#1f6feb"


@st.cache_resource
def get_driver():
    """One cached Neo4j driver for the whole dashboard session."""
    return GraphDatabase.driver(
        settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )


@st.cache_data(show_spinner=False)
def cached_ground_truth(dataset: str | None):
    """Fraud labels for the loaded dataset (CSV/sample parsing is cached)."""
    return load_ground_truth(dataset)


def loaded_dataset() -> str | None:
    """Name of the dataset stamped on the loaded graph (None = no marker)."""
    try:
        return detect_dataset(get_driver())
    except Exception:
        return None


def _load_benchmark_entry(entry: dict, loaded: str | None) -> dict | None:
    """Aggregate stored benchmark JSONs for one validation-table entry.

    ``entry`` = {"label", "files", "desc", "kind", "fraud_name"} where
    ``kind`` is ``"real"`` (data/benchmarks/real_<name>.json +
    fraud_detection_<name>.json), ``"pdf"`` (edge_cases.json +
    fraud_detection_synthetic.json) or ``"custom"`` (real_<name>.json when the
    user has benchmarked their own upload). ``fraud_name`` overrides the
    fraud-detection file name. Returns a display row or None (``no_bench``
    marks a custom session that has no benchmark results yet; ``no_fraud``
    marks a dataset whose source has no fraud labels).
    """
    kind = entry.get("kind", "real")
    files = entry["files"]
    fraud_name = entry.get("fraud_name", files[0] if files else None)
    q_total = 0
    acc_ret = acc_prn = sav = lat = 0.0
    entry_conf = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    has_fraud_file = False

    for name in files:
        if kind == "pdf":
            # the PDF demo row reads the 100-query synthetic benchmark
            # (real_synthetic.json); edge_cases.json is the older 20-query
            # fallback while real_synthetic hasn't been generated yet
            bpath = ROOT / "data" / "benchmarks" / "real_synthetic.json"
            if not bpath.exists():
                bpath = ROOT / "data" / "benchmarks" / "edge_cases.json"
        else:
            bpath = ROOT / "data" / "benchmarks" / f"real_{name}.json"
        if not bpath.exists():
            continue
        b = json.loads(bpath.read_text(encoding="utf-8"))
        if kind == "pdf":
            if "total_queries" in b:  # edge_cases fallback format
                q = b["total_queries"]
                acc_ret += b["retrieval_accuracy"] * q
                rs = b.get("results") or []
                acc_prn += (100.0 * sum(1 for r in rs if r.get("survived_prune")) / len(rs)) * q if rs else 0.0
                sav += b["avg_savings_pct"] * q
                lat += b["avg_latency_ms"] * q
            else:  # real_synthetic format (same as real_*.json)
                q = b["queries"]
                q_total += q
                acc_ret += b["retrieval_accuracy"] * q
                acc_prn += b["pruning_accuracy"] * q
                sav += b["avg_savings_pct"] * q
                lat += b["avg_latency_ms"] * q
            continue
        q = b["queries"]
        q_total += q
        acc_ret += b["retrieval_accuracy"] * q
        acc_prn += b["pruning_accuracy"] * q
        sav += b["avg_savings_pct"] * q
        lat += b["avg_latency_ms"] * q

    # fraud precision/recall — every kind reads a fraud_detection_*.json
    # (fraud_name maps the pdf_demo row to the synthetic benchmark)
    if fraud_name:
        fpath = ROOT / "data" / "benchmarks" / f"fraud_detection_{fraud_name}.json"
        if fpath.exists():
            has_fraud_file = True
            fj = json.loads(fpath.read_text(encoding="utf-8"))
            for k in entry_conf:
                entry_conf[k] += fj["confusion"][k]

    if not q_total:
        if kind == "custom":
            return {"no_bench": True, "Dataset": entry["label"],
                    "Size": entry["desc"], "Queries": 0,
                    "Retrieval acc": "—", "Pruning acc": "—",
                    "Token savings": "—", "Latency": "—",
                    "Fraud P/R/F1": "—", "Hint": entry["hint"]}
        return None

    p = entry_conf["tp"] / (entry_conf["tp"] + entry_conf["fp"]) if (entry_conf["tp"] + entry_conf["fp"]) else 0
    r = entry_conf["tp"] / (entry_conf["tp"] + entry_conf["fn"]) if (entry_conf["tp"] + entry_conf["fn"]) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    loaded = "● " if (loaded and loaded in files) else ""
    if sum(entry_conf.values()):
        fraud_display = f"{p * 100:.0f}% / {r * 100:.0f}% / {f1 * 100:.0f}%"
    elif entry.get("no_fraud"):
        fraud_display = "no labels"
    elif fraud_name and not has_fraud_file:
        fraud_display = "—"  # benchmark not run yet (auto-runs in background)
    else:
        fraud_display = "—"
    return {
        "Dataset": f"{loaded}{entry['label']}",
        "Size": entry["desc"],
        "Queries": q_total,
        "Retrieval acc": f"{acc_ret / q_total:.0f}%",
        "Pruning acc": f"{acc_prn / q_total:.0f}%",
        "Retrieval raw": acc_ret / q_total,
        "Pruning raw": acc_prn / q_total,
        "Token savings": f"{sav / q_total:.1f}%",
        "Latency": f"{lat / q_total:.0f} ms",
        "Fraud P/R/F1": fraud_display,
        "_confusion": entry_conf,
    }


def validation_entries(loaded: str | None) -> list[dict]:
    """All validation-table entries: 3 real datasets + PDF demo graph + any
    user-uploaded custom sessions (so a new upload appears automatically)."""
    entries = [
        {"label": "fraud_oracle", "files": ["fraud_oracle"],
         "desc": "15,420 claims · 923 fraud", "kind": "real"},
        {"label": "insurance_claims", "files": ["insurance_claims"],
         "desc": "1,000 claims · 247 fraud", "kind": "real"},
        {"label": "insurance_dataset (+ synthetic)",
         "files": ["insurance_dataset", "data_synthetic"],
         "desc": "13,000 + 53,503 rows · same Kaggle source", "kind": "real",
         "no_fraud": True},  # neither CSV has a fraud column
        {"label": "pdf_demo (PDF pipeline)", "files": ["synthetic"],
         "desc": "Synthetic demo graph — policies, claims, endorsements",
         "kind": "pdf", "fraud_name": "synthetic"},
    ]
    for rec in list_custom_sessions():
        entries.append({
            "label": rec["name"], "files": [rec["name"]], "kind": "custom",
            "fraud_name": rec["name"],
            "desc": f"Custom {rec['kind'].upper()} — {rec.get('note') or rec['name']}",
            "hint": "Run `scripts/benchmark_real_dataset.py <name>` to validate this upload.",
        })
    return entries


def neo4j_ok() -> bool:
    try:
        with get_driver().session() as s:
            s.run("RETURN 1")
        return True
    except Exception:
        return False


st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2rem; }}
      .kpi {{ background: #f5f7fa; border-left: 4px solid {ACCENT};
             border-radius: 8px; padding: 14px 18px; margin-bottom: 8px; }}
      .kpi .label {{ font-size: 0.8rem; color: #57606a; }}
      .kpi .value {{ font-size: 1.6rem; font-weight: 700; color: #0d1117; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 GraphRAG Token Optimization Dashboard")
st.caption("Cost-optimized GraphRAG for commercial insurance — Shot 2: retrieve → re-rank → prune")

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Status")
    st.write("**Neo4j:**", "✅ connected" if neo4j_ok() else "❌ not reachable — start it with `docker start graphrag-neo4j`")
    ds = loaded_dataset()
    st.write("**Loaded dataset:**", f"`{ds or 'synthetic (demo)'}`")
    gt = cached_ground_truth(ds)
    if gt:
        fraud = sum(1 for v in gt.values() if v)
        st.write("**Fraud labels:**", f"{fraud:,} of {len(gt):,} claims")
        st.caption("answers to fraud queries get a ground-truth check below")
    else:
        st.write("**Fraud labels:**", "none (this dataset has no fraud column)")
    active = make_reranker().name
    st.write("**Reranker:**", f"`{active}`")
    st.caption("auto → cross-encoder when available, else lexical")
    st.divider()
    st.header("Query runner")
    max_hops = st.slider("Max hops", 1, 4, settings.MAX_HOPS)
    token_budget = st.slider("Token budget", 256, 8192, settings.MAX_TOKENS, step=256)
    reranker_mode = st.selectbox("Reranker mode", ["auto", "cross-encoder", "lexical"])
    answer_mode = st.selectbox("Answer mode", ["extractive", "auto", "llm"],
                               index=["extractive", "auto", "llm"].index(settings.ANSWER_MODE)
                               if settings.ANSWER_MODE in ("extractive", "auto", "llm") else 0)
    st.caption("auto = Ollama LLM with extractive fallback")
    st.divider()
    try:
        # multipage-aware link; bare mode (AppTest/CI) lacks the page manager
        st.page_link("src/graphrag/audit_ui.py", label="Audit trail & lineage", icon="🧾")
    except Exception:
        st.markdown("**🧾 Audit trail & lineage** — run `streamlit run "
                    "src/graphrag/audit_ui.py`")
    st.caption("Shot 3: every query logs its traversal path + Cypher")

# ---------------- pipeline validation (all sessions) ----------------
st.header("Pipeline validation")
st.caption("Benchmarks per session — 3 real Excel datasets, the **PDF demo graph** (edge-case "
           "benchmark), and any **custom uploads** (added automatically when you upload on "
           "the Datasets page). The ● marks the currently loaded session. Regenerate with "
           "`scripts/benchmark_real_dataset.py <dataset>` + "
           "`scripts/benchmark_fraud_detection.py --dataset <dataset>` + "
           "`scripts/benchmark_edge_cases.py` (JSONs saved to `data/benchmarks/`).")

real_rows = []
fraud_conf = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
for entry in validation_entries(ds):
    row = _load_benchmark_entry(entry, ds)
    if row is None:
        continue
    real_rows.append(row)
    if row.pop("no_bench", False):
        continue  # custom upload without benchmarks yet
    # aggregate fraud confusion from the row (all kinds — real, pdf, custom)
    conf = row.pop("_confusion", {})
    for k in fraud_conf:
        fraud_conf[k] += conf.get(k, 0)

if not real_rows:
    st.info("No benchmark results yet — run `scripts/benchmark_real_dataset.py <dataset>` "
            "for each dataset or `scripts/benchmark_edge_cases.py` for the PDF demo graph.")
else:
    bench_rows = [r for r in real_rows if "Retrieval raw" in r]
    df = pd.DataFrame(real_rows)
    if "Retrieval raw" in df.columns:
        df = df.drop(columns=["Retrieval raw", "Pruning raw"])
    st.dataframe(df, width="stretch", hide_index=True)
    n_queries = sum(r["Queries"] for r in bench_rows)
    # weighted across datasets by query count, so the headline never lies
    w_ret = sum(r["Retrieval raw"] * r["Queries"] for r in bench_rows)
    w_prn = sum(r["Pruning raw"] * r["Queries"] for r in bench_rows)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ground-truth queries", f"{n_queries}")
    c2.metric("Retrieval / pruning acc",
              f"{w_ret / n_queries:.0f}% / {w_prn / n_queries:.0f}%")
    if fraud_conf["tp"] + fraud_conf["fp"] + fraud_conf["fn"]:
        denom_p = fraud_conf["tp"] + fraud_conf["fp"]
        denom_r = fraud_conf["tp"] + fraud_conf["fn"]
        prec = fraud_conf["tp"] / denom_p if denom_p else 0
        rec = fraud_conf["tp"] / denom_r if denom_r else 0
        c3.metric("Fraud labels evaluated",
                  f"{fraud_conf['tp'] + fraud_conf['fp'] + fraud_conf['tn'] + fraud_conf['fn']:,}")
        c4.metric("Fraud precision / recall", f"{prec * 100:.0f}% / {rec * 100:.0f}%")
    st.caption("● = session currently loaded in Neo4j (the live query runner below "
               "queries it). Fraud P/R/F1: precision / recall / F1 against the "
               "CSV labels, every fraud claim through the full pipeline.")
    for r in real_rows:
        if "Hint" in r:
            st.caption(f"No benchmarks yet for **{r['Dataset']}** — {r['Hint']}")

# ---------------- live query runner ----------------
st.header("Live query runner")
query = st.text_input("Ask a question about the insurance graph",
                      placeholder="e.g. Does claim CLM-0003 have a fraud flag?")
run = st.button("Run query", type="primary")

if run and query.strip():
    if not neo4j_ok():
        st.error("Neo4j is not reachable.")
    else:
        try:
            with st.spinner("Retrieving, re-ranking, pruning and answering…"):
                res = run_query(get_driver(), query.strip(), max_hops=max_hops,
                                token_budget=token_budget, reranker_mode=reranker_mode,
                                answer_mode=answer_mode)
        except Exception as exc:  # noqa: BLE001 - show a clean message, not a traceback
            st.error(f"Query failed: {exc}")
            st.stop()

        st.subheader("Answer")
        st.write(res["answer"])
        st.caption(f"answer mode: {res['answer_mode']}" +
                   (f" ({res['answer_model']})" if res.get("answer_model") else "") +
                   f" · reranker: {res['reranker']}")
        if res.get("answer_fallback"):
            st.warning(f"LLM answer unavailable — showing extractive answer. "
                       f"Reason: {res['answer_fallback']}")

        t = res["tokens"]
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Baseline tokens", f"{t['before']:,}")
        col_b.metric("Optimized tokens", f"{t['after']:,}")
        col_c.metric("Savings", f"{t['savings_percent']:.1f}%")
        col_d.metric("Latency", f"{res['execution_time_ms']:.0f} ms")

        # one data column (tokens) with the two phases as index labels — the
        # color list must match the column count (a transposed two-column
        # frame would otherwise raise StreamlitColorLengthError)
        st.bar_chart(pd.DataFrame(
            {"tokens": [t["before"], t["after"]]},
            index=["Standard retrieval", "Optimized retrieval"],
        ), color=[ACCENT])

        p = res["pruned"]
        st.markdown(
            f"**Retrieval:** {res['retrieval']['node_count']} nodes / "
            f"{res['retrieval']['edge_count']} edges from seeds "
            f"`{', '.join(res['retrieval']['seeds']) or 'keyword match'}` "
            f"— **pruned to {p['node_count']} nodes** (dropped "
            f"{p['dropped_count']} of {res['retrieval']['node_count']})."
        )
        if p["dropped"]:
            st.markdown("**Pruned (low-relevance) nodes:** " + ", ".join(
                f"`{nid}`" for nid in p["dropped"][:15]))
        if p["kept"]:
            with st.expander("Kept context nodes"):
                st.write(p["kept"])

        # --- fraud ground-truth check (real-dataset validation) ---
        gt_for_query = cached_ground_truth(ds or "synthetic")
        gt_rows = build_comparison(query.strip(), p, res["answer"], gt_for_query)
        if gt_rows:
            st.subheader("🛡️ Fraud ground-truth check")
            st.dataframe(pd.DataFrame(gt_rows).rename(columns={
                "claim": "Claim", "llm_verdict": "Answer says",
                "ground_truth": "Ground truth (dataset)", "check": "Match",
            }), width="stretch", hide_index=True)
            checked = [r for r in gt_rows if r["check"] != "— No verdict"]
            n_ok = sum(1 for r in checked if r["check"] == "✅ Correct")
            c1, c2 = st.columns([1, 3])
            c1.metric("Matches ground truth", f"{n_ok}/{len(checked)}")
            c2.caption(
                f"Ground truth from the `{ds or 'synthetic'}` dataset labels · claims "
                "named in the query are evaluated; context-only claims are listed "
                "for reference. Verdicts are parsed from the answer text, so "
                "\"not determinable\" responses show as no verdict."
            )
        elif gt_for_query:
            st.caption("No claim ids with known fraud ground truth in this query.")
        else:
            st.caption("This dataset has no fraud labels — no ground-truth "
                       "comparison available.")

        # --- explainability (Shot 3) ---
        tr = res["traversal"]
        st.caption(f"Audit record saved: `{tr['audit_id']}` — open the audit "
                   f"trail page with `streamlit run src/graphrag/audit_ui.py`")
        with st.expander("Traversal & Cypher used (explainability)"):
            tms = tr["timings_ms"]
            st.markdown(
                f"**Nodes visited:** {len(tr['nodes_visited'])} · "
                f"**Edges traversed:** {len(tr['edges_traversed'])} · "
                f"**Timings:** retrieval {tms['retrieval_ms']:.0f}ms → rerank "
                f"{tms['rerank_ms']:.0f}ms → prune {tms['prune_ms']:.0f}ms"
            )
            st.code(tr["cypher"], language="cypher")
elif run:
    st.warning("Enter a query first.")

st.caption("Data: synthetic commercial-insurance graph in Neo4j · EXL GraphRAG demo")
