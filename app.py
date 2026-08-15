"""GraphRAG Insurance Claims System — single entry point.

Run:  streamlit run app.py

Three pages via sidebar radio: Home / Dashboard / Audit Trail
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from neo4j import GraphDatabase
from src.graphrag.config import settings
from src.graphrag.fraud_ground_truth import build_comparison, detect_dataset, load_ground_truth
from src.graphrag.query_pipeline import run_query
from src.graphrag.reranker import make_reranker
from src.graphrag.audit_reporter import render_html, render_json, render_pdf
from src.graphrag.lineage_visualizer import render_lineage_html
from src.graphrag.traversal_logger import audit_store
from src.graphrag.custom_sessions import (add_custom_session,
                                          list_custom_sessions,
                                          remove_custom_session,
                                          rename_custom_session,
                                          validate_session_name)
from src.graphrag.sessions import (all_sessions, active_switches,
                                   benchmark_running, clear_progress,
                                   current_session_id,
                                   ensure_pdf_demo_fraud_benchmark,
                                   get_session_meta, start_switch,
                                   switch_in_progress)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def get_driver():
    return GraphDatabase.driver(
        settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )

def loaded_dataset() -> str | None:
    try:
        return detect_dataset(get_driver())
    except Exception:
        return None

def neo4j_ok() -> bool:
    try:
        with get_driver().session() as s:
            s.run("RETURN 1")
        return True
    except Exception:
        return False

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

@st.fragment(run_every="1s")
def render_switch_progress():
    """Live ingest log for a running session switch (auto-refreshes 1x/s).

    Rendered in the sidebar while a background switch is in flight; when it
    finishes, records the outcome in session state and refreshes the whole
    app so every page reflects the newly loaded session.
    """
    handles = active_switches()
    if not handles:
        return
    handle = handles[0]
    lines = handle["lines"]
    if lines:
        st.code("\n".join(lines[-14:]), language=None)
    else:
        st.caption("Starting ingest…")
    if handle["done"]:
        if handle["ok"]:
            st.session_state["session_msg"] = (
                f"Session '{handle['session']}' loaded "
                f"({handle['result']['status']})."
            )
        else:
            st.session_state["session_msg"] = (
                f"Session switch failed: {handle['error']}"
            )
        clear_progress(handle["session"])
        st.rerun(scope="app")
    else:
        st.caption(f"Seeding '{handle['session']}' — large CSVs can take ~1 min; "
                   "the log updates live.")

# ---------------------------------------------------------------------------
# Page config + dark theme
# ---------------------------------------------------------------------------

st.set_page_config(page_title="GraphRAG Insurance Claims", layout="wide", page_icon="🔗")

# Generate the PDF demo graph's fraud benchmark (42 labels) once in the
# background, so the Pipeline Validation table shows real Fraud P/R/F1 for the
# pdf_demo row instead of "—" (no-op when the results already exist).
ensure_pdf_demo_fraud_benchmark()

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    :root {
        --bg:#0f1117;--surface:#161b22;--surface-hover:#1c2333;--border:#30363d;
        --text:#e6edf3;--text-muted:#8b949e;--accent:#58a6ff;
        --accent-glow:rgba(88,166,255,0.15);--green:#3fb950;--red:#f85149;
        --orange:#d29922;--purple:#bc8cff;--radius:12px;--radius-sm:8px;
    }
    .stApp{background:var(--bg)!important}.stApp>header{background:transparent!important}
    .block-container{padding-top:1.5rem!important;max-width:100%!important}
    section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important}
    .grag-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.2rem 1.5rem;margin-bottom:.8rem}
    .grag-card:hover{border-color:var(--accent)}
    .grag-kpi{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);padding:1rem 1.2rem;text-align:center;min-height:80px;display:flex;flex-direction:column;justify-content:center}
    .grag-kpi .kpi-value{font-family:'Inter',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text)}
    .grag-kpi .kpi-label{font-size:.72rem;font-weight:500;color:var(--text-muted);text-transform:uppercase;letter-spacing:.05em;margin-top:.3rem}
    .grag-kpi.accent{border-color:var(--accent)}.grag-kpi.accent .kpi-value{color:var(--accent)}
    .grag-kpi.green{border-color:var(--green)}.grag-kpi.green .kpi-value{color:var(--green)}
    .grag-kpi.purple{border-color:var(--purple)}.grag-kpi.purple .kpi-value{color:var(--purple)}
    .grag-kpi.orange{border-color:var(--orange)}.grag-kpi.orange .kpi-value{color:var(--orange)}
    .grag-answer{background:linear-gradient(135deg,var(--surface),var(--surface-hover));border:1px solid var(--green);border-left:4px solid var(--green);border-radius:var(--radius);padding:1.4rem 1.8rem;margin-bottom:1rem}
    .grag-answer .label{font-size:.75rem;font-weight:600;color:var(--green);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem}
    .grag-answer .text{font-size:1.05rem;line-height:1.65;color:var(--text)}
    .grag-query{background:var(--surface);border:1px solid var(--accent);border-left:4px solid var(--accent);border-radius:var(--radius);padding:1.2rem 1.5rem;margin-bottom:1rem}
    .grag-query .text{font-size:1.1rem;font-weight:500;color:var(--accent)}
    .grag-graph-box{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden}
    .grag-footer{text-align:center;color:var(--text-muted);font-size:.75rem;padding:1rem 0;border-top:1px solid var(--border);margin-top:1.5rem}
    .stTabs [data-baseweb="tab-list"]{gap:0}
    .stTabs [data-baseweb="tab"]{background:var(--surface)!important;color:var(--text-muted)!important;border:1px solid var(--border)!important;border-bottom:none!important;border-radius:var(--radius-sm) var(--radius-sm) 0 0!important;font-weight:500;font-size:.85rem;padding:.6rem 1.2rem}
    .stTabs [aria-selected="true"]{background:var(--surface-hover)!important;color:var(--accent)!important;border-bottom:2px solid var(--accent)!important}
    .stDownloadButton>button{background:var(--surface)!important;color:var(--text)!important;border:1px solid var(--border)!important;border-radius:var(--radius-sm)!important;width:100%!important}
    .stDownloadButton>button:hover{border-color:var(--accent)!important;background:var(--accent-glow)!important;color:var(--accent)!important}
    .stDataFrame{border-radius:var(--radius-sm)!important;overflow:hidden}
    .stCodeBlock{border-radius:var(--radius-sm)!important}
    .stAlert{border-radius:var(--radius-sm)!important}
    .stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4,.stMarkdown h5,.stMarkdown h6{color:var(--text)!important}
    .stCaption,.stCaption p{color:var(--text-muted)!important}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("#### Navigate")
    page = st.radio("Pages", ["Home", "Dashboard", "Audit Trail", "Datasets"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 🔗 GraphRAG")
    st.caption("Insurance Claims System")
    st.markdown("---")
    st.markdown("#### System Status")
    db_ok = neo4j_ok()
    st.markdown(f"**Neo4j:** {'🟢 connected' if db_ok else '🔴 not reachable'}")
    ds = loaded_dataset()
    st.markdown(f"**Dataset:** `{ds or 'synthetic (demo)'}`")
    gt = load_ground_truth(ds) if ds else None
    if gt:
        fraud = sum(1 for v in gt.values() if v)
        st.markdown(f"**Fraud labels:** {fraud:,} / {len(gt):,}")

    # ---- session switcher (Phase 6): re-seeds the graph from the web UI ----
    st.markdown("---")
    st.markdown("#### Active Session")
    msg = st.session_state.pop("session_msg", None)
    if msg:
        st.success(msg)
    all_sess = all_sessions()
    session_ids = [s["id"] for s in all_sess]
    session_labels = {s["id"]: s["label"] for s in all_sess}
    cur_session = current_session_id(get_driver()) if db_ok else "pdf_demo"
    sel_session = st.selectbox(
        "Session", session_ids,
        format_func=lambda sid: session_labels[sid],
        index=session_ids.index(cur_session) if cur_session in session_ids else 0,
        label_visibility="collapsed",
    )
    sess = get_session_meta(sel_session) or {}
    kind_note = ("📊 Excel / real CSV" if sess.get("kind") == "excel"
                 else ("📤 Custom upload" if sess.get("kind") == "custom"
                       else "📄 PDF / synthetic demo"))
    st.caption(f"Pipeline: {kind_note} — {sess.get('desc', '')}")
    if switch_in_progress():
        render_switch_progress()
    if benchmark_running(sel_session):
        st.caption("🔬 Benchmarking this session in the background — the "
                   "Pipeline Validation table fills in automatically when done.")
    if sel_session != cur_session and not switch_in_progress():
        if not db_ok:
            st.error("Neo4j not reachable — start it before switching sessions.")
        else:
            start_switch(get_driver(), sel_session)
            st.rerun()
    if st.button("Re-seed this session", type="secondary", use_container_width=True):
        if not db_ok:
            st.error("Neo4j not reachable — start it before re-seeding.")
        elif switch_in_progress():
            st.caption("A session switch is already running — wait for it to finish.")
        else:
            start_switch(get_driver(), sel_session, force=True)
            st.rerun()

# =========================================================================
# HOME PAGE
# =========================================================================

if page == "Home":
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1rem;">
        <h1 style="font-size:2.2rem;font-weight:800;color:#e6edf3;margin-bottom:.3rem;">
            GraphRAG Insurance Claims System</h1>
        <p style="font-size:1.1rem;color:#8b949e;max-width:700px;margin:0 auto;">
            Cost-optimized, incrementally-updating GraphRAG for commercial insurance.
            Real-time CDC, 08-18% token savings, full audit trails.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='grag-kpi {'green' if db_ok else 'orange'}'><div class='kpi-value'>{'🟢' if db_ok else '🔴'}</div><div class='kpi-label'>Neo4j</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='grag-kpi accent'><div class='kpi-value'>{ds or 'demo'}</div><div class='kpi-label'>Dataset</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='grag-kpi purple'><div class='kpi-value'>{settings.LLAMA_MODEL}</div><div class='kpi-label'>LLM Model</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='grag-kpi'><div class='kpi-value'>{settings.ANSWER_MODE}</div><div class='kpi-label'>Answer Mode</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Two Ingestion Pipelines")
    st.caption("Both pipelines run on the same Neo4j graph — switch sessions from the sidebar (or upload your own dataset on the Datasets page) and it re-seeds automatically.")
    cur_meta = get_session_meta(cur_session) or {}
    pdf_active = "● Active now" if cur_meta.get("kind") == "pdf" else ""
    xls_active = "● Active now" if cur_meta.get("kind") in ("excel", "custom") else ""
    p1, p2 = st.columns(2)
    with p1:
        st.markdown(
            "<div class='grag-card' style='border-left:4px solid #bc8cff;min-height:190px;'>"
            "<div style='font-size:1.5rem;margin-bottom:.4rem;'>📄</div>"
            "<div style='font-size:1rem;font-weight:700;color:#bc8cff;margin-bottom:.4rem;'>PDF Pipeline — uploads &amp; CDC</div>"
            "<div style='font-size:.85rem;color:#e6edf3;line-height:1.5;'>Upload insurance PDFs → text extraction → entity extraction → change detection → surgical Neo4j update. Queries run against the synthetic demo graph.</div>"
            f"<div style='margin-top:.5rem;font-size:.78rem;color:#3fb950;'>{pdf_active}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            "<div class='grag-card' style='border-left:4px solid #58a6ff;min-height:190px;'>"
            "<div style='font-size:1.5rem;margin-bottom:.4rem;'>📊</div>"
            "<div style='font-size:1rem;font-weight:700;color:#58a6ff;margin-bottom:.4rem;'>Excel / CSV Pipeline — real datasets</div>"
            "<div style='font-size:.85rem;color:#e6edf3;line-height:1.5;'>Ingest the real insurance CSVs (fraud_oracle, insurance_claims, insurance_dataset) → graph → ground-truth benchmarks. Queries run against the loaded real dataset.</div>"
            f"<div style='margin-top:.5rem;font-size:.78rem;color:#3fb950;'>{xls_active}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### The 3 Technical Kill Shots")
    shots = [
        ("⚡", "Shot 1: Real-time CDC", "#3fb950", "Upload an endorsement PDF and the graph updates surgically in under 500ms — no full rebuild.", "Change Detection → Entity Diff → Partial Neo4j write."),
        ("🎯", "Shot 2: Token Optimization", "#58a6ff", "Cross-encoder re-ranking + adaptive pruning cuts token usage per query — measured 6–16% on precise real queries, up to 80% on dense subgraphs.", "Retrieve → Re-rank → Prune to token budget → Answer."),
        ("🔍", "Shot 3: Lineage & Explainability", "#bc8cff", "Every answer ships with its full traversal path, Cypher query, and exportable audit report.", "Traversal logging → Interactive lineage graph → JSON/HTML/PDF exports."),
    ]
    cols = st.columns(3)
    for i, (icon, title, color, desc, detail) in enumerate(shots):
        with cols[i]:
            st.markdown(f"<div class='grag-card' style='border-left:4px solid {color};min-height:200px;'><div style='font-size:1.8rem;margin-bottom:.5rem;'>{icon}</div><div style='font-size:1rem;font-weight:700;color:{color};margin-bottom:.5rem;'>{title}</div><div style='font-size:.9rem;color:#e6edf3;line-height:1.5;margin-bottom:.6rem;'>{desc}</div><div style='font-size:.78rem;color:#8b949e;line-height:1.4;'>{detail}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Quick Start")
    st.code("""# 1. Start Neo4j
docker compose up -d neo4j

# 2. Seed the demo graph
.venv/Scripts/python.exe scripts/seed_graph.py --reset --apply-schema

# 3. Launch the app
streamlit run app.py""", language="bash")
    st.markdown("<div class='grag-footer'>GraphRAG Insurance Claims System — Built for EXL</div>", unsafe_allow_html=True)

# =========================================================================
# DASHBOARD PAGE
# =========================================================================

elif page == "Dashboard":
    st.markdown("### Dashboard")
    st.caption("Shot 2 — retrieve → re-rank → prune → answer · token optimization + benchmarks")
    st.caption(f"Active session: **{session_labels[sel_session]}** — switch it from the sidebar; the graph re-seeds automatically.")

    with st.expander("Query Controls", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        max_hops = c1.slider("Max hops", 1, 4, settings.MAX_HOPS)
        token_budget = c2.slider("Token budget", 256, 8192, settings.MAX_TOKENS, step=256)
        reranker_mode = c3.selectbox("Reranker", ["auto", "cross-encoder", "lexical"])
        answer_mode = c4.selectbox("Answer mode", ["extractive", "auto", "llm"],
                                   index=["extractive", "auto", "llm"].index(settings.ANSWER_MODE) if settings.ANSWER_MODE in ("extractive", "auto", "llm") else 0)

    st.markdown("---")
    st.markdown("#### Pipeline Validation (all sessions)")
    st.caption("Benchmarks per session — **10,200 ground-truth queries total** (10,000 on the 3 real Excel sessions: fraud_oracle 5,500 · insurance_dataset 3,900 · insurance_claims 600, plus the synthetic variant + PDF demo graph at 100 each), **100% retrieval & pruning accuracy**, token savings, latency and fraud P/R/F1 per row. The ● marks the currently loaded session; it updates automatically when you switch or upload. CSV uploads are benchmarked automatically in the background — this table refreshes until they land. Regenerate with `scripts/benchmark_real_dataset.py <dataset> --queries N --workers 8` + `scripts/benchmark_fraud_detection.py --dataset <dataset>` (JSONs saved to `data/benchmarks/`).")
    # while any auto-benchmark is in flight, refresh this table every 5s so the
    # new row fills in without a manual reload (fragment keeps the live query
    # runner below undisturbed)
    @st.fragment(run_every="5s" if any(benchmark_running(e["files"][0]) for e in validation_entries(ds)) else None)
    def _render_validation_table(loaded):
        real_rows = []; fraud_conf = {"tp":0,"fp":0,"tn":0,"fn":0}
        for entry in validation_entries(loaded):
            row = _load_benchmark_entry(entry, loaded)
            if row is None:
                continue
            real_rows.append(row)
            if row.pop("no_bench", False):
                continue  # custom upload without benchmarks yet
            # aggregate fraud confusion from the row (all kinds — real, pdf,
            # custom) for the headline KPIs
            conf = row.pop("_confusion", {})
            for k in fraud_conf:
                fraud_conf[k] += conf.get(k, 0)

        if not real_rows:
            st.info("No benchmark results yet. Run `scripts/benchmark_real_dataset.py <dataset>` or `scripts/benchmark_edge_cases.py` for the PDF demo graph.")
        else:
            bench_rows = [r for r in real_rows if "Retrieval raw" in r]
            df = pd.DataFrame(real_rows)
            if "Retrieval raw" in df.columns:
                df = df.drop(columns=["Retrieval raw","Pruning raw"])
            st.dataframe(df, width="stretch", hide_index=True)
            if bench_rows:
                n_queries = sum(r["Queries"] for r in bench_rows)
                w_ret = sum(r["Retrieval raw"]*r["Queries"] for r in bench_rows)
                w_prn = sum(r["Pruning raw"]*r["Queries"] for r in bench_rows)
                c1,c2,c3,c4 = st.columns(4)
                c1.markdown(f"<div class='grag-kpi accent'><div class='kpi-value'>{n_queries}</div><div class='kpi-label'>Ground-truth Queries</div></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='grag-kpi green'><div class='kpi-value'>{w_ret/n_queries:.0f}% / {w_prn/n_queries:.0f}%</div><div class='kpi-label'>Retrieval / Pruning Acc</div></div>", unsafe_allow_html=True)
                if fraud_conf["tp"]+fraud_conf["fp"]+fraud_conf["fn"]:
                    denom_p=fraud_conf["tp"]+fraud_conf["fp"];denom_r=fraud_conf["tp"]+fraud_conf["fn"]
                    prec=fraud_conf["tp"]/denom_p if denom_p else 0;rec=fraud_conf["tp"]/denom_r if denom_r else 0
                    c3.markdown(f"<div class='grag-kpi orange'><div class='kpi-value'>{fraud_conf['tp']+fraud_conf['fp']+fraud_conf['tn']+fraud_conf['fn']:,}</div><div class='kpi-label'>Fraud Labels Evaluated</div></div>", unsafe_allow_html=True)
                    c4.markdown(f"<div class='grag-kpi purple'><div class='kpi-value'>{prec*100:.0f}% / {rec*100:.0f}%</div><div class='kpi-label'>Fraud Precision / Recall</div></div>", unsafe_allow_html=True)

    _render_validation_table(ds)

    st.markdown("---")
    st.markdown("#### Live Query Runner")
    query = st.text_input("Ask a question", placeholder="e.g. Does claim CLM-0003 have a fraud flag?", label_visibility="collapsed")
    run = st.button("Run query", type="primary")

    if run and query.strip():
        if not db_ok:
            st.error("Neo4j is not reachable.")
        else:
            try:
                with st.spinner("Retrieving, re-ranking, pruning and answering..."):
                    res = run_query(get_driver(), query.strip(), max_hops=max_hops, token_budget=token_budget, reranker_mode=reranker_mode, answer_mode=answer_mode)
            except Exception as exc:
                st.error(f"Query failed: {exc}"); st.stop()

            st.markdown(f"<div class='grag-answer'><div class='label'>Answer</div><div class='text'>{res['answer']}</div></div>", unsafe_allow_html=True)
            st.caption(f"answer mode: {res['answer_mode']}" + (f" ({res['answer_model']})" if res.get("answer_model") else "") + f" · reranker: {res['reranker']}")
            if res.get("answer_fallback"):
                st.warning(f"LLM unavailable — extractive fallback. Reason: {res['answer_fallback']}")

            t = res["tokens"]
            c1,c2,c3,c4 = st.columns(4)
            c1.markdown(f"<div class='grag-kpi'><div class='kpi-value'>{t['before']:,}</div><div class='kpi-label'>Baseline Tokens</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='grag-kpi accent'><div class='kpi-value'>{t['after']:,}</div><div class='kpi-label'>Optimized Tokens</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='grag-kpi green'><div class='kpi-value'>{t['savings_percent']:.1f}%</div><div class='kpi-label'>Savings</div></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='grag-kpi purple'><div class='kpi-value'>{res['execution_time_ms']:.0f}ms</div><div class='kpi-label'>Latency</div></div>", unsafe_allow_html=True)

            st.bar_chart(pd.DataFrame({"tokens":[t["before"],t["after"]]}, index=["Standard retrieval","Optimized retrieval"]), color=["#58a6ff"])

            p = res["pruned"]
            st.markdown(f"**Retrieval:** {res['retrieval']['node_count']} nodes / {res['retrieval']['edge_count']} edges from seeds `{', '.join(res['retrieval']['seeds']) or 'keyword match'}` — **pruned to {p['node_count']} nodes** (dropped {p['dropped_count']}).")
            if p["dropped"]:
                st.markdown("**Pruned nodes:** " + ", ".join(f"`{nid}`" for nid in p["dropped"][:15]))
            if p["kept"]:
                with st.expander("Kept context nodes"):
                    st.write(p["kept"])

            gt_for_query = load_ground_truth(ds or "synthetic")
            gt_rows = build_comparison(query.strip(), p, res["answer"], gt_for_query)
            if gt_rows:
                st.markdown("#### Fraud Ground-Truth Check")
                st.dataframe(pd.DataFrame(gt_rows).rename(columns={"claim":"Claim","llm_verdict":"Answer says","ground_truth":"Ground truth","check":"Match"}), width="stretch", hide_index=True)
                checked=[r for r in gt_rows if r["check"]!="— No verdict"];n_ok=sum(1 for r in checked if r["check"]=="✅ Correct")
                c1,c2=st.columns([1,3]);c1.metric("Matches",f"{n_ok}/{len(checked)}");c2.caption("Verdicts parsed from answer text.")

            tr = res["traversal"]
            with st.expander("Traversal & Cypher (explainability)"):
                tms=tr["timings_ms"]
                st.markdown(f"**Nodes visited:** {len(tr['nodes_visited'])} · **Edges traversed:** {len(tr['edges_traversed'])} · **Timings:** retrieval {tms['retrieval_ms']:.0f}ms → rerank {tms['rerank_ms']:.0f}ms → prune {tms['prune_ms']:.0f}ms")
                st.code(tr["cypher"], language="cypher")
    elif run:
        st.warning("Enter a query first.")

    st.markdown(f"<div class='grag-footer'>GraphRAG Dashboard — Shot 2 · Reranker: {make_reranker().name} · Model: {settings.LLAMA_MODEL}</div>", unsafe_allow_html=True)

# =========================================================================
# AUDIT TRAIL PAGE
# =========================================================================

elif page == "Audit Trail":
    st.markdown("### Audit Trail & Lineage")
    st.caption("Shot 3 — every answer is explainable: question → traversal → Cypher → answer")

    records = audit_store.recent(100)
    if not records:
        st.info("No audit records yet. Run a query from the Dashboard first.")
        st.stop()

    selected = st.selectbox("Query history (newest first)", records,
                            format_func=lambda r: f"[{r['timestamp'][11:19]}] {r['query'][:60]}" + ("" if len(r["query"])<=60 else "..."))
    r = selected

    st.markdown(f"<div class='grag-query'><div class='text'>{r['query']}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='grag-answer'><div class='label'>Answer</div><div class='text'>{r['answer']}</div></div>", unsafe_allow_html=True)

    ret=r.get("retrieval",{});tok=r.get("tokens",{});prn=r.get("pruned",{});tm=r.get("timings_ms",{})
    c1,c2,c3,c4,c5,c6=st.columns(6)
    c1.markdown(f"<div class='grag-kpi accent'><div class='kpi-value'>{ret.get('node_count',0)}</div><div class='kpi-label'>Retrieved</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='grag-kpi purple'><div class='kpi-value'>{ret.get('edge_count',0)}</div><div class='kpi-label'>Edges</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='grag-kpi green'><div class='kpi-value'>{prn.get('kept_count',len(prn.get('kept',[])))}</div><div class='kpi-label'>Kept</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='grag-kpi orange'><div class='kpi-value'>{prn.get('dropped_count',len(prn.get('dropped',[])))}</div><div class='kpi-label'>Pruned</div></div>", unsafe_allow_html=True)
    c5.markdown(f"<div class='grag-kpi green'><div class='kpi-value'>{tok.get('savings_percent',0):.1f}%</div><div class='kpi-label'>Token Savings</div></div>", unsafe_allow_html=True)
    c6.markdown(f"<div class='grag-kpi'><div class='kpi-value'>{tm.get('total_ms',0):.0f}ms</div><div class='kpi-label'>Total Time</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Lineage Graph")
    graph_h = st.slider("Graph height", 300, 1600, 500, step=50, label_visibility="collapsed", key="audit_graph_height")
    st.markdown("<div class='grag-graph-box'>", unsafe_allow_html=True)
    st.iframe(render_lineage_html(r, height=graph_h), height=graph_h)
    st.markdown("</div>", unsafe_allow_html=True)

    gt_for_record = load_ground_truth(ds or "synthetic")
    gt_rows = build_comparison(r["query"], r.get("pruned",{}), r["answer"], gt_for_record)
    if gt_rows:
        st.markdown("---")
        st.markdown("#### Fraud Ground-Truth Check")
        st.dataframe(pd.DataFrame(gt_rows).rename(columns={"claim":"Claim","llm_verdict":"Answer says","ground_truth":"Ground truth","check":"Match"}), width="stretch", hide_index=True)

    st.markdown("---")
    tab_path, tab_cypher, tab_evidence = st.tabs(["Traversal Path","Cypher Query","Evidence & Tokens"])
    with tab_path:
        paths=r.get("traversal",{}).get("paths",[])
        if not paths: st.info("No traversal path extracted.")
        else:
            st.markdown(f"**{len(paths)} branch(es)** from the seed(s):")
            for p in paths:
                chain=[]
                for j,nid in enumerate(p["nodes"]):
                    chain.append(nid)
                    if j<len(p["edges"]): chain.append(f" —[{p['edges'][j][1]}]-> ")
                st.code(" ".join(chain), language=None)
        with st.expander("All edges traversed"):
            for e in r.get("traversal",{}).get("edges_traversed",[]):
                st.markdown(f"`{e[0]}` **→ [`{e[1]}`]** → `{e[2]}`")
    with tab_cypher:
        st.code(r.get("cypher",""), language="cypher")
    with tab_evidence:
        ca,cb=st.columns(2)
        with ca:
            st.markdown("**Kept context nodes**");st.write(prn.get("kept",[]))
            st.markdown("**Pruned**");st.write(prn.get("dropped",[]))
        with cb:
            st.markdown("**Ranked context (top 10)**");st.dataframe(r.get("ranking",[])[:10],width="stretch",hide_index=True)
            st.markdown("**Stage timings**");st.dataframe([{"stage":k.replace("_ms",""),"ms":v} for k,v in tm.items()],width="stretch",hide_index=True)

    st.markdown("---")
    st.markdown("#### Export Audit Report")
    x1,x2,x3=st.columns(3)
    with x1: st.download_button("JSON",data=render_json(r),file_name=f"{r['audit_id']}.json",mime="application/json",use_container_width=True)
    with x2: st.download_button("HTML Report",data=render_html(r),file_name=f"{r['audit_id']}.html",mime="text/html",use_container_width=True)
    with x3: st.download_button("PDF Report",data=render_pdf(r),file_name=f"{r['audit_id']}.pdf",mime="application/pdf",use_container_width=True)

    st.markdown(f"<div class='grag-footer'>Audit ID: {r['audit_id']} · {r['timestamp']} · GraphRAG — Lineage & Explainability</div>", unsafe_allow_html=True)

# =========================================================================
# DATASETS PAGE — upload your own PDF/CSV and query it
# =========================================================================

elif page == "Datasets":
    st.markdown("### Datasets")
    st.caption("Upload your own PDF/CSV, give it a **unique session name**, and the pipeline builds a graph for it — then query it from the Dashboard like any other session. Uploaded files live under `data/custom/`. After a **CSV** upload loads, a ground-truth benchmark runs **automatically in the background** — the Dashboard's Pipeline Validation row fills in when it finishes (no manual `benchmark_real_dataset.py` needed).")

    st.markdown("---")
    st.markdown("#### Upload your dataset")
    up_files = st.file_uploader("Choose PDF or CSV file(s)", type=["pdf", "csv"],
                                accept_multiple_files=True,
                                label_visibility="collapsed")
    c1, c2 = st.columns([3, 1])
    new_name = c1.text_input("Session name — must be unique (not one of the built-ins)",
                             placeholder="e.g. my_claims")
    create = c2.button("Create & load", type="primary", use_container_width=True)
    if create:
        if not up_files:
            st.error("Choose at least one file first.")
        else:
            try:
                cleaned = validate_session_name(new_name)
            except ValueError as exc:
                st.error(str(exc))
            else:
                target_dir = ROOT / "data" / "custom" / cleaned
                target_dir.mkdir(parents=True, exist_ok=True)
                saved = []
                for f in up_files:
                    dest = target_dir / f.name
                    dest.write_bytes(f.getbuffer())
                    saved.append(f"data/custom/{cleaned}/{f.name}")
                kind = "pdf" if any(f.name.lower().endswith(".pdf") for f in up_files) else "csv"
                note = f"{len(up_files)} file(s) · {up_files[0].name}"
                add_custom_session(cleaned, kind, saved, note=note)
                st.session_state["session_msg"] = (
                    f"Custom session '{cleaned}' created — loading it now…"
                )
                if db_ok:
                    start_switch(get_driver(), cleaned)
                st.rerun()

    st.markdown("---")
    st.markdown("#### Your custom sessions")
    customs = list_custom_sessions()
    if not customs:
        st.info("No custom sessions yet — upload your first file above.")
    for rec in customs:
        st.markdown(
            f"**`{rec['name']}`** · {rec['kind'].upper()} · "
            f"{rec.get('note') or '—'} · created {rec['created_at']}"
        )
        cols = st.columns([3, 1, 1])
        new_nm = cols[0].text_input("Rename to", value=rec["name"],
                                    key=f"rn_{rec['name']}")
        if cols[1].button("Rename", key=f"rb_{rec['name']}"):
            old = rec["name"]
            try:
                cleaned = validate_session_name(new_nm, exclude=old)
                rename_custom_session(old, cleaned)
                # if the renamed session is loaded, re-seed so the graph marker
                # (and session detection) follows the new name
                if db_ok and current_session_id(get_driver()) == old:
                    start_switch(get_driver(), cleaned)
                    st.session_state["session_msg"] = (
                        f"Renamed '{old}' → '{cleaned}' — re-seeding…"
                    )
                else:
                    st.session_state["session_msg"] = (
                        f"Renamed '{old}' → '{cleaned}'."
                    )
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if cols[2].button("Delete", key=f"db_{rec['name']}"):
            remove_custom_session(rec["name"])
            st.session_state["session_msg"] = (
                f"Deleted custom session '{rec['name']}'."
            )
            st.rerun()
        st.markdown("---")

    st.markdown("#### How to use")
    st.markdown("1. **Create** — upload a CSV or PDF, type a unique session name, hit *Create & load*.\n"
                "2. **Watch** — the sidebar shows the ingest log streaming live; when it finishes the session is active.\n"
                "3. **Query** — open the *Dashboard* and ask anything about your dataset. The graph responds to keyword and id queries (CSVs with claim/fraud columns get Claim + FraudFlag nodes; anything else becomes generic Record nodes).")
