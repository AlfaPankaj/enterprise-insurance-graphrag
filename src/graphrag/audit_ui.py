"""GraphRAG Audit Trail & Lineage UI (Shot 3) — Streamlit app.

Run:  streamlit run src/graphrag/audit_ui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from neo4j import GraphDatabase  # noqa: E402

from graphrag.audit_reporter import render_html, render_json, render_pdf  # noqa: E402
from graphrag.config import settings  # noqa: E402
from graphrag.fraud_ground_truth import (  # noqa: E402
    build_comparison,
    detect_dataset,
    load_ground_truth,
)
from graphrag.lineage_visualizer import render_lineage_html  # noqa: E402
from graphrag.traversal_logger import audit_store  # noqa: E402

# ---------------------------------------------------------------------------
# Common theme template — shared across dashboard + audit_ui
# ---------------------------------------------------------------------------

st.set_page_config(page_title="GraphRAG Audit Trail & Lineage", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bg: #0f1117;
        --surface: #161b22;
        --surface-hover: #1c2333;
        --border: #30363d;
        --border-light: #21262d;
        --text: #e6edf3;
        --text-muted: #8b949e;
        --accent: #58a6ff;
        --accent-glow: rgba(88, 166, 255, 0.15);
        --green: #3fb950;
        --green-bg: rgba(63, 185, 80, 0.12);
        --red: #f85149;
        --red-bg: rgba(248, 81, 73, 0.12);
        --orange: #d29922;
        --orange-bg: rgba(210, 153, 34, 0.12);
        --purple: #bc8cff;
        --purple-bg: rgba(188, 140, 255, 0.12);
        --radius: 12px;
        --radius-sm: 8px;
    }

    /* Global resets */
    .stApp { background: var(--bg) !important; }
    .stApp > header { background: transparent !important; }
    .block-container { padding-top: 1.5rem !important; max-width: 100% !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 { color: var(--text) !important; }

    /* Cards */
    .grag-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s;
    }
    .grag-card:hover { border-color: var(--accent); }

    .grag-card-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.6rem;
    }
    .grag-card-header .icon {
        font-size: 1.1rem;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        flex-shrink: 0;
    }
    .grag-card-header .title {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* Answer card */
    .grag-answer {
        background: linear-gradient(135deg, var(--surface) 0%, var(--surface-hover) 100%);
        border: 1px solid var(--green);
        border-left: 4px solid var(--green);
        border-radius: var(--radius);
        padding: 1.4rem 1.8rem;
        margin-bottom: 1rem;
    }
    .grag-answer .label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--green);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    .grag-answer .text {
        font-size: 1.05rem;
        line-height: 1.65;
        color: var(--text);
    }

    /* Query card */
    .grag-query {
        background: var(--surface);
        border: 1px solid var(--accent);
        border-left: 4px solid var(--accent);
        border-radius: var(--radius);
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .grag-query .text {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 500;
        color: var(--accent);
        line-height: 1.5;
    }

    /* KPI metric cards */
    .grag-kpi {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 1rem 1.2rem;
        text-align: center;
        min-height: 80px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .grag-kpi .kpi-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1.2;
    }
    .grag-kpi .kpi-label {
        font-size: 0.72rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.3rem;
    }

    /* Accent color KPI */
    .grag-kpi.accent { border-color: var(--accent); }
    .grag-kpi.accent .kpi-value { color: var(--accent); }
    .grag-kpi.green { border-color: var(--green); }
    .grag-kpi.green .kpi-value { color: var(--green); }
    .grag-kpi.purple { border-color: var(--purple); }
    .grag-kpi.purple .kpi-value { color: var(--purple); }
    .grag-kpi.orange { border-color: var(--orange); }
    .grag-kpi.orange .kpi-value { color: var(--orange); }

    /* Graph container */
    .grag-graph-box {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        overflow: hidden;
        font-weight: 1000;
        font-size: 1.85rem;
        padding: 1.6rem 2.2rem;
    }

    /* Tabs override */
    .stTabs [data-baseweb="tab-list"] { gap: 0; }
    .stTabs [data-baseweb="tab"] {
        background: var(--surface) !important;
        color: var(--text-muted) !important;
        border: 1px solid var(--border) !important;
        border-bottom: none !important;
        border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 0.6rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background: var(--surface-hover) !important;
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
    }

    /* Download buttons */
    .stDownloadButton > button {
        background: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        width: 100% !important;
        transition: all 0.2s !important;
    }
    .stDownloadButton > button:hover {
        border-color: var(--accent) !important;
        background: var(--accent-glow) !important;
        color: var(--accent) !important;
    }

    /* Tables */
    .stDataFrame { border-radius: var(--radius-sm) !important; overflow: hidden; }
    .stDataFrame table { border-color: var(--border) !important; }
    .stDataFrame th { background: var(--surface-hover) !important; }

    /* Code blocks */
    .stCodeBlock { border-radius: var(--radius-sm) !important; }
    .stCodeBlock pre { background: var(--surface) !important; }

    /* Expander */
    .streamlit-expanderHeader {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }

    /* Warning/info boxes */
    .stAlert { border-radius: var(--radius-sm) !important; }

    /* Selectbox / input overrides */
    .stSelectbox > div > div,
    .stTextInput > div > div > input {
        background: var(--surface) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }

    /* Caption */
    .stCaption, .stCaption p { color: var(--text-muted) !important; }

    /* Subheader / header */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 { color: var(--text) !important; }

    /* Footer */
    .grag-footer {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.75rem;
        padding: 1rem 0;
        border-top: 1px solid var(--border);
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


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


def kpi_card(value: str, label: str, css_class: str = "") -> str:
    return (f"<div class='grag-kpi {css_class}'>"
            f"<div class='kpi-value'>{value}</div>"
            f"<div class='kpi-label'>{label}</div></div>")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Audit Trail")
    st.caption("Shot 3 — explainability for every query")

    ds = loaded_dataset()
    gt = load_ground_truth(ds) if ds else None

    st.markdown("---")
    st.markdown("#### Dataset Status")
    st.markdown(f"**Loaded:** `{ds or 'synthetic (demo)'}`")
    if gt:
        fraud = sum(1 for v in gt.values() if v)
        st.markdown(f"**Fraud labels:** {fraud:,} / {len(gt):,}")

    records = audit_store.recent(100)
    st.markdown(f"**Records:** {len(records)}")
    st.caption("Appended by every query (dashboard / API).")

    st.markdown("---")
    st.markdown("#### Query Selector")
    if not records:
        st.info("No records yet.")
        st.stop()

    selected = st.selectbox(
        "Pick a query",
        records,
        format_func=lambda r: f"[{r['timestamp'][11:19]}] {r['query'][:50]}"
                              + ("" if len(r["query"]) <= 50 else "..."),
        label_visibility="collapsed",
    )

    st.markdown("---")
    if st.button("Clear audit trail", type="secondary", use_container_width=True):
        audit_store.clear()
        st.rerun()


r = selected

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown("### Audit Trail & Lineage")
st.caption("Shot 3 — every answer is explainable: question → traversal → Cypher → answer")

# ---- Query card ----
st.markdown(
    f"<div class='grag-query'>"
    f"<div class='text'>{r['query']}</div>"
    f"</div>",
    unsafe_allow_html=True,
)

# ---- Answer card ----
st.markdown(
    f"<div class='grag-answer'>"
    f"<div class='label'>Answer</div>"
    f"<div class='text'>{r['answer']}</div>"
    f"</div>",
    unsafe_allow_html=True,
)

# ---- KPI row ----
ret = r.get("retrieval", {})
tok = r.get("tokens", {})
prn = r.get("pruned", {})
tm = r.get("timings_ms", {})

seeds_display = ", ".join(ret.get("seeds", [])) or "keyword"
node_before = ret.get("node_count", 0)
edge_before = ret.get("edge_count", 0)
node_after = prn.get("kept_count", len(prn.get("kept", [])))
dropped = prn.get("dropped_count", len(prn.get("dropped", [])))
tokens_before = tok.get("before", 0)
tokens_after = tok.get("after", 0)
savings_pct = tok.get("savings_percent", 0)
total_ms = tm.get("total_ms", 0)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.markdown(kpi_card(f"{node_before}", "Nodes Retrieved", "accent"), unsafe_allow_html=True)
c2.markdown(kpi_card(f"{edge_before}", "Edges Traversed", "purple"), unsafe_allow_html=True)
c3.markdown(kpi_card(f"{node_after}", "Kept", "green"), unsafe_allow_html=True)
c4.markdown(kpi_card(f"{dropped}", "Pruned", "orange"), unsafe_allow_html=True)
c5.markdown(kpi_card(f"{savings_pct:.1f}%", "Token Savings", "green"), unsafe_allow_html=True)
c6.markdown(kpi_card(f"{total_ms:.0f}ms", "Total Time", ""), unsafe_allow_html=True)

# ---- LINEAGE GRAPH — first, big, fills the page ----
st.markdown("---")
st.markdown("#### Lineage Graph")
st.caption("Interactive — drag nodes, hover for details. Bold border = seed nodes. "
           "Faded = pruned by token budget.")

graph_h = st.slider("Graph height", 300, 1000, 500, step=50,
                     label_visibility="collapsed",
                     key="audit_graph_height")

st.markdown("<div class='grag-graph-box'>", unsafe_allow_html=True)
st.iframe(render_lineage_html(r, height=graph_h), height=graph_h)
st.markdown("</div>", unsafe_allow_html=True)

# ---- Fraud ground-truth check ----
gt_for_record = load_ground_truth(ds or "synthetic")
gt_rows = build_comparison(r["query"], r.get("pruned", {}), r["answer"], gt_for_record)
if gt_rows:
    st.markdown("---")
    st.markdown("#### Fraud Ground-Truth Check")
    st.caption("Labels of the currently loaded dataset")
    st.dataframe(
        pd.DataFrame(gt_rows).rename(columns={
            "claim": "Claim", "llm_verdict": "Answer says",
            "ground_truth": "Ground truth (dataset)", "check": "Match",
        }),
        width="stretch",
        hide_index=True,
    )

# ---- Tabs: path / cypher / evidence ----
st.markdown("---")
tab_path, tab_cypher, tab_evidence = st.tabs(
    ["Traversal Path", "Cypher Query", "Evidence & Tokens"]
)

with tab_path:
    paths = r.get("traversal", {}).get("paths", [])
    if not paths:
        st.info("No traversal path extracted (no seed matched the query).")
    else:
        st.markdown(f"**{len(paths)} branch(es)** from the seed(s):")
        for i, p in enumerate(paths):
            chain = []
            nodes_list, edges_list = p["nodes"], p["edges"]
            for j, nid in enumerate(nodes_list):
                chain.append(nid)
                if j < len(edges_list):
                    chain.append(f" —[{edges_list[j][1]}]-> ")
            st.code(" ".join(chain), language=None)
    with st.expander("All edges traversed"):
        for e in r.get("traversal", {}).get("edges_traversed", []):
            st.markdown(f"`{e[0]}` **→ [`{e[1]}`]** → `{e[2]}`")

with tab_cypher:
    st.caption("Cypher statements that reproduce this retrieval")
    st.code(r.get("cypher", ""), language="cypher")

with tab_evidence:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Kept context nodes**")
        st.write(prn.get("kept", []))
        st.markdown("**Pruned (low relevance)**")
        st.write(prn.get("dropped", []))
    with col_b:
        st.markdown("**Ranked context (top 10)**")
        st.dataframe(r.get("ranking", [])[:10], width="stretch", hide_index=True)
        st.markdown("**Stage timings**")
        st.dataframe(
            [{"stage": k.replace("_ms", ""), "ms": v} for k, v in tm.items()],
            width="stretch",
            hide_index=True,
        )

# ---- Exports ----
st.markdown("---")
st.markdown("#### Export Audit Report")
st.caption("Compliance-ready artifacts")
x1, x2, x3 = st.columns(3)
with x1:
    st.download_button("JSON", data=render_json(r),
                       file_name=f"{r['audit_id']}.json",
                       mime="application/json", use_container_width=True)
with x2:
    st.download_button("HTML Report", data=render_html(r),
                       file_name=f"{r['audit_id']}.html",
                       mime="text/html", use_container_width=True)
with x3:
    st.download_button("PDF Report", data=render_pdf(r),
                       file_name=f"{r['audit_id']}.pdf",
                       mime="application/pdf", use_container_width=True)

# ---- Footer ----
st.markdown(
    f"<div class='grag-footer'>"
    f"Audit ID: {r['audit_id']} · {r['timestamp']} · "
    f"GraphRAG Insurance Claims System — Lineage & Explainability"
    f"</div>",
    unsafe_allow_html=True,
)
