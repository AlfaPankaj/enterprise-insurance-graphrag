"""Audit report exports — JSON / HTML / PDF (Phase 4, Shot 3).

Compliance-grade artifacts for a single audit record:

  * ``export_json`` — the raw record (lossless, machine-readable).
  * ``export_html`` — self-contained printable report from a template.
  * ``export_pdf``  — single-page PDF via reportlab (installed).

Each returns the rendered payload; ``save_exports`` writes all three to disk
under ``settings.AUDIT_DIR/exports/<audit_id>.*`` for the demo/repo proof.
"""

from __future__ import annotations

import html as _html
import io
import json
from pathlib import Path
from string import Template

from graphrag.config import settings
from graphrag.traversal_logger import ROOT

_TEMPLATE = Template(
    (ROOT / "src" / "graphrag" / "templates" / "audit_report_template.html")
    .read_text(encoding="utf-8")
)


# ---------------------------------------------------------------------------
# renderers
# ---------------------------------------------------------------------------

def _path_block(record: dict) -> str:
    """Monospace tree of every extracted chain (nodes + edge types)."""
    paths = record.get("traversal", {}).get("paths", [])
    if not paths:
        return "<i>No traversal path extracted (no seed matched).</i>"
    blocks = []
    for p in paths:
        nodes = p["nodes"]
        edges = p["edges"]
        chain = []
        for i, nid in enumerate(nodes):
            chain.append(_html.escape(str(nid)))
            if i < len(edges):
                chain.append(f"─[{_html.escape(edges[i][1])}]→")
        blocks.append("&nbsp;&nbsp;".join(chain))
    return "<br>".join(blocks)


def _ranking_rows(record: dict) -> str:
    rows = record.get("ranking", [])[:10]
    return "".join(
        f"<tr><td>{i + 1}</td><td>{_html.escape(r['id'])}</td>"
        f"<td>{_html.escape(r['label'])}</td><td>{r['score']}</td></tr>"
        for i, r in enumerate(rows)
    )


def _esc(value: str) -> str:
    """HTML-escape dynamic text (queries/answers/cypher can hold &, <, >)."""
    return _html.escape(str(value))


def render_html(record: dict) -> str:
    """Self-contained HTML audit report (all dynamic text HTML-escaped)."""
    pruned = record.get("pruned", {})
    tokens = record.get("tokens", {})
    timing = record.get("timings_ms", {})
    return _TEMPLATE.substitute(
        audit_id=_esc(record.get("audit_id", "?")),
        timestamp=_esc(record.get("timestamp", "?")),
        reranker=_esc(record.get("reranker", "?")),
        reranker_mode_note="neural" if record.get("reranker") == "cross-encoder" else "BM25 lexical",
        max_hops=record.get("max_hops", settings.MAX_HOPS),
        query=_esc(record.get("query", "")),
        answer=_esc(record.get("answer", "")),
        answer_mode=_esc(record.get("answer_mode", "extractive")),
        path_block=_path_block(record),
        cypher=_esc(record.get("cypher", "")),
        seeds=", ".join(_esc(s) for s in record.get("retrieval", {}).get("seeds", []))
               or "keyword match",
        node_count=record.get("retrieval", {}).get("node_count", 0),
        edge_count=record.get("retrieval", {}).get("edge_count", 0),
        kept_count=pruned.get("kept_count", 0),
        dropped_count=pruned.get("dropped_count", 0),
        budget=pruned.get("budget", "—"),
        tokens_before=tokens.get("before", 0),
        tokens_after=tokens.get("after", 0),
        savings=tokens.get("savings_percent", 0),
        retrieval_ms=timing.get("retrieval_ms", 0),
        rerank_ms=timing.get("rerank_ms", 0),
        prune_ms=timing.get("prune_ms", 0),
        total_ms=timing.get("total_ms", 0),
        ranking_rows=_ranking_rows(record),
    )


def render_json(record: dict) -> str:
    return json.dumps(record, indent=2, default=str)


def render_pdf(record: dict) -> bytes:
    """Single-page PDF via reportlab platypus (wraps + wraps text cleanly)."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Title"], fontSize=16, spaceAfter=4)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11,
                        textColor=colors.HexColor("#1f6feb"), spaceBefore=10)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.5,
                          leading=13, wordWrap="CJK")
    code = ParagraphStyle("Code", parent=styles["Code"], fontSize=8, leading=10.5,
                          wordWrap="CJK", backColor=colors.HexColor("#f6f8fa"),
                          borderColor=colors.HexColor("#d0d7de"), borderWidth=0.5,
                          borderPadding=6)
    muted = ParagraphStyle("Muted", parent=body, fontSize=8, textColor=colors.HexColor("#57606a"))

    pruned = record.get("pruned", {})
    tokens = record.get("tokens", {})
    timing = record.get("timings_ms", {})

    # reportlab Paragraphs parse a mini-XML — escape text so & < > in
    # queries/answers/ids cannot crash or garble the report
    e = _html.escape
    story = [
        Paragraph("GraphRAG Audit Trail Report", h1),
        Paragraph(
            f"Audit ID: {e(record.get('audit_id', ''))} · "
            f"{e(record.get('timestamp', ''))} · "
            f"Reranker: {e(record.get('reranker', ''))} · "
            f"Max hops: {record.get('max_hops')}",
            muted,
        ),
        Spacer(1, 8),
        Paragraph("Question", h2),
        Paragraph(e(record.get("query", "")), body),
        Paragraph("Answer", h2),
        Paragraph(e(record.get("answer", "")), body),
        Paragraph("Traversal path (nodes &amp; edges visited)", h2),
    ]

    paths = record.get("traversal", {}).get("paths", [])
    if paths:
        for p in paths[:10]:
            chain = []
            nodes, edges = p["nodes"], p["edges"]
            for i, nid in enumerate(nodes):
                chain.append(e(nid))
                if i < len(edges):
                    chain.append(f"-[{e(edges[i][1])}]->")
            story.append(Paragraph("&nbsp;&nbsp;".join(chain), code))
    else:
        story.append(Paragraph("No traversal path extracted (no seed matched).", body))

    story += [
        Paragraph("Cypher used", h2),
        Paragraph(e(record.get("cypher", "")).replace("\n", "<br/>"), code),
        Paragraph("Retrieval, pruning &amp; performance", h2),
        Table(
            [
                ["Seed nodes", ", ".join(e(s) for s in record.get("retrieval", {}).get("seeds", []))
                              or "keyword match"],
                ["Retrieved", f"{record.get('retrieval', {}).get('node_count', 0)} nodes / "
                              f"{record.get('retrieval', {}).get('edge_count', 0)} edges"],
                ["After pruning", f"{pruned.get('kept_count', 0)} kept / "
                                  f"{pruned.get('dropped_count', 0)} dropped (budget {pruned.get('budget')})"],
                ["Tokens", f"{tokens.get('before', 0)} -> {tokens.get('after', 0)} "
                           f"({tokens.get('savings_percent', 0)}% saved)"],
                ["Timings", f"retrieval {timing.get('retrieval_ms', 0)}ms · "
                            f"rerank {timing.get('rerank_ms', 0)}ms · "
                            f"prune {timing.get('prune_ms', 0)}ms · "
                            f"total {timing.get('total_ms', 0)}ms"],
            ],
            colWidths=[1.4 * inch, 5.2 * inch],
        ),
        Paragraph("Ranked context (top 10)", h2),
        Table(
            [["Rank", "Node", "Label", "Score"]] + [
                [i + 1, e(r["id"]), e(r["label"]), r["score"]]
                for i, r in enumerate(record.get("ranking", [])[:10])
            ],
            colWidths=[0.6 * inch, 2.4 * inch, 2.4 * inch, 1.2 * inch],
        ),
    ]
    story[-1].setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f6f8fa")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d0d7de")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                      topMargin=0.6 * inch, bottomMargin=0.6 * inch).build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# disk exports
# ---------------------------------------------------------------------------

def _export_dir() -> Path:
    path = ROOT / settings.AUDIT_DIR / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_exports(record: dict) -> dict[str, str]:
    """Write JSON + HTML + PDF to disk; returns {kind: filename}."""
    audit_id = record.get("audit_id", "unknown")
    out: dict[str, str] = {}
    for kind, payload, ext in (
        ("json", render_json(record), "json"),
        ("html", render_html(record), "html"),
        ("pdf", render_pdf(record), "pdf"),
    ):
        target = _export_dir() / f"{audit_id}.{ext}"
        data = payload.encode("utf-8") if isinstance(payload, str) else payload
        target.write_bytes(data)
        out[kind] = str(target.relative_to(ROOT))
    return out
