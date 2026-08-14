"""Interactive lineage visualization (Phase 4, Shot 3).

``render_lineage_html(record)`` renders the traversal of one audit record as an
interactive force-directed graph (pyvis/vis.js): seed nodes get a bold border,
kept context nodes are highlighted, pruned nodes are faded, and every edge
carries its relationship type as a label.

pyvis is optional — if it is not installed (or fails to import), a readable
HTML edge-list is produced instead, so the audit UI never breaks.
"""

from __future__ import annotations

import html

# label -> (fill color, border color)
_LABEL_COLORS = {
    "Policyholder": ("#f68b08", "#0969da"),
    "Policy": ("#dafbdf", "#1a7f37"),
    "Claim": ("#ffe9e9", "#bf3989"),
    "Coverage": ("#ffc5c7", "#9a6700"),
    "Endorsement": ("#ffefef", "#8250df"),
    "FraudFlag": ("#ffe9ec", "#cf222e"),
    "Investigator": ("#faf6f6", "#57606a"),
    "DocSnapshot": ("#faf6f6", "#6e7781"),
}
_DEFAULT_COLOR = ("#f6f8fa", "#57606a")


def _record_nodes(record: dict) -> dict[str, dict]:
    """Merge kept/pruned ids into ``{id: {"label": str, "kept": bool}}``."""
    pruned = record.get("pruned", {})
    labels: dict[str, str] = {}
    for row in record.get("ranking", []):
        labels[row["id"]] = row.get("label", "Node")
    nodes: dict[str, dict] = {
        nid: {"label": labels.get(nid, "Node"), "kept": True} for nid in pruned.get("kept", [])
    }
    for nid in pruned.get("dropped", []):
        nodes.setdefault(nid, {"label": labels.get(nid, "Node"), "kept": False})["kept"] = False
    # any visited node missing from kept/dropped is still "traversed"
    for nid in record.get("traversal", {}).get("nodes_visited", []):
        nodes.setdefault(nid, {"label": labels.get(nid, "Node"), "kept": True})
    return nodes


def _tooltip(node_id: str, label: str) -> str:
    return html.escape(f"{label} {node_id}")


def render_lineage_html(record: dict, height: int = 520) -> str:
    """Self-contained HTML embedding the traversal graph (interactive)."""
    try:
        return _render_pyvis(record, height)
    except Exception:
        return _render_fallback(record, height)


def _render_pyvis(record: dict, height: int = 520) -> str:
    from pyvis.network import Network

    nodes = _record_nodes(record)
    net = Network(
        height=f"{height}px", width="200%", directed=True, notebook=False,
        bgcolor="#0f1117", font_color="#e6edf3",
        cdn_resources="in_line",
    )
    net.set_options(
        '{"physics": {"barnesHut": {"gravitationalConstant": -3000,'
        ' "springLength": 500, "springConstant": 0.01,'
        ' "damping": 0.04, "centralGravity": 0.005},'
        ' "minVelocity": 0.1, "stabilization": {"iterations": 300}},'
        ' "interaction": {"hover": true, "zoomView": true, "dragView": true,'
        ' "navigationButtons": false},'
        ' "edges": {"smooth": {"type": "continuous"}}}'
    )
    seeds = set(record.get("retrieval", {}).get("seeds", []))
    for nid, meta in nodes.items():
        fill, border = _LABEL_COLORS.get(meta["label"], _DEFAULT_COLOR)
        label_text = html.escape(f"{meta['label']} {nid}")
        net.add_node(
            nid, label=label_text, title=_tooltip(nid, meta["label"]),
            color={"background": fill, "border": border},
            borderWidth=4 if nid in seeds else (2 if meta["kept"] else 1),
            shape="box" if nid in seeds else "dot",
            opacity=1.0 if meta["kept"] else 0.45,
            size=22 if meta["kept"] else 14,
            font={"color": "#e6edf3", "size": 13},
        )
    for e in record.get("traversal", {}).get("edges_traversed", []):
        src, rel, dst = e
        if src not in nodes or dst not in nodes:
            continue
        net.add_edge(src, dst, label=html.escape(rel),
                     font={"size": 11, "color": "#8b949e"})
    out = net.generate_html()

    # Force html/body/.card/#mynetwork to fill the Streamlit iframe.
    # Pyvis wraps #mynetwork in a .card div with no height — must fix both.
    inject = (
        "<style>"
        "html,body{height:100%!important;margin:0!important;padding:0!important;"
        "overflow:hidden!important;background:#0f1117!important;}"
        ".card,.card-body,#mynetwork{height:100%!important;width:100%!important;"
        "margin:0!important;padding:0!important;}"
        "#header,.vis-navigation{display:none!important;}"
        "</style>"
    )
    if "</head>" in out:
        return out.replace("</head>", inject + "</head>", 1)
    return inject + out


def _render_fallback(record: dict, height: int = 520) -> str:
    """Plain (still pretty) HTML edge list when pyvis is unavailable."""
    nodes = _record_nodes(record)
    seeds = set(record.get("retrieval", {}).get("seeds", []))
    node_rows = "".join(
        f"<li><b>{html.escape(nid)}</b> "
        f"<span style='color:#57606a'>({html.escape(m['label'])}"
        f"{', seed' if nid in seeds else ''}"
        f"{'' if m['kept'] else ', pruned'})</span></li>"
        for nid, m in sorted(nodes.items())
    )
    edge_rows = "".join(
        f"<li><code>{html.escape(e[0])}</code> "
        f"<span style='color:#1f6feb'>─[{html.escape(e[1])}]→</span> "
        f"<code>{html.escape(e[2])}</code></li>"
        for e in record.get("traversal", {}).get("edges_traversed", [])
    )
    return f"""
    <div style="font-family:-apple-system,'Segoe UI',sans-serif;height:{height}px;
                overflow:auto;border:1px solid #d0d7de;border-radius:8px;padding:12px;">
      <h4 style="margin:4px 0 8px">Traversal graph (interactive graph unavailable —
      showing edge list)</h4>
      <h5 style="margin:8px 0 4px">Nodes ({len(nodes)})</h5>
      <ul style="columns:2;font-size:0.85rem">{node_rows}</ul>
      <h5 style="margin:8px 0 4px">Edges ({len(record.get('traversal', {}).get('edges_traversed', []))})</h5>
      <ul style="font-size:0.85rem">{edge_rows}</ul>
    </div>
    """
