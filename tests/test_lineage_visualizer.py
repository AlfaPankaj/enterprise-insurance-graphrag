"""Unit tests for the lineage visualizer (Phase 4)."""

from graphrag.lineage_visualizer import _render_fallback, render_lineage_html

RECORD = {
    "retrieval": {"seeds": ["CLM-0003"]},
    "ranking": [{"id": "CLM-0003", "label": "Claim", "score": 5.0},
                {"id": "FRD-CLM-0003", "label": "FraudFlag", "score": 1.0}],
    "pruned": {"kept": ["CLM-0003", "FRD-CLM-0003"], "dropped": ["POL-0005"]},
    "traversal": {
        "nodes_visited": ["CLM-0003", "FRD-CLM-0003", "POL-0005"],
        "edges_traversed": [["CLM-0003", "FRAUD_DETECTED", "FRD-CLM-0003"],
                            ["POL-0005", "HAS_CLAIM", "CLM-0003"]],
        "paths": [],
    },
}


def test_render_lineage_html_embeds_nodes_and_edges():
    html = render_lineage_html(RECORD)
    assert "CLM-0003" in html
    assert "FRD-CLM-0003" in html
    assert "FRAUD_DETECTED" in html


def test_render_lineage_html_pyvis_path():
    # with pyvis installed the interactive (vis.js) renderer is used
    html = render_lineage_html(RECORD)
    assert "vis" in html.lower() or "network" in html.lower()
    assert "FRAUD_DETECTED" in html


def test_fallback_marks_pruned_and_seed():
    html = _render_fallback(RECORD, height=400)
    assert "CLM-0003" in html
    assert "pruned" in html        # POL-0005 faded in the list
    assert "seed" in html          # CLM-0003 flagged as seed
    assert "FRAUD_DETECTED" in html
