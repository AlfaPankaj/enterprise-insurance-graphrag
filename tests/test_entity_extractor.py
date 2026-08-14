"""Extraction accuracy: the heuristic parser must recover the exact entity
ids of every generated PDF (the "CDC detects 100% of entity changes" proof)."""

import json
from pathlib import Path

import pytest

from graphrag.entity_extractor import extract_entities_heuristic
from graphrag.pdf_processor import extract_text_from_pdf

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "pdfs"
GT_FILE = ROOT / "data" / "samples" / "ground_truth.json"

pytestmark = pytest.mark.skipif(
    not (PDF_DIR / "policy_POL-0001.pdf").exists(),
    reason="Sample PDFs not generated — run scripts/data_pipeline.py first",
)


@pytest.mark.parametrize("pdf_name", sorted(p.name for p in PDF_DIR.glob("*.pdf")))
def test_extracted_entity_ids_match_ground_truth(pdf_name):
    ground_truth = json.loads(GT_FILE.read_text(encoding="utf-8"))
    truth_entities = {(e["label"], e["id"]) for e in ground_truth[pdf_name]["entities"]}

    text = extract_text_from_pdf((PDF_DIR / pdf_name).read_bytes())
    result = extract_entities_heuristic(text, doc_id_hint=pdf_name)
    got = {(label, eid) for label, ents in result["entities"].items() for eid in ents}

    assert result["entities"], f"extracted nothing from {pdf_name}"
    assert got == truth_entities, (
        f"{pdf_name}: missing={sorted(truth_entities - got)} extra={sorted(got - truth_entities)}"
    )


def test_doc_id_is_stable_entity_id():
    text = extract_text_from_pdf((PDF_DIR / "policy_POL-0001.pdf").read_bytes())
    result = extract_entities_heuristic(text)
    assert result["doc_id"] == "POL-0001"
    assert "Policyholder" in result["entities"] and "Coverage" in result["entities"]
