"""Tests for pdf_processor (requires the generated sample PDFs)."""

from pathlib import Path

import pytest

from graphrag.pdf_processor import extract_text_from_pdf

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "pdfs"

pytestmark = pytest.mark.skipif(
    not (PDF_DIR / "policy_POL-0001.pdf").exists(),
    reason="Sample PDFs not generated — run scripts/data_pipeline.py first",
)


def test_policy_pdf_contains_kv_text():
    text = extract_text_from_pdf((PDF_DIR / "policy_POL-0001.pdf").read_bytes())
    assert "Policy ID POL-0001" in text
    assert "Policy Number" in text
    assert "Annual Premium" in text


def test_claim_pdf_contains_kv_text():
    text = extract_text_from_pdf((PDF_DIR / "claim_CLM-0001.pdf").read_bytes())
    assert "Claim ID CLM-0001" in text
    assert "Policy ID" in text


def test_empty_bytes_returns_empty():
    assert extract_text_from_pdf(b"") == ""
