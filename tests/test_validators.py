"""Tests for Phase 5 upload validation (src/graphrag/validators.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from graphrag.validators import (
    MAX_PDF_BYTES,
    UploadValidationError,
    validate_pdf_upload,
)

_PDF_HEADER = b"%PDF-1.7\n..."


def test_rejects_non_pdf():
    with pytest.raises(UploadValidationError, match="PDF"):
        validate_pdf_upload("policy.txt", _PDF_HEADER)
    with pytest.raises(UploadValidationError, match="PDF"):
        validate_pdf_upload("policy.pdf.exe", _PDF_HEADER)


def test_rejects_empty_and_missing_name():
    with pytest.raises(UploadValidationError, match="Empty"):
        validate_pdf_upload("policy.pdf", b"")
    with pytest.raises(UploadValidationError, match="PDF"):
        validate_pdf_upload(None, _PDF_HEADER)


def test_rejects_oversized():
    big = _PDF_HEADER + b"x" * (MAX_PDF_BYTES + 1)
    with pytest.raises(UploadValidationError, match="too large"):
        validate_pdf_upload("policy.pdf", big)


def test_rejects_non_pdf_magic_bytes():
    with pytest.raises(UploadValidationError, match="header"):
        validate_pdf_upload("policy.pdf", b"PK\x03\x04 not a pdf")


def test_accepts_valid():
    validate_pdf_upload("endorsement_001.pdf", _PDF_HEADER)  # no raise
