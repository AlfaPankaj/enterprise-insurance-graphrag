"""Compatibility wrappers for the shared root-level upload pipeline.

New code should import from ``upload.py`` directly.  This module keeps the
original PDF-validation API stable for existing integrations and tests.
"""

from __future__ import annotations

from upload import (
    DEFAULT_MAX_PDF_BYTES as MAX_PDF_BYTES,
    DEFAULT_MAX_PDF_PAGES as MAX_PDF_PAGES,
    UploadLimits,
    UploadValidationError,
    validate_upload,
)


def validate_pdf_upload(filename: str | None, contents: bytes | None) -> None:
    """Validate a PDF using filename, size, parseability, and page-count checks."""
    validate_upload(
        filename,
        contents,
        limits=UploadLimits(
            max_pdf_bytes=MAX_PDF_BYTES,
            max_pdf_pages=MAX_PDF_PAGES,
        ),
        allowed_kinds=("pdf",),
    )
