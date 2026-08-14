"""Input validation (Phase 5): upload constraints for the CDC endpoint.

PDF uploads are the only file type the pipeline ingests — anything else is
rejected before a byte is read. Size caps keep a misbehaving client from
feeding the extractor a multi-GB file.
"""

from __future__ import annotations

import re

# conservative caps: real insurance PDFs are a few hundred KB
MAX_PDF_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_PDF_PAGES = 400
_ALLOWED_NAME = re.compile(r"^[\w\-. ]+\.pdf$", re.IGNORECASE)


class UploadValidationError(ValueError):
    """Raised when an upload fails validation; carries a client-safe message."""


def validate_pdf_upload(filename: str | None, contents: bytes | None) -> None:
    """Raise ``UploadValidationError`` for any invalid PDF upload."""
    if not filename or not filename.lower().endswith(".pdf"):
        raise UploadValidationError("Only PDF files are allowed")
    if not _ALLOWED_NAME.match(filename):
        raise UploadValidationError("Filename contains invalid characters")
    if not contents:
        raise UploadValidationError("Empty file")
    if len(contents) > MAX_PDF_BYTES:
        raise UploadValidationError(
            f"File too large (max {MAX_PDF_BYTES // (1024 * 1024)} MB)"
        )
    if not contents.startswith(b"%PDF"):
        raise UploadValidationError("Not a valid PDF (missing %PDF header)")
