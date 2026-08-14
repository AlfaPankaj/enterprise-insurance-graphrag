"""PDF text extraction (pdfplumber preferred, PyPDF2 fallback)."""

from __future__ import annotations

from io import BytesIO


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from an insurance PDF (bytes) as a single string.

    pdfplumber gives the most faithful layout for the generated reportlab
    documents; PyPDF2 is a dependency-light fallback.
    """
    if not pdf_bytes:
        return ""

    try:
        import pdfplumber

        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        if any(pages):
            return "\n\n".join(pages)
    except Exception:
        pass  # fall through to PyPDF2

    from PyPDF2 import PdfReader

    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)
