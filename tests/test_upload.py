"""Tests for the shared root-level upload.py pipeline."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from io import BytesIO

import pytest
from PyPDF2 import PdfWriter

from upload import (
    DuplicateUploadError,
    MalwareDetectedError,
    MalwareScanResult,
    UploadLimits,
    UploadTooLargeError,
    UploadValidationError,
    ensure_no_duplicate_uploads,
    read_upload_limited,
    record_upload_audit,
    rollback_upload_directory,
    save_upload_batch_atomic,
    validate_upload,
    validate_upload_batch,
    validate_upload_manifest,
    verify_recorded_checksums,
)


def _pdf(pages: int = 1, *, encrypted: bool = False) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    if encrypted:
        writer.encrypt("secret")
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


def _csv() -> bytes:
    return b"claim_id,amount,fraud\nCLM-1,100,0\nCLM-2,200,1\n"


def test_pdf_validation_returns_checksum_and_page_count():
    item = validate_upload("policy.pdf", _pdf(), allowed_kinds=("pdf",))
    assert item.kind == "pdf"
    assert item.details == {"page_count": 1}
    assert len(item.sha256) == 64
    assert item.metadata()["size_bytes"] == len(item.contents)


def test_rejects_unsafe_and_unsupported_filenames():
    for name in (
        "../policy.pdf",
        "..\\policy.pdf",
        ".hidden.pdf",
        "policy?.pdf",
        "CON.pdf",
    ):
        with pytest.raises(UploadValidationError, match="Filename"):
            validate_upload(name, _pdf())
    with pytest.raises(UploadValidationError, match="PDF and CSV"):
        validate_upload("claims.exe", b"data")
    assert validate_upload("दावा.csv", _csv()).filename == "दावा.csv"


def test_rejects_corrupt_encrypted_and_over_page_limit_pdfs():
    with pytest.raises(UploadValidationError, match="corrupt"):
        validate_upload("bad.pdf", b"%PDF-1.7\nnot a real PDF")
    with pytest.raises(UploadValidationError, match="encrypted"):
        validate_upload("locked.pdf", _pdf(encrypted=True))
    with pytest.raises(UploadValidationError, match="too many pages"):
        validate_upload(
            "long.pdf",
            _pdf(2),
            limits=replace(UploadLimits(), max_pdf_pages=1),
        )


def test_rejects_oversized_file_and_batch():
    with pytest.raises(UploadTooLargeError):
        validate_upload(
            "claims.csv",
            _csv(),
            limits=replace(UploadLimits(), max_csv_bytes=5),
        )
    with pytest.raises(UploadTooLargeError, match="batch"):
        validate_upload_batch(
            [("a.csv", _csv(), "text/csv")],
            limits=replace(UploadLimits(), max_batch_bytes=5),
        )


def test_csv_validation_and_duplicate_ids():
    item = validate_upload("claims.csv", _csv())
    assert item.details == {"row_count": 2, "column_count": 3}

    with pytest.raises(UploadValidationError, match="duplicate ID"):
        validate_upload("dup.csv", b"claim_id,value\nA,1\nA,2\n")
    with pytest.raises(UploadValidationError, match="duplicate column"):
        validate_upload("dup_headers.csv", b"id,ID\n1,2\n")
    with pytest.raises(UploadValidationError, match="whitespace"):
        validate_upload("spaced_header.csv", b"id, amount\n1,2\n")
    with pytest.raises(UploadValidationError, match="UTF-8"):
        validate_upload("latin.csv", b"name\n\xff\n")
    with pytest.raises(UploadValidationError, match="no data rows"):
        validate_upload("empty.csv", b"id,name\n")


def test_csv_resource_limits():
    with pytest.raises(UploadValidationError, match="too many rows"):
        validate_upload(
            "rows.csv",
            b"id\n1\n2\n",
            limits=replace(UploadLimits(), max_csv_rows=1),
        )
    with pytest.raises(UploadValidationError, match="too many columns"):
        validate_upload(
            "columns.csv",
            b"a,b\n1,2\n",
            limits=replace(UploadLimits(), max_csv_columns=1),
        )
    with pytest.raises(UploadValidationError, match="field longer"):
        validate_upload(
            "field.csv",
            b"id,value\n1,abcd\n",
            limits=replace(UploadLimits(), max_csv_field_chars=3),
        )


def test_manifest_preflights_declared_size_before_body_copy():
    with pytest.raises(UploadTooLargeError):
        validate_upload_manifest(
            [("large.pdf", 11)],
            limits=replace(UploadLimits(), max_pdf_bytes=10),
        )
    assert validate_upload_manifest([("claims.csv", len(_csv()))]) == [
        ("claims.csv", "csv")
    ]


def test_batch_rejects_mixed_and_duplicates_but_accepts_relational_csv_bundle():
    with pytest.raises(UploadValidationError, match="mix"):
        validate_upload_batch(
            [
                ("policy.pdf", _pdf(), "application/pdf"),
                ("claims.csv", _csv(), "text/csv"),
            ]
        )
    csv_bundle = validate_upload_batch(
        [
            ("one.csv", _csv(), "text/csv"),
            ("two.csv", b"id,value\n2,x\n", "text/csv"),
        ]
    )
    assert [item.filename for item in csv_bundle] == ["one.csv", "two.csv"]
    with pytest.raises(DuplicateUploadError, match="duplicates"):
        validate_upload_batch(
            [
                ("one.pdf", _pdf(), "application/pdf"),
                ("two.pdf", _pdf(), "application/pdf"),
            ]
        )


def test_malware_scanner_hook_and_metadata():
    def clean(name: str, contents: bytes) -> MalwareScanResult:
        return MalwareScanResult("clean", "test-scanner")

    item = validate_upload("claims.csv", _csv(), malware_scanner=clean)
    assert item.malware_scan == MalwareScanResult("clean", "test-scanner")

    def infected(name: str, contents: bytes) -> MalwareScanResult:
        raise MalwareDetectedError()

    with pytest.raises(MalwareDetectedError):
        validate_upload("claims.csv", _csv(), malware_scanner=infected)


def test_atomic_save_and_rollback(tmp_path):
    files = validate_upload_batch([("claims.csv", _csv(), "text/csv")])
    target = tmp_path / "session"
    saved = save_upload_batch_atomic(files, target)
    assert saved == [target / "claims.csv"]
    assert saved[0].read_bytes() == _csv()
    assert not list(tmp_path.glob(".session-upload-*"))

    with pytest.raises(UploadValidationError, match="already exists"):
        save_upload_batch_atomic(files, target)
    rollback_upload_directory(target)
    assert not target.exists()


def test_registry_deduplication_and_checksum_verification(tmp_path):
    upload = validate_upload("claims.csv", _csv())
    records = [{"name": "existing", "uploads": [upload.metadata()], "sources": []}]
    with pytest.raises(DuplicateUploadError, match="existing"):
        ensure_no_duplicate_uploads([upload], records, project_root=tmp_path)

    verify_recorded_checksums([upload], [upload.metadata()])
    bad_metadata = [{**upload.metadata(), "sha256": "0" * 64}]
    with pytest.raises(UploadValidationError, match="checksum"):
        verify_recorded_checksums([upload], bad_metadata)


def test_upload_audit_contains_metadata_but_not_contents(tmp_path):
    upload = validate_upload("claims.csv", _csv())
    path = tmp_path / "uploads.jsonl"
    event = record_upload_audit(
        audit_path=path,
        outcome="success",
        actor="analyst-1",
        tenant_id="tenant-a",
        entry_point="test",
        uploads=[upload],
        session_name="claims",
    )
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == event
    assert stored["files"][0]["sha256"] == upload.sha256
    assert "contents" not in path.read_text(encoding="utf-8")


def test_bounded_async_reader_stops_after_limit():
    class FakeUpload:
        def __init__(self, data: bytes):
            self.data = data
            self.offset = 0

        async def read(self, size: int) -> bytes:
            result = self.data[self.offset : self.offset + size]
            self.offset += len(result)
            return result

    assert asyncio.run(read_upload_limited(FakeUpload(b"1234"), 4)) == b"1234"
    oversized = FakeUpload(b"12345" + b"x" * 100)
    with pytest.raises(UploadTooLargeError):
        asyncio.run(read_upload_limited(oversized, 4, chunk_bytes=2))
    assert oversized.offset == 5
