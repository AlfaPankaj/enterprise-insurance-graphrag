"""Secure, shared upload pipeline for the GraphRAG application.

This module intentionally lives beside :mod:`app` so the Streamlit UI,
FastAPI endpoint, and custom-session ingestion script all use one upload
contract.  It provides:

* bounded reads and configurable file/batch limits;
* safe filename, PDF, and CSV validation;
* homogeneous-batch enforcement (PDF bundles or relational CSV bundles);
* SHA-256 checksums and custom-session duplicate detection;
* an optional command-based malware scanner hook;
* atomic persistence with rollback helpers; and
* append-only JSONL upload audit metadata (never file contents).

The module has no dependency on Streamlit or FastAPI.  Framework-specific
callers pass uploaded bytes in and receive immutable ``ValidatedUpload``
objects back.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import unicodedata
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

logger = logging.getLogger("graphrag.upload")

MIB = 1024 * 1024
DEFAULT_MAX_PDF_BYTES = 25 * MIB
DEFAULT_MAX_CSV_BYTES = 50 * MIB
DEFAULT_MAX_BATCH_BYTES = 100 * MIB
DEFAULT_MAX_FILES = 20
DEFAULT_MAX_PDF_PAGES = 400
DEFAULT_MAX_CSV_ROWS = 100_000
DEFAULT_MAX_CSV_COLUMNS = 256
DEFAULT_MAX_CSV_FIELD_CHARS = 100_000
DEFAULT_MAX_FILENAME_CHARS = 128
READ_CHUNK_BYTES = MIB

_ALLOWED_EXTENSIONS = frozenset({".pdf", ".csv"})
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)
_ID_COLUMN_RE = re.compile(r"(^|_)(id|number|no)$|_id$|^id$", re.IGNORECASE)
_AUDIT_LOCK = threading.Lock()


class UploadValidationError(ValueError):
    """Client-safe upload error with a stable code and HTTP status hint."""

    def __init__(
        self, message: str, *, code: str = "invalid_upload", status_code: int = 400
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class UploadTooLargeError(UploadValidationError):
    """An individual upload or complete batch exceeds its configured limit."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="upload_too_large", status_code=413)


class DuplicateUploadError(UploadValidationError):
    """The same content was already uploaded or repeated in a batch."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="duplicate_upload", status_code=409)


class MalwareDetectedError(UploadValidationError):
    """A configured malware scanner rejected the content."""

    def __init__(self, message: str = "Upload rejected by malware scanner") -> None:
        super().__init__(message, code="malware_detected", status_code=422)


@dataclass(frozen=True)
class UploadLimits:
    """Resource limits shared by every upload entry point."""

    max_pdf_bytes: int = DEFAULT_MAX_PDF_BYTES
    max_csv_bytes: int = DEFAULT_MAX_CSV_BYTES
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES
    max_files: int = DEFAULT_MAX_FILES
    max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES
    max_csv_rows: int = DEFAULT_MAX_CSV_ROWS
    max_csv_columns: int = DEFAULT_MAX_CSV_COLUMNS
    max_csv_field_chars: int = DEFAULT_MAX_CSV_FIELD_CHARS
    max_filename_chars: int = DEFAULT_MAX_FILENAME_CHARS

    @classmethod
    def from_settings(cls, settings: object) -> UploadLimits:
        """Build limits from the app settings without importing the app package."""
        return cls(
            max_pdf_bytes=int(
                getattr(settings, "UPLOAD_MAX_PDF_BYTES", DEFAULT_MAX_PDF_BYTES)
            ),
            max_csv_bytes=int(
                getattr(settings, "UPLOAD_MAX_CSV_BYTES", DEFAULT_MAX_CSV_BYTES)
            ),
            max_batch_bytes=int(
                getattr(settings, "UPLOAD_MAX_BATCH_BYTES", DEFAULT_MAX_BATCH_BYTES)
            ),
            max_files=int(getattr(settings, "UPLOAD_MAX_FILES", DEFAULT_MAX_FILES)),
            max_pdf_pages=int(
                getattr(settings, "UPLOAD_MAX_PDF_PAGES", DEFAULT_MAX_PDF_PAGES)
            ),
            max_csv_rows=int(
                getattr(settings, "UPLOAD_MAX_CSV_ROWS", DEFAULT_MAX_CSV_ROWS)
            ),
            max_csv_columns=int(
                getattr(settings, "UPLOAD_MAX_CSV_COLUMNS", DEFAULT_MAX_CSV_COLUMNS)
            ),
            max_csv_field_chars=int(
                getattr(
                    settings, "UPLOAD_MAX_CSV_FIELD_CHARS", DEFAULT_MAX_CSV_FIELD_CHARS
                )
            ),
            max_filename_chars=int(
                getattr(
                    settings, "UPLOAD_MAX_FILENAME_CHARS", DEFAULT_MAX_FILENAME_CHARS
                )
            ),
        )

    def max_bytes_for(self, kind: str) -> int:
        if kind == "pdf":
            return self.max_pdf_bytes
        if kind == "csv":
            return self.max_csv_bytes
        raise UploadValidationError("Only PDF and CSV files are allowed")


@dataclass(frozen=True)
class MalwareScanResult:
    """Non-sensitive result persisted with upload metadata."""

    status: str
    scanner: str = "none"


MalwareScanner = Callable[[str, bytes], MalwareScanResult]


@dataclass(frozen=True)
class ValidatedUpload:
    """A structurally validated file ready for persistence or ingestion."""

    filename: str
    kind: str
    contents: bytes = field(repr=False)
    content_type: str | None = None
    sha256: str = ""
    size_bytes: int = 0
    details: Mapping[str, int] = field(default_factory=dict)
    malware_scan: MalwareScanResult = field(
        default_factory=lambda: MalwareScanResult("disabled")
    )

    def metadata(self) -> dict[str, Any]:
        """Serializable audit/registry metadata; deliberately excludes bytes."""
        return {
            "filename": self.filename,
            "kind": self.kind,
            "content_type": self.content_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            **dict(self.details),
            "malware_scan": asdict(self.malware_scan),
        }


def _positive_limit(value: int, name: str) -> int:
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


def validate_filename(
    filename: str | None,
    *,
    limits: UploadLimits,
    allowed_kinds: Iterable[str] = ("pdf", "csv"),
) -> tuple[str, str]:
    """Return ``(safe_name, kind)`` or reject unsafe/unsupported names."""
    if not filename or not isinstance(filename, str):
        raise UploadValidationError("A filename is required", code="missing_filename")

    normalized = unicodedata.normalize("NFKC", filename)
    if normalized != filename:
        raise UploadValidationError(
            "Filename must use normalized characters", code="unsafe_filename"
        )
    if len(filename) > _positive_limit(
        limits.max_filename_chars, "UPLOAD_MAX_FILENAME_CHARS"
    ):
        raise UploadValidationError(
            f"Filename is too long (max {limits.max_filename_chars} characters)",
            code="unsafe_filename",
        )
    if filename in {".", ".."} or filename != Path(filename).name:
        raise UploadValidationError(
            "Filename must not contain a path", code="unsafe_filename"
        )
    if "/" in filename or "\\" in filename:
        raise UploadValidationError(
            "Filename must not contain path separators", code="unsafe_filename"
        )
    if any(unicodedata.category(ch).startswith("C") for ch in filename):
        raise UploadValidationError(
            "Filename contains control characters", code="unsafe_filename"
        )
    allowed_punctuation = {".", "_", "-", " "}
    if (
        filename.endswith((" ", "."))
        or not filename[0].isalnum()
        or any(
            not (
                ch.isalnum()
                or unicodedata.category(ch).startswith("M")
                or ch in allowed_punctuation
            )
            for ch in filename
        )
    ):
        raise UploadValidationError(
            "Filename may contain only letters, digits, spaces, '.', '_' and '-'",
            code="unsafe_filename",
        )
    if filename.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise UploadValidationError(
            "Filename uses a reserved system name", code="unsafe_filename"
        )

    extension = Path(filename).suffix.lower()
    if extension not in _ALLOWED_EXTENSIONS:
        raise UploadValidationError(
            "Only PDF and CSV files are allowed", code="unsupported_file_type"
        )
    kind = extension[1:]
    allowed = {str(item).lower().lstrip(".") for item in allowed_kinds}
    if kind not in allowed:
        label = " or ".join(sorted(item.upper() for item in allowed))
        raise UploadValidationError(
            f"Only {label} files are allowed", code="unsupported_file_type"
        )
    return filename, kind


def _validate_size(
    contents: bytes | None, filename: str, kind: str, limits: UploadLimits
) -> bytes:
    if not contents:
        raise UploadValidationError(f"Empty file: '{filename}'", code="empty_file")
    if not isinstance(contents, bytes):
        contents = bytes(contents)
    max_bytes = _positive_limit(limits.max_bytes_for(kind), f"max_{kind}_bytes")
    if len(contents) > max_bytes:
        raise UploadTooLargeError(
            f"'{filename}' is too large (max {max_bytes // MIB} MB)"
        )
    return contents


def _validate_pdf(
    contents: bytes, filename: str, limits: UploadLimits
) -> dict[str, int]:
    if not contents.startswith(b"%PDF-"):
        raise UploadValidationError(
            f"'{filename}' is not a valid PDF (missing PDF header)",
            code="invalid_pdf",
        )
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(BytesIO(contents), strict=True)
        if reader.is_encrypted:
            raise UploadValidationError(
                f"'{filename}' is encrypted; upload an unlocked PDF",
                code="encrypted_pdf",
            )
        page_count = len(reader.pages)
    except UploadValidationError:
        raise
    except Exception as exc:
        raise UploadValidationError(
            f"'{filename}' is corrupt or not a supported PDF",
            code="invalid_pdf",
        ) from exc

    if page_count < 1:
        raise UploadValidationError(f"'{filename}' has no pages", code="invalid_pdf")
    if page_count > _positive_limit(limits.max_pdf_pages, "UPLOAD_MAX_PDF_PAGES"):
        raise UploadValidationError(
            f"'{filename}' has too many pages (max {limits.max_pdf_pages})",
            code="too_many_pdf_pages",
        )
    return {"page_count": page_count}


def _validate_csv(
    contents: bytes, filename: str, limits: UploadLimits
) -> dict[str, int]:
    if b"\x00" in contents:
        raise UploadValidationError(
            f"'{filename}' contains NUL bytes and is not a valid CSV",
            code="invalid_csv",
        )
    try:
        text = contents.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise UploadValidationError(
            f"'{filename}' must be UTF-8 encoded", code="invalid_csv_encoding"
        ) from exc

    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        raw_headers = next(reader, None)
    except csv.Error as exc:
        raise UploadValidationError(
            f"'{filename}' is malformed CSV", code="invalid_csv"
        ) from exc
    if raw_headers is None:
        raise UploadValidationError(
            f"'{filename}' has no header row", code="invalid_csv"
        )

    headers = [header.strip() for header in raw_headers]
    if any(
        header != stripped
        for header, stripped in zip(raw_headers, headers, strict=True)
    ):
        raise UploadValidationError(
            f"'{filename}' has whitespace around a column name",
            code="invalid_csv_header",
        )
    if not headers or any(not header for header in headers):
        raise UploadValidationError(
            f"'{filename}' has an empty column name", code="invalid_csv_header"
        )
    if len(headers) > _positive_limit(limits.max_csv_columns, "UPLOAD_MAX_CSV_COLUMNS"):
        raise UploadValidationError(
            f"'{filename}' has too many columns (max {limits.max_csv_columns})",
            code="too_many_csv_columns",
        )
    folded = [header.casefold() for header in headers]
    if len(folded) != len(set(folded)):
        raise UploadValidationError(
            f"'{filename}' has duplicate column names", code="duplicate_csv_header"
        )
    max_field = _positive_limit(
        limits.max_csv_field_chars, "UPLOAD_MAX_CSV_FIELD_CHARS"
    )
    if any(len(header) > max_field for header in headers):
        raise UploadValidationError(
            f"'{filename}' contains a field longer than {max_field} characters",
            code="csv_field_too_large",
        )

    id_index = next(
        (i for i, header in enumerate(headers) if _ID_COLUMN_RE.search(header)), None
    )
    seen_ids: set[str] = set()
    row_count = 0
    try:
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            row_count += 1
            if row_count > _positive_limit(limits.max_csv_rows, "UPLOAD_MAX_CSV_ROWS"):
                raise UploadValidationError(
                    f"'{filename}' has too many rows (max {limits.max_csv_rows})",
                    code="too_many_csv_rows",
                )
            if len(row) != len(headers):
                raise UploadValidationError(
                    f"'{filename}' row {row_count + 1} has {len(row)} values; "
                    f"expected {len(headers)}",
                    code="invalid_csv_row",
                )
            if any(len(cell) > max_field for cell in row):
                raise UploadValidationError(
                    f"'{filename}' row {row_count + 1} contains a field longer "
                    f"than {max_field} characters",
                    code="csv_field_too_large",
                )
            if id_index is not None:
                entity_id = row[id_index].strip()
                if entity_id:
                    if entity_id in seen_ids:
                        raise UploadValidationError(
                            f"'{filename}' contains duplicate ID '{entity_id}'",
                            code="duplicate_csv_id",
                        )
                    seen_ids.add(entity_id)
    except csv.Error as exc:
        raise UploadValidationError(
            f"'{filename}' is malformed CSV", code="invalid_csv"
        ) from exc

    if row_count == 0:
        raise UploadValidationError(f"'{filename}' has no data rows", code="empty_csv")
    return {"row_count": row_count, "column_count": len(headers)}


def make_command_malware_scanner(
    command: str,
    *,
    timeout_s: float = 30.0,
    fail_closed: bool = True,
) -> MalwareScanner | None:
    """Create an optional scanner hook from a command such as ``clamscan``.

    The command is parsed with ``shlex`` and is never executed through a shell.
    ``{path}`` in an argument is replaced with a temporary file path; when no
    placeholder is present, the path is appended. Exit 0 means clean, exit 1
    means malware detected (ClamAV convention), and other exits are scanner
    failures. Scanner failures reject uploads when ``fail_closed`` is true.
    """
    if not command.strip():
        return None
    argv_template = shlex.split(command)
    if not argv_template:
        return None
    scanner_name = Path(argv_template[0]).name
    timeout_s = max(float(timeout_s), 0.1)

    def scan(filename: str, contents: bytes) -> MalwareScanResult:
        suffix = Path(filename).suffix
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="graphrag-upload-", suffix=suffix, delete=False
            ) as tmp:
                tmp.write(contents)
                temp_path = tmp.name
            has_placeholder = any("{path}" in arg for arg in argv_template)
            argv = [arg.replace("{path}", temp_path) for arg in argv_template]
            if not has_placeholder:
                argv.append(temp_path)
            try:
                completed = subprocess.run(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                    check=False,
                    text=True,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                if fail_closed:
                    raise UploadValidationError(
                        "Malware scanner is unavailable; upload rejected",
                        code="malware_scanner_unavailable",
                        status_code=503,
                    ) from exc
                logger.warning("malware scanner unavailable; allowing upload: %s", exc)
                return MalwareScanResult("error_allowed", scanner_name)
            if completed.returncode == 0:
                return MalwareScanResult("clean", scanner_name)
            if completed.returncode == 1:
                raise MalwareDetectedError()
            if fail_closed:
                raise UploadValidationError(
                    "Malware scanner failed; upload rejected",
                    code="malware_scanner_failed",
                    status_code=503,
                )
            logger.warning(
                "malware scanner exited %d; allowing upload", completed.returncode
            )
            return MalwareScanResult("error_allowed", scanner_name)
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    return scan


def scanner_from_settings(settings: object) -> MalwareScanner | None:
    """Build the configured scanner hook, or ``None`` when scanning is disabled."""
    return make_command_malware_scanner(
        str(getattr(settings, "UPLOAD_MALWARE_SCAN_COMMAND", "")),
        timeout_s=float(getattr(settings, "UPLOAD_MALWARE_SCAN_TIMEOUT_S", 30.0)),
        fail_closed=bool(getattr(settings, "UPLOAD_MALWARE_SCAN_FAIL_CLOSED", True)),
    )


def validate_upload(
    filename: str | None,
    contents: bytes | None,
    *,
    content_type: str | None = None,
    limits: UploadLimits | None = None,
    allowed_kinds: Iterable[str] = ("pdf", "csv"),
    malware_scanner: MalwareScanner | None = None,
) -> ValidatedUpload:
    """Validate one upload and return its checksum and structural metadata."""
    limits = limits or UploadLimits()
    safe_name, kind = validate_filename(
        filename, limits=limits, allowed_kinds=allowed_kinds
    )
    data = _validate_size(contents, safe_name, kind, limits)
    scan_result = MalwareScanResult("disabled")
    if malware_scanner is not None:
        try:
            scan_result = malware_scanner(safe_name, data)
        except UploadValidationError:
            raise
        except Exception as exc:
            raise UploadValidationError(
                "Malware scanner failed; upload rejected",
                code="malware_scanner_failed",
                status_code=503,
            ) from exc
        if not isinstance(scan_result, MalwareScanResult):
            raise RuntimeError("malware scanner must return MalwareScanResult")

    details = (
        _validate_pdf(data, safe_name, limits)
        if kind == "pdf"
        else _validate_csv(data, safe_name, limits)
    )
    return ValidatedUpload(
        filename=safe_name,
        kind=kind,
        contents=data,
        content_type=content_type,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        details=details,
        malware_scan=scan_result,
    )


def validate_upload_manifest(
    uploads: Iterable[tuple[str | None, int]],
    *,
    limits: UploadLimits | None = None,
    allowed_kinds: Iterable[str] = ("pdf", "csv"),
) -> list[tuple[str, str]]:
    """Preflight names/types/sizes before callers copy upload bodies into memory."""
    limits = limits or UploadLimits()
    entries = list(uploads)
    allowed_kinds = tuple(allowed_kinds)
    if not entries:
        raise UploadValidationError("Choose at least one file", code="empty_batch")
    if len(entries) > _positive_limit(limits.max_files, "UPLOAD_MAX_FILES"):
        raise UploadValidationError(
            f"Too many files (max {limits.max_files})", code="too_many_files"
        )

    manifest: list[tuple[str, str]] = []
    names: set[str] = set()
    total_bytes = 0
    for filename, raw_size in entries:
        safe_name, kind = validate_filename(
            filename, limits=limits, allowed_kinds=allowed_kinds
        )
        try:
            size = int(raw_size)
        except (TypeError, ValueError) as exc:
            raise UploadValidationError(
                f"Could not determine the size of '{safe_name}'",
                code="invalid_file_size",
            ) from exc
        if size < 0:
            raise UploadValidationError(
                f"Could not determine the size of '{safe_name}'",
                code="invalid_file_size",
            )
        max_bytes = _positive_limit(limits.max_bytes_for(kind), f"max_{kind}_bytes")
        if size > max_bytes:
            raise UploadTooLargeError(
                f"'{safe_name}' is too large (max {max_bytes // MIB} MB)"
            )
        total_bytes += size
        folded_name = safe_name.casefold()
        if folded_name in names:
            raise DuplicateUploadError(f"Filename '{safe_name}' appears more than once")
        names.add(folded_name)
        manifest.append((safe_name, kind))

    if total_bytes > _positive_limit(limits.max_batch_bytes, "UPLOAD_MAX_BATCH_BYTES"):
        raise UploadTooLargeError(
            f"Upload batch is too large (max {limits.max_batch_bytes // MIB} MB)"
        )
    kinds = {kind for _name, kind in manifest}
    if len(kinds) != 1:
        raise UploadValidationError(
            "Do not mix PDF and CSV files in one session",
            code="mixed_upload_types",
        )
    return manifest


def validate_upload_batch(
    uploads: Iterable[tuple[str | None, bytes | None, str | None]],
    *,
    limits: UploadLimits | None = None,
    allowed_kinds: Iterable[str] = ("pdf", "csv"),
    malware_scanner: MalwareScanner | None = None,
) -> list[ValidatedUpload]:
    """Validate a homogeneous PDF or relational CSV bundle."""
    limits = limits or UploadLimits()
    payloads = list(uploads)
    manifest = validate_upload_manifest(
        [(filename, len(contents or b"")) for filename, contents, _type in payloads],
        limits=limits,
        allowed_kinds=allowed_kinds,
    )

    validated = [
        validate_upload(
            safe_name,
            contents,
            content_type=content_type,
            limits=limits,
            allowed_kinds=(kind,),
            malware_scanner=malware_scanner,
        )
        for (safe_name, kind), (_filename, contents, content_type) in zip(
            manifest, payloads, strict=True
        )
    ]
    checksums: set[str] = set()
    for item in validated:
        if item.sha256 in checksums:
            raise DuplicateUploadError(
                f"File '{item.filename}' duplicates another file in this batch"
            )
        checksums.add(item.sha256)
    return validated


async def read_upload_limited(
    upload_file: object, max_bytes: int, *, chunk_bytes: int = READ_CHUNK_BYTES
) -> bytes:
    """Read a FastAPI/Starlette-style upload incrementally with a hard cap."""
    max_bytes = _positive_limit(int(max_bytes), "upload max bytes")
    chunk_bytes = _positive_limit(int(chunk_bytes), "upload chunk bytes")
    chunks: list[bytes] = []
    total = 0
    while True:
        # Read at most one byte beyond the cap so oversized requests never get
        # copied into an unbounded in-memory object by application code.
        remaining = max_bytes - total
        chunk = await upload_file.read(min(chunk_bytes, remaining + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise UploadTooLargeError(f"File is too large (max {max_bytes // MIB} MB)")
        chunks.append(bytes(chunk))
    return b"".join(chunks)


def save_upload_batch_atomic(
    uploads: Sequence[ValidatedUpload], target_dir: Path
) -> list[Path]:
    """Persist a validated batch by atomically renaming a temporary directory."""
    if not uploads:
        raise UploadValidationError("No validated files to save", code="empty_batch")
    target_dir = Path(target_dir)
    parent = target_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists():
        raise UploadValidationError(
            f"Upload directory already exists for '{target_dir.name}'",
            code="upload_destination_exists",
            status_code=409,
        )

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{target_dir.name}-upload-", dir=parent))
    try:
        for item in uploads:
            destination = temp_dir / item.filename
            # Validated filenames have no path component; retain a defense in
            # depth check at the filesystem boundary.
            if destination.parent.resolve() != temp_dir.resolve():
                raise UploadValidationError(
                    "Unsafe upload destination", code="unsafe_filename"
                )
            with destination.open("xb") as handle:
                handle.write(item.contents)
        os.replace(temp_dir, target_dir)
        return [target_dir / item.filename for item in uploads]
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def rollback_upload_directory(target_dir: Path) -> None:
    """Remove a newly-created upload directory after downstream registration fails."""
    shutil.rmtree(Path(target_dir), ignore_errors=True)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_duplicate_uploads(
    uploads: Sequence[ValidatedUpload],
    records: Iterable[Mapping[str, Any]],
    *,
    project_root: Path,
) -> list[dict[str, str]]:
    """Find content already registered by another custom session.

    New registry records carry checksums. For backward compatibility, old
    records are hashed from their source files when those files still exist.
    """
    wanted = {item.sha256: item.filename for item in uploads}
    duplicates: list[dict[str, str]] = []
    for record in records:
        session_name = str(record.get("name") or "unknown")
        known: dict[str, str] = {}
        for metadata in record.get("uploads") or []:
            checksum = str(metadata.get("sha256") or "")
            if checksum:
                known[checksum] = str(metadata.get("filename") or "unknown")
        if not known:
            for source in record.get("sources") or []:
                path = Path(str(source))
                if not path.is_absolute():
                    path = Path(project_root) / path
                try:
                    if path.is_file():
                        known[_hash_file(path)] = path.name
                except OSError:
                    continue
        for checksum, existing_name in known.items():
            if checksum in wanted:
                duplicates.append(
                    {
                        "filename": wanted[checksum],
                        "session": session_name,
                        "existing_filename": existing_name,
                        "sha256": checksum,
                    }
                )
    return duplicates


def ensure_no_duplicate_uploads(
    uploads: Sequence[ValidatedUpload],
    records: Iterable[Mapping[str, Any]],
    *,
    project_root: Path,
) -> None:
    duplicates = find_duplicate_uploads(uploads, records, project_root=project_root)
    if duplicates:
        first = duplicates[0]
        raise DuplicateUploadError(
            f"'{first['filename']}' was already uploaded in session "
            f"'{first['session']}'"
        )


def verify_recorded_checksums(
    uploads: Sequence[ValidatedUpload], metadata: Iterable[Mapping[str, Any]]
) -> None:
    """Reject stored source files modified after their registry entry was written."""
    expected = {
        str(item.get("filename")): str(item.get("sha256"))
        for item in metadata
        if item.get("filename") and item.get("sha256")
    }
    for upload in uploads:
        checksum = expected.get(upload.filename)
        if expected and not checksum:
            raise UploadValidationError(
                f"Stored upload '{upload.filename}' is missing checksum metadata",
                code="checksum_metadata_missing",
            )
        if checksum and not hmac.compare_digest(checksum, upload.sha256):
            raise UploadValidationError(
                f"Stored upload '{upload.filename}' failed checksum verification",
                code="checksum_mismatch",
            )


def revalidate_persisted_uploads(
    paths: Sequence[str | Path],
    *,
    allowed_root: str | Path,
    recorded_metadata: Iterable[Mapping[str, Any]] = (),
    limits: UploadLimits | None = None,
    malware_scanner: MalwareScanner | None = None,
) -> list[ValidatedUpload]:
    """Reapply the complete upload contract before persisted data is ingested."""
    limits = limits or UploadLimits()
    root = Path(allowed_root).resolve()
    payloads: list[tuple[str, bytes, None]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise UploadValidationError(
                f"Stored source is outside the allowed upload directory: {path.name}",
                code="unsafe_stored_path",
            ) from exc
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise UploadValidationError(
                f"Stored source cannot be read: {path.name}",
                code="stored_file_error",
            ) from exc
        safe_name, kind = validate_filename(path.name, limits=limits)
        max_size = limits.max_bytes_for(kind)
        if size > max_size:
            raise UploadTooLargeError(
                f"Stored source '{safe_name}' is too large (max {max_size // MIB} MB)"
            )
        try:
            contents = path.read_bytes()
        except OSError as exc:
            raise UploadValidationError(
                f"Stored source cannot be read: {safe_name}",
                code="stored_file_error",
            ) from exc
        payloads.append((safe_name, contents, None))
    validated = validate_upload_batch(
        payloads, limits=limits, malware_scanner=malware_scanner,
    )
    verify_recorded_checksums(validated, recorded_metadata)
    return validated


def record_upload_audit(
    *,
    audit_path: str | Path,
    outcome: str,
    actor: str,
    tenant_id: str,
    entry_point: str,
    uploads: Sequence[ValidatedUpload] = (),
    session_name: str | None = None,
    reason: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a content-free upload audit event and return it."""
    event: dict[str, Any] = {
        "audit_id": uuid.uuid4().hex[:16],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "upload",
        "outcome": outcome,
        "actor": actor,
        "tenant_id": tenant_id,
        "entry_point": entry_point,
        "session_name": session_name,
        "files": [item.metadata() for item in uploads],
    }
    if reason:
        event["reason"] = reason
    if extra:
        event["metadata"] = dict(extra)

    path = Path(audit_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
    with _AUDIT_LOCK:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
    return event


def safe_record_upload_audit(**kwargs: Any) -> dict[str, Any] | None:
    """Best-effort audit write that never corrupts an already-applied graph update."""
    try:
        return record_upload_audit(**kwargs)
    except Exception:
        logger.exception("could not write upload audit event")
        return None
