"""Reversible entity resolution with conservative automatic matching.

Exact matches on normalized, high-signal keys (email/phone) may be merged
automatically. Fuzzy name/address matches are emitted as review candidates and
are never merged unless their pair appears in ``accepted_merges``. Every merge
retains source IDs and a decision record so it can be audited or reversed.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher


@dataclass(frozen=True)
class ResolutionDecision:
    left_id: str
    right_id: str
    canonical_id: str
    score: float
    method: str
    status: str
    fields: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "canonical_id": self.canonical_id,
            "score": self.score,
            "method": self.method,
            "status": self.status,
            "fields": list(self.fields),
        }


@dataclass
class ResolutionResult:
    records: list[dict]
    decisions: list[ResolutionDecision] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)


def normalize_text(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def normalize_email(value: object) -> str:
    return str(value or "").strip().casefold()


def normalize_phone(value: object) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    # Country-code and local spellings should block together. Keep at least
    # seven digits to avoid merging on extensions or malformed numbers.
    return digits[-10:] if len(digits) >= 10 else (digits if len(digits) >= 7 else "")


def normalize_address(value: object) -> str:
    value = normalize_text(value)
    replacements = {
        r"\bstreet\b": "st", r"\broad\b": "rd", r"\bavenue\b": "ave",
        r"\bapartment\b": "apt", r"\bsuite\b": "ste",
    }
    for pattern, replacement in replacements.items():
        value = re.sub(pattern, replacement, value)
    return value


def normalize_value(field: str, value: object) -> str:
    field = field.casefold()
    if "email" in field:
        return normalize_email(value)
    if any(token in field for token in ("phone", "mobile", "telephone")):
        return normalize_phone(value)
    if "address" in field:
        return normalize_address(value)
    return normalize_text(value)


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def _fuzzy_score(a: dict, b: dict, fields: Iterable[str]) -> tuple[float, tuple[str, ...]]:
    scores: list[float] = []
    used: list[str] = []
    for field_name in fields:
        left = normalize_value(field_name, a.get(field_name))
        right = normalize_value(field_name, b.get(field_name))
        if left and right:
            scores.append(SequenceMatcher(None, left, right).ratio())
            used.append(field_name)
    return ((sum(scores) / len(scores)) if scores else 0.0), tuple(used)


def resolve_records(
    records: list[dict],
    *,
    id_field: str = "id",
    exact_keys: Iterable[str] = ("email", "phone"),
    fuzzy_fields: Iterable[str] = ("name", "address"),
    blocking_fields: Iterable[str] = ("postal_code", "postcode", "zip"),
    fuzzy_threshold: float = 0.92,
    accepted_merges: Iterable[Iterable[str]] = (),
    rejected_merges: Iterable[Iterable[str]] = (),
    auto_exact: bool = True,
) -> ResolutionResult:
    """Resolve an in-memory record batch while retaining full provenance.

    This pure function is used by profiling/review tests. The streaming loader
    applies the same exact-key rules through a disk-backed SQLite index.
    """
    accepted = {_pair_key(*pair) for pair in accepted_merges if len(tuple(pair)) == 2}
    rejected = {_pair_key(*pair) for pair in rejected_merges if len(tuple(pair)) == 2}
    by_id = {str(row[id_field]): dict(row) for row in records}
    parent = {eid: eid for eid in by_id}
    decisions: list[ResolutionDecision] = []

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def merge(left: str, right: str) -> str:
        lroot, rroot = find(left), find(right)
        canonical = min(lroot, rroot)
        parent[lroot] = canonical
        parent[rroot] = canonical
        return canonical

    exact_index: dict[tuple[str, str], str] = {}
    for eid, row in by_id.items():
        for field_name in exact_keys:
            value = normalize_value(field_name, row.get(field_name))
            if not value:
                continue
            key = (field_name, value)
            other = exact_index.get(key)
            if other and other != eid:
                pair = _pair_key(other, eid)
                allowed = pair not in rejected and (auto_exact or pair in accepted)
                canonical = merge(other, eid) if allowed else min(pair)
                decisions.append(ResolutionDecision(
                    other, eid, canonical, 1.0, f"exact:{field_name}",
                    "accepted" if allowed else "rejected", (field_name,),
                ))
            else:
                exact_index[key] = eid

    # Candidate generation is blocked by coarse geography when available, or
    # by the first normalized name character. This avoids an O(N²) global scan.
    blocks: dict[str, list[str]] = {}
    for eid, row in by_id.items():
        block = ""
        for field_name in blocking_fields:
            block = normalize_value(field_name, row.get(field_name))
            if block:
                break
        if not block:
            name = normalize_value("name", row.get("name"))
            block = name[:1] if name else hashlib.sha256(eid.encode()).hexdigest()[:4]
        blocks.setdefault(block, []).append(eid)

    for ids in blocks.values():
        for index, left in enumerate(ids):
            for right in ids[index + 1:]:
                if find(left) == find(right):
                    continue
                score, fields = _fuzzy_score(by_id[left], by_id[right], fuzzy_fields)
                if score < fuzzy_threshold:
                    continue
                pair = _pair_key(left, right)
                if pair in rejected:
                    status = "rejected"
                    canonical = min(pair)
                elif pair in accepted:
                    status = "accepted"
                    canonical = merge(left, right)
                else:
                    status = "pending_review"
                    canonical = min(pair)
                decisions.append(ResolutionDecision(
                    left, right, canonical, round(score, 4), "fuzzy", status, fields,
                ))

    aliases = {eid: find(eid) for eid in by_id if find(eid) != eid}
    merged: dict[str, dict] = {}
    for eid, row in by_id.items():
        canonical = find(eid)
        target = merged.setdefault(canonical, {id_field: canonical, "source_ids": []})
        target["source_ids"].append(eid)
        # Canonical non-empty values win; aliases fill gaps rather than silently
        # overwriting conflicting evidence.
        for key, value in row.items():
            if key == id_field:
                continue
            if target.get(key) in (None, "") and value not in (None, ""):
                target[key] = value
    return ResolutionResult(list(merged.values()), decisions, aliases)


def find_csv_candidates(path, file_mapping: dict, *, threshold: float = 0.92,
                        limit: int = 5000) -> list[dict]:
    """Disk-backed fuzzy candidate generation for a CSV review queue."""
    import csv
    import sqlite3
    import tempfile
    from pathlib import Path

    reverse = {target: source for source, target
               in file_mapping.get("properties", {}).items()}
    id_column = file_mapping.get("id_column")
    name_column = next((reverse[key] for key in reverse if "name" in key), None)
    address_column = next((reverse[key] for key in reverse if "address" in key), None)
    postal_column = next((reverse[key] for key in reverse
                          if any(part in key for part in ("postal", "postcode", "zip"))), None)
    if not name_column or id_column == "__generated_id__":
        return []
    with tempfile.NamedTemporaryFile(
        prefix="graphrag-er-", suffix=".db", delete=False
    ) as tmp:
        db_path = Path(tmp.name)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE records(id TEXT PRIMARY KEY,name TEXT,address TEXT,block TEXT)")
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                eid = str(row.get(id_column) or "").strip()
                name = normalize_text(row.get(name_column))
                if not eid or not name:
                    continue
                address = normalize_address(row.get(address_column)) if address_column else ""
                postal = normalize_text(row.get(postal_column)) if postal_column else ""
                block = postal or name[:1]
                conn.execute("INSERT OR IGNORE INTO records VALUES (?,?,?,?)",
                             (eid, name, address, block))
        conn.execute("CREATE INDEX records_block ON records(block)")
        conn.commit()
        candidates = []
        rows = conn.execute(
            "SELECT a.id,a.name,a.address,b.id,b.name,b.address FROM records a "
            "JOIN records b ON a.block=b.block AND a.id<b.id LIMIT ?",
            (max(limit * 20, limit),),
        )
        for left_id, left_name, left_address, right_id, right_name, right_address in rows:
            scores = [SequenceMatcher(None, left_name, right_name).ratio()]
            fields = ["name"]
            if left_address and right_address:
                scores.append(SequenceMatcher(None, left_address, right_address).ratio())
                fields.append("address")
            score = sum(scores) / len(scores)
            if score >= threshold:
                candidates.append({
                    "left_id": left_id, "right_id": right_id,
                    "score": round(score, 4), "method": "fuzzy",
                    "status": "pending_review", "fields": fields,
                })
                if len(candidates) >= limit:
                    break
        return candidates
    finally:
        conn.close()
        db_path.unlink(missing_ok=True)


def reverse_alias(aliases: dict[str, str], source_id: str) -> dict[str, str]:
    """Return a copy with one accepted alias removed (audit-friendly undo)."""
    out = dict(aliases)
    out.pop(str(source_id), None)
    return out
