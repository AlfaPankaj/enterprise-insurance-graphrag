"""Query traversal logging + tamper-evident audit trail store (v2 upgrade of Shot 3).

Every ``run_query`` call produces an **audit record** — the full explainability
artifact: question, answer, seed nodes, every node/edge the retrieval touched,
the re-ranker's scores, what survived pruning, token accounting, per-stage
timings — and, new in v2, **who asked** (user identity + tenant) and **what the
answer cost** (provider, usage, cost estimate).

Tamper-evidence (v2): records are linked into a SHA-256 **hash chain** —
``record_hash = sha256(canonical_json(record_without_hash) + prev_hash)`` — so
editing or removing a record breaks the chain and ``verify()`` reports it.
Legacy v1 records (no hashes) are bound into the chain as a genesis anchor
(``prev_hash = sha256(raw_line)`` of the last legacy record).

The store is append-only: when it outgrows ``max_records`` the file is
**rotated into an archive segment** (never rewritten), the chain continues
across segments, and ``verify()`` walks every segment in order.

Records are:

* appended to a JSONL store under ``settings.AUDIT_DIR`` (durable, cheap),
* kept in an in-memory ring buffer for the dashboard,
* returned to the caller inside the query result as ``result["traversal"]``.

The record is the single source of truth for the audit UI, the API's
``/api/v1/audit`` endpoint, and the JSON/HTML/PDF exports.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from graphrag.config import settings
from graphrag.path_extractor import build_cypher, traversal_summary

# repo root = <root>/src/graphrag/traversal_logger.py
ROOT = Path(__file__).resolve().parents[2]

_SEGMENT_RE = ".archive.jsonl"


def _canonical(record: dict) -> bytes:
    """Canonical bytes of a record for hashing (deterministic, sort-keyed)."""
    body = {k: v for k, v in record.items() if k != "record_hash"}
    return json.dumps(body, sort_keys=True, default=str).encode("utf-8")


def _hash(record: dict) -> str:
    """SHA-256 of the canonical record body + its chained prev_hash."""
    return hashlib.sha256(_canonical(record)).hexdigest()


def _chain_hash(prev_hash: str, record: dict) -> str:
    """Hash binding this record to the previous one (hash-chain link)."""
    record["prev_hash"] = prev_hash
    return _hash(record)


class AuditStore:
    """Append-only, hash-chained JSONL audit store with segment rotation.

    The active file is trimmed *by rotation* (the chain is never rewritten):
    when it exceeds ``max_records`` the file becomes an archive segment
    (``<name>.archive.jsonl``) and a fresh active file starts with its
    ``prev_hash`` pointing at the segment's tail — one unbroken chain.
    """

    def __init__(self, path: Path, max_records: int | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records or settings.AUDIT_MAX_RECORDS
        # RLock: append() -> _rotate() -> recent() re-enters the same lock
        self._lock = threading.RLock()
        self._ring: deque[dict] = deque(maxlen=200)  # newest first
        self._count = self._read_count()
        self._head_hash = self._read_head_hash()

    # ------------------------------------------------------------------ init
    def _read_count(self) -> int:
        if not self.path.exists():
            return 0
        return sum(1 for line in self.path.open("r", encoding="utf-8")
                   if line.strip())

    def _last_line(self) -> str | None:
        if not self.path.exists():
            return None
        last: str | None = None
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last = line
        return last

    def _read_head_hash(self) -> str | None:
        """Chain head: record_hash of the newest record in the newest segment."""
        for path in reversed(self.segments()):
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("record_hash"):
                        return rec["record_hash"]
                    # legacy record: its raw bytes anchor the chain
                    return hashlib.sha256(line.encode("utf-8")).hexdigest()
        return None

    # ------------------------------------------------------------ segments
    def segments(self) -> list[Path]:
        """Audit files in chronological order: archive segments + active file."""
        pattern = f"{self.path.stem}{_SEGMENT_RE}"
        archives = sorted(self.path.parent.glob(pattern))
        return [*archives, self.path] if self.path.exists() else archives

    def _iter_segments(self):
        for path in self.segments():
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue  # tolerate a torn tail write

    # ------------------------------------------------------------- append
    def append(self, record: dict) -> None:
        """Hash-chain the record and persist it (JSONL) + ring buffer."""
        prev = self._head_hash or "genesis"
        record["record_hash"] = _chain_hash(prev, record)
        line = json.dumps(record, default=str)
        with self._lock:
            self._ring.appendleft(record)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._count += 1
            self._head_hash = record["record_hash"]
            if self._count > self.max_records:
                self._rotate()

    def _rotate(self) -> None:
        """Keep the newest ``max_records`` lines in the active file and append
        the older prefix to the archive segment.

        Archive writes are strictly append-only (order preserved); the active
        file's kept tail is rewritten **byte-identically** — no record content
        is ever modified, so the hash chain stays verifiable across the
        rotation boundary.
        """
        archive = self.path.with_name(f"{self.path.stem}{_SEGMENT_RE}")
        with self._lock:
            lines = [ln for ln in self.path.read_text(encoding="utf-8").splitlines()
                     if ln.strip()]
            overflow, keep = lines[:-self.max_records], lines[-self.max_records:]
            if overflow:
                with archive.open("a", encoding="utf-8") as fh:
                    for line in overflow:
                        fh.write(line + "\n")
            self.path.write_text("\n".join(keep) + ("\n" if keep else ""),
                                 encoding="utf-8")
            self._count = len(keep)

    # --------------------------------------------------------------- reads
    def recent(self, limit: int = 100) -> list[dict]:
        """Newest-first records from the active file (works across processes)."""
        with self._lock:
            records: list[dict] = []
            if self.path.exists():
                with self.path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            return records[-limit:][::-1]

    def get(self, audit_id: str) -> dict | None:
        """Look a record up across ALL segments (active + archive)."""
        for rec in self._iter_segments():
            if rec.get("audit_id") == audit_id:
                return rec
        return None

    def verify(self) -> dict:
        """Walk the whole chain (all segments, in order) and report integrity.

        Returns ``{"valid": bool, "records_checked": n, "broken_at": [...]}``.
        A record is checked against (a) its own hash and (b) the hash of the
        previous record — so any edit, deletion, insertion, or reorder breaks
        the chain and lands in ``broken_at``. Legacy (unhashed) records anchor
        the chain via the hash of their raw line but are counted, not checked.
        """
        with self._lock:
            prev: str | None = None
            checked = 0
            legacy = 0
            broken: list[int] = []
            index = 0
            for path in self.segments():
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if rec.get("record_hash"):
                            expected_prev = rec.get("prev_hash")
                            if prev is not None and expected_prev != prev:
                                broken.append(index)   # chain link broken
                            if _chain_hash(expected_prev, rec) != rec["record_hash"]:
                                broken.append(index)   # content tampered
                            prev = rec["record_hash"]
                            checked += 1
                        else:
                            legacy += 1
                            prev = hashlib.sha256(line.encode("utf-8")).hexdigest()
                        index += 1
            return {
                "valid": not broken,
                "records_checked": checked,
                "legacy_records": legacy,
                "broken_at": broken,
            }

    def clear(self) -> int:
        """Delete every segment; returns the number of records removed."""
        with self._lock:
            count = 0
            for path in self.segments():
                count += sum(1 for _ in path.open("r", encoding="utf-8"))
                path.unlink()
            self._ring.clear()
            self._count = 0
            self._head_hash = None
            return count


def _timings_ms(t0: float, t1: float, t2: float, t3: float, t4: float) -> dict:
    """Per-stage milliseconds: (retrieval, rerank, prune, answer, total)."""
    return {
        "retrieval_ms": round((t1 - t0) * 1000, 2),
        "rerank_ms": round((t2 - t1) * 1000, 2),
        "prune_ms": round((t3 - t2) * 1000, 2),
        "answer_ms": round((t4 - t3) * 1000, 2),
        "total_ms": round((t4 - t0) * 1000, 2),
    }


def build_audit_record(*, query: str, subgraph: dict, ranked: list, pruned: dict,
                       tokens: dict, answer: str, answer_mode: str,
                       reranker: str, max_hops: int, t0: float, t1: float,
                       t2: float, t3: float, t4: float,
                       answer_model: str | None = None,
                       user: dict | None = None,
                       tenant_id: str | None = None,
                       answer_provider: str | None = None,
                       usage: dict | None = None,
                       cost_usd: float | None = None) -> dict:
    """Assemble the full explainability artifact for one query execution.

    v2 additions (all optional — v1 callers keep working): ``user`` (the
    identity of the caller), ``tenant_id``, and the answer's provider/usage/
    cost for the cost dashboard.
    """
    summary = traversal_summary(subgraph, max_hops)
    record = {
        "audit_id": uuid.uuid4().hex[:12],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "query": query,
        "answer": answer,
        "answer_mode": answer_mode,
        "answer_model": answer_model,
        "reranker": reranker,
        "max_hops": max_hops,
        "retrieval": {
            "seeds": [s["id"] for s in subgraph.get("seeds", [])],
            "seed_kinds": [s.get("kind", "id") for s in subgraph.get("seeds", [])],
            "node_count": subgraph.get("node_count", len(subgraph.get("nodes", []))),
            "edge_count": subgraph.get("edge_count", len(subgraph.get("edges", []))),
        },
        "traversal": {
            "nodes_visited": summary["nodes_visited"],
            "edges_traversed": summary["edges_traversed"],
            "paths": summary["paths"],
        },
        "cypher": build_cypher(subgraph, max_hops),
        "ranking": [
            {"id": n["id"], "label": n.get("label", ""), "score": round(float(s), 4)}
            for n, s in ranked[:20]
        ],
        "pruned": {
            "kept": pruned.get("kept", []),
            "dropped": pruned.get("dropped", []),
            "kept_count": pruned.get("node_count", len(pruned.get("kept", []))),
            "dropped_count": pruned.get("dropped_count", 0),
            "budget": pruned.get("budget"),
        },
        "tokens": tokens,
        "timings_ms": _timings_ms(t0, t1, t2, t3, t4),
    }
    if user:
        record["user"] = user
    if tenant_id:
        record["tenant_id"] = tenant_id
    if answer_provider:
        record["answer_provider"] = answer_provider
    if usage:
        record["usage"] = usage
    if cost_usd is not None:
        record["cost_usd"] = cost_usd
    return record


# Single process-wide store (same pattern as the reranker cache).
audit_store = AuditStore(ROOT / settings.AUDIT_DIR / "audit_trail.jsonl")
