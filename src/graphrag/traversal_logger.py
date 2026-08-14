"""Query traversal logging + audit trail store (Phase 4, Shot 3).

Every ``run_query`` call produces an **audit record** — the full explainability
artifact: question, answer, seed nodes, every node/edge the retrieval touched,
the re-ranker's scores, what survived pruning, token accounting, and per-stage
timings. Records are:

  * appended to a JSONL store under ``settings.AUDIT_DIR`` (durable, cheap),
  * kept in an in-memory ring buffer for the dashboard,
  * returned to the caller inside the query result as ``result["traversal"]``.

The record is the single source of truth for the audit UI, the API's
``/api/v1/audit`` endpoint, and the JSON/HTML/PDF exports.
"""

from __future__ import annotations

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


class AuditStore:
    """Append-only JSONL store of audit records, with a recent-record ring.

    The file is trimmed to ``max_records`` on append (a bounded, rotating
    trail — audit history is capped so the store cannot grow without limit).
    """

    def __init__(self, path: Path, max_records: int | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records or settings.AUDIT_MAX_RECORDS
        # RLock: append() -> _trim() -> recent() re-enters the same lock
        self._lock = threading.RLock()
        self._ring: deque[dict] = deque(maxlen=200)  # newest first
        self._count = self._read_count()

    def _read_count(self) -> int:
        if not self.path.exists():
            return 0
        count = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
        return count

    def append(self, record: dict) -> None:
        """Persist one record (JSONL) and pin it in the ring buffer."""
        line = json.dumps(record, default=str)
        with self._lock:
            self._ring.appendleft(record)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._count += 1
            if self._count > self.max_records:
                self._trim()

    def _trim(self) -> None:
        """Rewrite the store keeping only the newest ``max_records`` lines."""
        keep = self.recent(self.max_records)[::-1]  # oldest -> newest
        with self.path.open("w", encoding="utf-8") as fh:
            for r in keep:
                fh.write(json.dumps(r, default=str) + "\n")
        self._count = len(keep)

    def recent(self, limit: int = 100) -> list[dict]:
        """Newest-first records, from the file (works across processes)."""
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
                            continue  # tolerate a torn tail write
            return records[-limit:][::-1]

    def get(self, audit_id: str) -> dict | None:
        return next((r for r in self.recent(limit=10_000) if r.get("audit_id") == audit_id), None)

    def clear(self) -> int:
        with self._lock:
            count = 0
            if self.path.exists():
                count = sum(1 for _ in self.path.open("r", encoding="utf-8"))
                self.path.unlink()
            self._ring.clear()
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
                       answer_model: str | None = None) -> dict:
    """Assemble the full explainability artifact for one query execution."""
    summary = traversal_summary(subgraph, max_hops)
    return {
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


# Single process-wide store (same pattern as the reranker cache).
audit_store = AuditStore(ROOT / settings.AUDIT_DIR / "audit_trail.jsonl")
