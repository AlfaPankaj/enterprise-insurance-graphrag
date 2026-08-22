"""Extraction review queue (v2 — WS-C, G17).

Low-confidence extractions are **held** instead of written to the graph:
a human reviews each item and approves (apply via the normal CDC machinery +
snapshot update) or rejects it. The queue lives in SQLite
(``settings.REVIEW_DB_PATH``), survives restarts, and is exposed through
``/api/v1/review`` + a Streamlit page.

Lifecycle: ``pending → approved | rejected``. A pending item is keyed by
``(doc_id, entity_id)`` — re-uploading the same document updates the pending
item instead of duplicating it, so a user can iterate on a document until the
extraction is clean.

``partition_entities`` splits an extraction by confidence; ``apply_review_item``
is the approve path — it merges the entity into the document snapshot and
runs ``update_graph_surgically`` (atomic: graph + snapshot + cache-revision
bump in one transaction). CDC therefore only ever sees **confirmed** changes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from graphrag.config import settings
from graphrag.prometheus import review_decisions_total, review_held_total, \
    review_pending

logger = logging.getLogger("graphrag.review")

ROOT = Path(__file__).resolve().parents[2]

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"

_MAX_PROPS_PREVIEW = 24


class ReviewStore:
    """SQLite-backed review queue. Thread-safe; one store per process."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS review ("
            " id TEXT PRIMARY KEY,"
            " doc_id TEXT NOT NULL,"
            " source_file TEXT,"
            " label TEXT NOT NULL,"
            " entity_id TEXT NOT NULL,"
            " props TEXT NOT NULL,"
            " confidence REAL NOT NULL,"
            " reasons TEXT NOT NULL DEFAULT '[]',"
            " status TEXT NOT NULL,"
            " created_at TEXT NOT NULL,"
            " decided_at TEXT,"
            " decided_by TEXT)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS review_doc_entity "
            "ON review (doc_id, entity_id)")
        self._conn.commit()

    # ------------------------------------------------------------- queries
    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> dict:
        item = dict(zip(row.keys(), row))
        item["props"] = json.loads(item["props"] or "{}")
        item["reasons"] = json.loads(item["reasons"] or "[]")
        return item

    def get(self, review_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM review WHERE id=?", (review_id,)).fetchone()
        return self._row_to_item(row) if row else None

    def list(self, status: str | None = None, limit: int = 100) -> list[dict]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM review WHERE status=? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (status, int(limit))).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM review ORDER BY created_at DESC LIMIT ?",
                    (int(limit),)).fetchall()
        return [self._row_to_item(r) for r in rows]

    def summary(self) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, count(*) AS c FROM review GROUP BY status"
            ).fetchall()
        out = {STATUS_PENDING: 0, STATUS_APPROVED: 0, STATUS_REJECTED: 0}
        for row in rows:
            out[row["status"]] = row["c"]
        return out

    def clear(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM review")

    # ------------------------------------------------------------- lifecycle
    def submit(self, doc_id: str, source_file: str | None, label: str,
               entity_id: str, props: dict, confidence: float,
               reasons: list[str] | None = None) -> tuple[str, bool]:
        """Hold one low-confidence entity for review.

        Returns ``(review_id, created)`` — a pending item for the same
        (doc, entity) is **updated** (fresh props/confidence) rather than
        duplicated, so repeated uploads iterate on one review entry.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._lock:
            existing = self._conn.execute(
                "SELECT id FROM review WHERE doc_id=? AND entity_id=? "
                "AND status=? LIMIT 1",
                (doc_id, entity_id, STATUS_PENDING)).fetchone()
            if existing:
                with self._conn:
                    self._conn.execute(
                        "UPDATE review SET props=?, confidence=?, reasons=?, "
                        "source_file=? WHERE id=?",
                        (json.dumps(props, default=str), float(confidence),
                         json.dumps(reasons or []), source_file, existing["id"]))
                return existing["id"], False
            review_id = uuid.uuid4().hex[:12]
            with self._conn:
                self._conn.execute(
                    "INSERT INTO review (id, doc_id, source_file, label, "
                    "entity_id, props, confidence, reasons, status, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (review_id, doc_id, source_file, label, entity_id,
                     json.dumps(props, default=str), float(confidence),
                     json.dumps(reasons or []), STATUS_PENDING, now))
            review_pending.inc()
            review_held_total.inc()
            return review_id, True

    def decide(self, review_id: str, decision: str,
               decided_by: str | None = None) -> dict | None:
        """Approve/reject a PENDING item; returns the updated item (or None).

        ``decision`` in {approved, rejected}. Only a pending item can be
        decided — decided items are final (no re-decision).
        """
        if decision not in (STATUS_APPROVED, STATUS_REJECTED):
            raise ValueError(f"decision must be approved or rejected, got {decision!r}")
        with self._lock:
            row = self._conn.execute(
                "SELECT id, status FROM review WHERE id=?", (review_id,)).fetchone()
            if row is None:
                return None
            if row["status"] != STATUS_PENDING:
                return None
            with self._conn:
                self._conn.execute(
                    "UPDATE review SET status=?, decided_at=?, decided_by=? "
                    "WHERE id=?",
                    (decision, time.strftime("%Y-%m-%dT%H:%M:%S"), decided_by,
                     review_id))
        review_pending.dec()
        review_decisions_total.inc(decision=decision)
        return self.get(review_id)


# ---------------------------------------------------------------------------
# partition + approve (the CDC integration)
# ---------------------------------------------------------------------------

def partition_entities(entities: dict, confidence: dict, threshold: float):
    """Split an extraction by confidence.

    Returns ``(to_apply, held)`` — ``to_apply`` keeps the ``{label: {id:
    props}}`` shape for the CDC path; ``held`` is a list of review items.
    """
    to_apply: dict = {}
    held: list[dict] = []
    for label, ents in entities.items():
        for eid, props in ents.items():
            score = confidence.get(label, {}).get(eid, 1.0)
            if score >= threshold:
                to_apply.setdefault(label, {})[eid] = props
            else:
                held.append({"label": label, "entity_id": eid, "props": props,
                             "confidence": score})
    return to_apply, held


def apply_review_item(driver, item: dict) -> dict:
    """Approve path: merge the entity into its document snapshot and apply
    the CDC add — atomically (graph + snapshot + revision bump in one
    transaction, same contract as the normal upload path).
    """
    from graphrag.graph_store import get_existing_entities
    from graphrag.graph_updater import update_graph_surgically

    doc_id = item["doc_id"]
    label = item["label"]
    entity_id = item["entity_id"]

    with driver.session() as session:
        old_snapshot = get_existing_entities(session, doc_id)

    merged: dict = {lbl: dict(ents) for lbl, ents in old_snapshot.items()}
    merged.setdefault(label, {})[entity_id] = item["props"]
    changes = {
        "added": [{"label": label, "id": entity_id, "props": item["props"]}],
        "modified": [],
        "deleted": [],
    }
    stats = update_graph_surgically(driver, doc_id, changes,
                                    new_entities=merged)
    logger.info("review item applied doc=%s entity=%s", doc_id, entity_id,
                extra={"doc_id": doc_id, "entity_id": entity_id,
                       "update_ms": stats["update_time_ms"]})
    return stats


# ---------------------------------------------------------------------------
# process-wide store (lazy — no db file created unless review is used)
# ---------------------------------------------------------------------------

_store: ReviewStore | None = None
_store_lock = threading.Lock()


def get_review_store() -> ReviewStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = ReviewStore(ROOT / settings.REVIEW_DB_PATH)
        return _store
