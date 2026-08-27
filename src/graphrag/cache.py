"""In-process answer cache (v2 — WS-A, G9).

TTL + LRU-bounded cache of ``run_query`` results. Cache keys bind every input
the answer depends on:

* query + pipeline parameters (hops, budget, reranker, answer mode)
* tenant scope (answers are tenant-specific)
* PII masking scope (an analyst's masked answer must never be served to an
  auditor — the key carries the effective PII scope)
* **dataset name + revision** — every write path bumps
  ``(:Dataset).rev`` (ingests, session seeds, CDC uploads), so any graph
  change invalidates stale entries across processes.

``graph_revision(driver)`` reads that marker; a cache entry is only ever
created when the revision is readable (unreadable -> caching is skipped, the
query still runs).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict

from graphrag.config import settings

CACHE_SCHEMA_VERSION = "v1"


def build_cache_key(**parts: object) -> str:
    """Deterministic hash key from the answer's input signature."""
    body = json.dumps({"schema": CACHE_SCHEMA_VERSION,
                       **{k: v for k, v in sorted(parts.items())}},
                      sort_keys=True, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def graph_revision(driver) -> tuple[str, int] | None:
    """(dataset name, revision) of the loaded graph; None when unreadable.

    ``rev`` is bumped by every write path (ingest/seed/CDC), so the cache key
    changes whenever the graph does. Reads are cheap (marker node lookup).
    """
    try:
        with driver.session() as session:
            row = session.run(
                "MATCH (d:Dataset) RETURN d.name AS name, coalesce(d.rev, 0) AS rev "
                "ORDER BY d.name LIMIT 1"
            ).single()
        if not row:
            return None
        return (row["name"], int(row["rev"]))
    except Exception:  # noqa: BLE001 - caching must never break the query path
        return None


def bump_revision(runner) -> None:
    """Increment the Dataset marker's rev (run inside a write transaction).

    Any graph mutation calls this so cached answers cannot survive a write.
    """
    runner.run("MATCH (d:Dataset) SET d.rev = coalesce(d.rev, 0) + 1")


class QueryCache:
    """Thread-safe TTL + LRU cache. Entries are ``{"ts": monotonic, ...}``."""

    def __init__(self, max_entries: int | None = None, ttl_s: float | None = None):
        self.max_entries = max_entries if max_entries is not None else settings.CACHE_MAX_ENTRIES
        self.ttl_s = ttl_s if ttl_s is not None else settings.CACHE_TTL_S
        self._entries: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def _evict_expired(self, now: float) -> None:
        while self._entries:
            _, entry = next(iter(self._entries.items()))
            if entry["ts"] + self.ttl_s > now:
                break
            self._entries.popitem(last=False)

    def get(self, key: str) -> dict | None:
        with self._lock:
            self._evict_expired(time.monotonic())
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry

    def put(self, key: str, entry: dict) -> None:
        with self._lock:
            now = time.monotonic()
            self._evict_expired(now)
            if key in self._entries:
                del self._entries[key]
            self._entries[key] = dict(entry)
            self._entries[key]["ts"] = now
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            self._evict_expired(time.monotonic())
            return len(self._entries)


query_cache = QueryCache()
