"""Performance monitoring (Phase 5).

A tiny in-memory metrics store: every query/upload appends one record; the
store rolls up latency + token-savings stats for the ``/api/v1/metrics``
endpoint. Thread-safe (FastAPI handlers run on a thread pool).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from statistics import mean
from typing import Any


class PerformanceMonitor:
    """Rolling in-memory metrics with aggregate rollups."""

    def __init__(self, max_records: int = 500):
        self.max_records = max_records
        self._records: deque[dict] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def record(self, **fields: Any) -> None:
        """Append one metric record with a timestamp."""
        with self._lock:
            self._records.appendleft({"ts": time.time(), **fields})

    def recent(self, limit: int = 100) -> list[dict]:
        with self._lock:
            return list(self._records)[:limit]

    def summary(self) -> dict:
        """Aggregate: request counts, avg latency, avg token savings."""
        with self._lock:
            rows = list(self._records)
        queries = [r for r in rows if r.get("kind") == "query"]
        uploads = [r for r in rows if r.get("kind") == "upload"]
        lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
        savings = [r["savings_pct"] for r in queries if r.get("savings_pct") is not None]
        return {
            "total_requests": len(rows),
            "queries": len(queries),
            "uploads": len(uploads),
            "avg_query_latency_ms": round(mean(lat), 2) if lat else None,
            "avg_token_savings_pct": round(mean(savings), 2) if savings else None,
        }


monitor = PerformanceMonitor()
