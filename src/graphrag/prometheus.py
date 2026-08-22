"""Zero-dependency Prometheus exposition (v2 — WS-D observability).

A tiny counter/histogram registry that renders the exact text format
Prometheus scrapes (``GET /metrics``, admin/auditor roles). No external
client library — same zero-dependency philosophy as the BM25 backend.

Metrics exposed:

* ``graphrag_requests_total{kind}``            query | upload | stream
* ``graphrag_errors_total{kind}``
* ``graphrag_rate_limited_total``
* ``graphrag_query_latency_seconds``           histogram
* ``graphrag_upload_latency_seconds``          histogram
* ``graphrag_token_savings_ratio``             histogram (0..1)
* ``graphrag_llm_cost_usd_total``
* ``graphrag_llm_fallbacks_total``
* ``graphrag_cache_hits_total`` / ``graphrag_cache_misses_total``
* ``graphrag_audit_records_total``
"""

from __future__ import annotations

import json
import threading

_LOCK = threading.Lock()

_DEFS: dict[str, tuple[str, str]] = {}          # name -> (type, help)
_COUNTERS: dict[str, dict[tuple, float]] = {}   # name -> {labeltuple: value}
_HISTOGRAMS: dict[str, dict] = {}               # name -> {"buckets": [...], "data": {labeltuple: [.., +inf_count, sum]}}

_LATENCY_BUCKETS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
_SAVINGS_BUCKETS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


def _labels_tuple(labels: dict) -> tuple:
    return tuple(sorted(labels.items()))


class Counter:
    """A monotonically-increasing counter, optionally labelled."""

    def __init__(self, name: str):
        self.name = name

    def inc(self, value: float = 1.0, **labels) -> None:
        key = _labels_tuple(labels)
        with _LOCK:
            bucket = _COUNTERS.setdefault(self.name, {})
            bucket[key] = bucket.get(key, 0.0) + value


class Gauge:
    """A settable metric (increment/decrement/set), optionally labelled."""

    def __init__(self, name: str):
        self.name = name

    def _value(self, key: tuple) -> float:
        return _COUNTERS.setdefault(self.name, {}).get(key, 0.0)

    def inc(self, value: float = 1.0, **labels) -> None:
        self.set(self._value(_labels_tuple(labels)) + value, **labels)

    def dec(self, value: float = 1.0, **labels) -> None:
        self.inc(-value, **labels)

    def set(self, value: float, **labels) -> None:
        key = _labels_tuple(labels)
        with _LOCK:
            _COUNTERS.setdefault(self.name, {})[key] = value


class Histogram:
    """A cumulative histogram over preconfigured buckets."""

    def __init__(self, name: str, buckets: list[float]):
        self.name = name
        self.buckets = buckets

    def observe(self, value: float, **labels) -> None:
        key = _labels_tuple(labels)
        with _LOCK:
            entry = _HISTOGRAMS.setdefault(self.name, {"buckets": self.buckets, "data": {}})
            row = entry["data"].get(key)
            if row is None:
                row = [0] * len(self.buckets) + [0, 0.0]  # bucket counts + inf + sum
                entry["data"][key] = row
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    row[i] += 1
            row[-2] += 1      # +Inf count
            row[-1] += value  # sum


def counter(name: str, help_text: str) -> Counter:
    _DEFS[name] = ("counter", help_text)
    return Counter(name)


def gauge(name: str, help_text: str) -> Gauge:
    _DEFS[name] = ("gauge", help_text)
    return Gauge(name)


def histogram(name: str, help_text: str, buckets: list[float]) -> Histogram:
    _DEFS[name] = ("histogram", help_text)
    return Histogram(name, buckets)


# ---------------------------------------------------------------------------
# the production metric set
# ---------------------------------------------------------------------------

requests_total = counter("graphrag_requests_total", "API requests by kind (query/upload/stream).")
errors_total = counter("graphrag_errors_total", "Request failures by kind.")
rate_limited_total = counter("graphrag_rate_limited_total", "Requests rejected by the rate limiter.")
query_latency = histogram("graphrag_query_latency_seconds",
                          "End-to-end query latency (seconds).", _LATENCY_BUCKETS)
upload_latency = histogram("graphrag_upload_latency_seconds",
                           "PDF upload/CDC latency (seconds).", _LATENCY_BUCKETS)
token_savings = histogram("graphrag_token_savings_ratio",
                          "Token savings ratio after pruning (0..1).", _SAVINGS_BUCKETS)
llm_cost_total = counter("graphrag_llm_cost_usd_total", "Cumulative LLM answer cost (USD).")
llm_fallbacks_total = counter("graphrag_llm_fallbacks_total",
                              "Answers that fell back to the deterministic extractor.")
cache_hits_total = counter("graphrag_cache_hits_total", "Answer cache hits.")
cache_misses_total = counter("graphrag_cache_misses_total", "Answer cache misses.")
audit_records_total = counter("graphrag_audit_records_total", "Audit trail records written.")
jobs_running = gauge("graphrag_jobs_running", "Background jobs currently running.")
jobs_completed_total = counter("graphrag_jobs_completed_total",
                               "Jobs that reached a terminal state, by status.")


# ---------------------------------------------------------------------------
# exposition rendering
# ---------------------------------------------------------------------------

def _fmt_label(key: tuple) -> str:
    if not key:
        return ""
    return "{" + ",".join(f"{k}={json.dumps(v)}" for k, v in key) + "}"


def render() -> str:
    """Prometheus text-format exposition of all observed metrics."""
    lines: list[str] = []
    with _LOCK:
        for name in sorted(_DEFS):
            kind, help_text = _DEFS[name]
            if kind in ("counter", "gauge"):
                bucket = _COUNTERS.get(name)
                if not bucket:
                    continue
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} {kind}")
                for key in sorted(bucket):
                    lines.append(f"{name}{_fmt_label(key)} {bucket[key]:.6g}")
            else:
                entry = _HISTOGRAMS.get(name)
                if not entry or not entry["data"]:
                    continue
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} histogram")
                buckets = entry["buckets"]
                for key in sorted(entry["data"]):
                    row = entry["data"][key]
                    for i, bound in enumerate(buckets):
                        lines.append(f"{name}_bucket{_fmt_label(key + (('le', str(bound)),))} {row[i]}")
                    lines.append(f"{name}_bucket{_fmt_label(key + (('le', '+Inf'),))} {row[-2]}")
                    lines.append(f"{name}_sum{_fmt_label(key)} {row[-1]:.6g}")
                    lines.append(f"{name}_count{_fmt_label(key)} {row[-2]}")
    return "\n".join(lines) + ("\n" if lines else "")


def reset() -> None:
    """Drop all observations (tests)."""
    with _LOCK:
        _COUNTERS.clear()
        _HISTOGRAMS.clear()
