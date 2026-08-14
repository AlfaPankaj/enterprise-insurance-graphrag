"""Unit tests for the benchmark-proof aggregator's pure math helpers."""

import json
import tempfile
from pathlib import Path

from scripts.export_benchmark_proof import aggregate_benchmarks, aggregate_fraud, f2i


def _write(bench_dir: Path, name: str, payload: dict) -> None:
    (bench_dir / name).write_text(json.dumps(payload), encoding="utf-8")


def _sample_real(n_queries: int = 4) -> dict:
    return {
        "dataset": "sample",
        "queries": n_queries,
        "retrieval_accuracy": 100.0,
        "pruning_accuracy": 100.0,
        "avg_savings_pct": 20.0,
        "avg_latency_ms": 100.0,
        "results": [
            {"tokens_before": 100, "tokens_after": 80} for _ in range(n_queries)
        ],
    }


def test_f2i_handles_none():
    assert f2i(None) == 0
    assert f2i(10.4) == 10
    assert f2i(10.6) == 11


def test_aggregate_benchmarks_sums_tokens():
    with tempfile.TemporaryDirectory() as tmp:
        bench = Path(tmp)
        _write(bench, "real_a.json", _sample_real(4))
        _write(bench, "real_b.json", _sample_real(6))
        agg = aggregate_benchmarks(bench)
        assert agg["queries"] == 10
        assert agg["raw_tokens"] == 1000
        assert agg["optimized_tokens"] == 800
        assert agg["retrieval_accuracy"] == 100.0


def test_aggregate_benchmarks_edge_cases_counts():
    with tempfile.TemporaryDirectory() as tmp:
        bench = Path(tmp)
        _write(bench, "edge_cases.json", {
            "total_queries": 5,
            "retrieval_accuracy": 100.0,
            "avg_savings_pct": 10.0,
            "avg_latency_ms": 50.0,
            "results": [
                {"tokens_before": 200, "tokens_after": 150,
                 "survived_prune": True} for _ in range(5)
            ],
        })
        agg = aggregate_benchmarks(bench)
        assert agg["queries"] == 5
        assert agg["raw_tokens"] == 1000
        assert agg["optimized_tokens"] == 750


def test_aggregate_fraud_merges_confusion():
    with tempfile.TemporaryDirectory() as tmp:
        bench = Path(tmp)
        _write(bench, "fraud_detection_a.json",
               {"confusion": {"tp": 10, "fp": 0, "tn": 90, "fn": 0}})
        _write(bench, "fraud_detection_b.json",
               {"confusion": {"tp": 5, "fp": 1, "tn": 20, "fn": 1}})
        fraud = aggregate_fraud(bench)
        assert fraud["confusion"] == {"tp": 15, "fp": 1, "tn": 110, "fn": 1}
        assert fraud["fraud_evaluated"] == 16
        assert fraud["clean_evaluated"] == 111
        assert round(fraud["precision"], 4) == round(15 / 16, 4)
        assert round(fraud["recall"], 4) == round(15 / 16, 4)
