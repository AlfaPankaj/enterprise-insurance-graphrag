"""Unit tests for the edge-case benchmark helpers."""

from __future__ import annotations

from scripts.benchmark_edge_cases import (
    DEFAULT_HEAD,
    DEFAULT_MIDDLE,
    DEFAULT_TAIL,
    build_query,
    sample_positions,
)


def test_sample_positions_spread_and_sorted():
    pos = sample_positions(100, DEFAULT_HEAD, DEFAULT_MIDDLE, DEFAULT_TAIL)
    # 5 unique positions, sorted, within range
    assert pos == sorted(pos)
    assert len(pos) == 5
    assert all(1 <= p <= 100 for p in pos)
    # head before middle before tail
    assert pos[0] < pos[1] < pos[2] < pos[3] < pos[4]


def test_sample_positions_clamps_small_files():
    # 3-row file still yields head/middle/tail coverage without dupes
    pos = sample_positions(3, DEFAULT_HEAD, DEFAULT_MIDDLE, DEFAULT_TAIL)
    assert pos == [1, 2, 3]


def test_sample_positions_dedupes():
    pos = sample_positions(10, [0.5], [0.5], [0.5])
    assert pos == [5]


def test_build_query_fraud_datasets():
    q = build_query("fraud_oracle", {"PolicyNumber": "A123", "FraudFound_P": "1"}, 771)
    assert q["query"] == "Is claim CLM-00771 flagged as fraud?"
    assert q["expected"] == ["CLM-00771"]
    assert q["fraud_label"] is True

    q = build_query("insurance_claims",
                    {"fraud_reported": "N", "total_claim_amount": "5000"}, 950)
    assert q["query"] == "Was claim CLM-00950 reported as fraud?"
    assert q["fraud_label"] is False


def test_build_query_insurance_dataset():
    q = build_query("insurance_dataset", {"Claim_Amount": "25000"}, 7150)
    assert q["query"] == "What is the status of claim CLM-07150?"
    assert q["expected"] == ["CLM-07150"]


def test_build_query_data_synthetic_anchors_on_ph_id():
    q = build_query("data_synthetic", {"Customer ID": "59266"}, 50828)
    # the query must name the graph's node id (PH-…) so retrieval can seed
    assert q["query"] == "Show me the coverage for policyholder PH-59266"
    assert q["expected"] == ["POL-59266", "COV-59266"]
