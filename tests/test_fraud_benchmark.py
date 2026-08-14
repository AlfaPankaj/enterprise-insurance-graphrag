"""Unit tests for the fraud-detection benchmark's pure metric helpers."""

import random

from scripts.benchmark_fraud_detection import (
    compute_confusion,
    metrics,
    predict_fraud,
    sample_clean_claims,
)


def test_predict_fraud_rule():
    assert predict_fraud("YES") is True
    assert predict_fraud("NO") is False
    assert predict_fraud("UNKNOWN") is False  # refusal/absence != fraud claim


def test_compute_confusion_perfect():
    # 3 fraud claims all flagged, 2 clean claims all unflagged
    pairs = [(True, "YES"), (True, "YES"), (True, "YES"),
             (False, "UNKNOWN"), (False, "NO")]
    assert compute_confusion(pairs) == (3, 0, 2, 0)


def test_compute_confusion_mixed():
    pairs = [
        (True, "YES"),   # TP
        (True, "UNKNOWN"),  # FN — fraud claim the answer refused to flag
        (False, "YES"),  # FP — clean claim falsely flagged
        (False, "NO"),   # TN
        (False, "UNKNOWN"),  # TN — no verdict = not fraud
    ]
    assert compute_confusion(pairs) == (1, 1, 2, 1)


def test_metrics_basic():
    m = metrics(tp=80, fp=5, tn=900, fn=20)
    assert m["precision"] == round(80 / 85, 4)          # 0.9412
    assert m["recall"] == round(80 / 100, 4)            # 0.8
    assert m["accuracy"] == round(980 / 1005, 4)
    expected_f1 = 2 * 0.9412 * 0.8 / (0.9412 + 0.8)
    assert abs(m["f1"] - round(expected_f1, 4)) < 1e-3


def test_sample_clean_claims_zero_is_none():
    # --negatives 0 must mean NO clean claims — regresses the bug where 0
    # silently kept all 14.5k negatives (an hour-long run)
    clean = [f"CLM-{i:05d}" for i in range(100)]
    assert sample_clean_claims(clean, 0, random.Random(1)) == []
    sampled = sample_clean_claims(clean, 10, random.Random(1))
    assert len(sampled) == 10 and set(sampled) <= set(clean)
    # more requested than available -> all of them
    assert len(sample_clean_claims(clean, 500, random.Random(1))) == 100


def test_metrics_zero_denominators():
    m = metrics(tp=0, fp=0, tn=100, fn=0)
    assert m == {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                 "accuracy": 1.0}
    m2 = metrics(tp=0, fp=0, tn=0, fn=0)
    assert m2["precision"] == 0.0 and m2["recall"] == 0.0 and m2["f1"] == 0.0
