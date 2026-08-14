"""Unit tests for the fraud ground-truth comparison module (no Neo4j needed)."""

from graphrag.fraud_ground_truth import (
    build_comparison,
    claim_ids_in,
    evaluate_verdict,
    load_ground_truth,
    parse_fraud_verdict,
    verdict_for_claim,
)


def test_parse_fraud_verdict_affirmative():
    assert parse_fraud_verdict("Yes, claim CLM-00029 is flagged as fraud with a "
                               "FraudFlag entity having a severity of CONFIRMED") == "YES"
    assert parse_fraud_verdict("Yes — 2 fraud flag(s) found for claims CLM-0001, "
                               "CLM-0003: FRD-CLM-0003 (severity=MEDIUM)") == "YES"


def test_parse_fraud_verdict_negative():
    assert parse_fraud_verdict("Claim CLM-00117 is not flagged as fraud") == "NO"
    assert parse_fraud_verdict("No fraud flag was detected for this claim") == "NO"


def test_parse_fraud_verdict_refusal_is_unknown():
    # contains the word "fraudulent" but refuses -> must NOT be read as YES
    assert parse_fraud_verdict("Not determinable from the retrieved context. The "
                               "context does not contain any information that would "
                               "indicate whether claim CLM-00117 is fraudulent or not.") == "UNKNOWN"
    assert parse_fraud_verdict("The context does not provide any information that "
                               "would indicate whether claim CLM-00117 is "
                               "fraudulent or not.") == "UNKNOWN"
    assert parse_fraud_verdict("Most relevant context: CLM-0003 (Claim) — 9 nodes "
                               "retained.") == "UNKNOWN"


def test_verdict_for_claim_uses_window():
    answer = "The fraud claims under policy POL-5215 are CLM-00042 and CLM-00117. " \
             "Claim CLM-00099 is not flagged as fraud."
    assert verdict_for_claim(answer, "CLM-00042") == "YES"
    assert verdict_for_claim(answer, "CLM-00099") == "NO"
    assert verdict_for_claim(answer, "CLM-00001") == "UNKNOWN"  # never mentioned
    # refusal sentence mentioning the claim must not be read as an accusation
    answer2 = "Not determinable from the retrieved context. The context does not " \
              "provide any information that would indicate whether claim CLM-00117 " \
              "is fraudulent or not."
    assert verdict_for_claim(answer2, "CLM-00117") == "UNKNOWN"


def test_claim_ids_in():
    assert claim_ids_in("Does claim CLM-0003 or CLM-0005 have a flag?") == ["CLM-0003", "CLM-0005"]
    assert claim_ids_in("no ids here") == []


def test_evaluate_verdict():
    assert evaluate_verdict(True, "YES") == "correct"
    assert evaluate_verdict(False, "NO") == "correct"
    assert evaluate_verdict(False, "YES") == "wrong"
    assert evaluate_verdict(True, "NO") == "wrong"
    assert evaluate_verdict(True, "UNKNOWN") == "no-verdict"


def test_ground_truth_synthetic_samples():
    gt = load_ground_truth(None)  # demo graph
    assert gt and len(gt) >= 200
    assert gt["CLM-0003"] is True      # FRD-CLM-0003 in the demo graph
    assert gt["CLM-0001"] is False
    assert gt["CLM-0005"] is False


def test_ground_truth_insurance_claims_csv():
    gt = load_ground_truth("insurance_claims")
    assert gt is not None and len(gt) == 1000
    assert gt["CLM-00001"] is not None
    # labels are the CSV's fraud_reported column, not a constant
    assert any(gt.values()) and any(not v for v in gt.values())


def test_ground_truth_fraud_oracle_csv():
    gt = load_ground_truth("fraud_oracle")
    assert gt is not None and len(gt) == 15420
    assert sum(1 for v in gt.values() if v) == 923  # matches the documented fraud count


def test_datasets_without_labels_return_none():
    assert load_ground_truth("insurance_dataset") is None
    assert load_ground_truth("data_synthetic") is None


def test_build_comparison():
    pruned = {"kept": ["CLM-0003", "CLM-0001"], "dropped": []}
    gt = {"CLM-0003": True, "CLM-0001": False}
    rows = build_comparison("Is claim CLM-0003 fraudulent?", pruned,
                            "Yes, claim CLM-0003 is flagged as fraud.", gt)
    by_claim = {r["claim"]: r for r in rows}
    assert by_claim["CLM-0003"]["check"] == "✅ Correct"
    # CLM-0001 was never addressed by the answer -> no verdict, still listed
    assert by_claim["CLM-0001"]["check"] == "— No verdict"
    assert by_claim["CLM-0001"]["ground_truth"] == "Not fraud"
    # claims with no ground truth are excluded
    assert all(r["claim"] in gt for r in rows)


def test_build_comparison_none_ground_truth_returns_empty():
    # a label-less dataset (insurance_dataset/data_synthetic) must not crash
    assert build_comparison("Is claim CLM-0003 fraudulent?",
                            {"kept": ["CLM-0003"], "dropped": []},
                            "yes", None) == []


def test_build_comparison_includes_context_only_claims():
    # claims that only appear in the pruned context (list queries) get rows
    pruned = {"kept": ["CLM-00042", "CLM-00117"], "dropped": ["CLM-00099"]}
    gt = {"CLM-00042": True, "CLM-00117": True, "CLM-00099": False}
    rows = build_comparison("Show me fraud claims under policy POL-5215", pruned,
                            "The fraud claims are CLM-00042 and CLM-00117.", gt)
    by_claim = {r["claim"]: r for r in rows}
    assert set(by_claim) == {"CLM-00042", "CLM-00117", "CLM-00099"}
    assert by_claim["CLM-00042"]["check"] == "✅ Correct"
    assert by_claim["CLM-00099"]["check"] == "— No verdict"
