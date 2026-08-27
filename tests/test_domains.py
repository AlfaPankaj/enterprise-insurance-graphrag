"""v2 domain registry tests (WS-E, G18) — merged specs must be supersets."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from graphrag.domains import (BANKING, DOMAINS, INSURANCE, get_domain,
                              merged_entity_id_re, merged_id_patterns,
                              merged_keyword_props, merged_label_hints,
                              merged_node_kinds, merged_numeric_props,
                              merged_pii_classes, merged_required_fields,
                              merged_stopwords, merged_text_props)


def test_registry_has_both_domains():
    assert get_domain("insurance") is INSURANCE
    assert get_domain("banking") is BANKING
    assert DOMAINS[0].name == "insurance"     # insurance first in every merge
    assert get_domain("nope") is None


def test_merged_entity_id_re_matches_both_domains():
    regex = merged_entity_id_re()
    assert regex.findall("Does claim CLM-0003 have a fraud flag?") == ["CLM-0003"]
    assert regex.findall("status of account ACC-0012") == ["ACC-0012"]
    assert regex.findall("AML alert AML-0003 and TXN-000001") == \
        ["AML-0003", "TXN-000001"]
    assert regex.findall("nothing here") == []


def test_merged_keyword_and_numeric_props_are_supersets():
    props = merged_keyword_props()
    for p in ("claim_number", "cause", "account_number", "merchant",
              "alert_id"):
        assert p in props
    numeric = merged_numeric_props()
    assert ("amount", "Claim") in numeric
    assert ("balance", "Account") in numeric
    assert ("amount", "Transaction") in numeric


def test_merged_text_props_and_kinds():
    text = merged_text_props()
    assert text["Policy"] == ["policy_number", "type", "status", "premium",
                              "deductible", "start_date", "end_date"]
    assert "balance" in text["Account"]
    assert "merchant" in text["Transaction"]
    kinds = merged_node_kinds()
    assert kinds["Claim"] == "insurance claim"
    assert kinds["Account"] == "bank account"


def test_merged_label_hints_keep_insurance_order():
    hints = merged_label_hints()
    assert hints[0] == (("coverage", "coverages", "covers"), "Coverage")
    assert (("account", "accounts"), "Account") in hints
    assert (("aml", "laundering", "anti-money", "structuring"), "AMLAlert") in hints


def test_merged_stopwords_add_banking_nouns():
    sw = merged_stopwords()
    assert "account" in sw and "dispute" in sw and "merchant" in sw
    assert "laundering" not in sw          # value tokens must stay searchable


def test_merged_required_fields_and_id_patterns():
    required = merged_required_fields()
    assert required["Claim"] == ["id", "claim_number", "date", "amount",
                                 "status", "cause"]
    assert "balance" in required["Account"]
    patterns = merged_id_patterns()
    assert patterns["Claim"].fullmatch("CLM-0003")
    assert patterns["Account"].fullmatch("ACC-0012")
    assert not patterns["Account"].fullmatch("CLM-0003")


def test_merged_pii_classes_include_banking_customer():
    pii = merged_pii_classes()
    assert pii[("Policyholder", "name")] == "PII_IDENTITY"
    assert pii[("Customer", "name")] == "PII_IDENTITY"
    assert pii[("Customer", "phone")] == "PII_CONTACT"


def test_banking_spec_is_complete():
    assert set(BANKING.required_fields) == \
        {"Customer", "Account", "Transaction", "Dispute", "AMLAlert"}
    # every label has text props (serialization surface) and a kind
    for label in BANKING.required_fields:
        assert label in BANKING.text_props
        assert label in BANKING.node_kinds
        assert label in BANKING.id_patterns
    rels = {r[2] for r in BANKING.relationships}
    assert {"HOLDS", "POSTED", "HAS_DISPUTE", "ABOUT", "HAS_ALERT"} <= rels
