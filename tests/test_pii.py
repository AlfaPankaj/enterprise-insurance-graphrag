"""v2 PII classification & masking tests (WS-B, G2)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag.config import settings
from graphrag.pii import (MaskingPolicy, classify, mask_text, mask_value,
                          redact_node, scrub_answer)

HOLDER = {"id": "PH-0001", "label": "Policyholder",
          "props": {"name": "Alice Example", "dob": "1985-03-04",
                    "address": "12 Park Lane, London", "phone": "+1 555 010 2222",
                    "email": "alice@example.com", "risk_score": 42.0}}

CLAIM = {"id": "CLM-0003", "label": "Claim",
         "props": {"status": "PAID", "amount": 12000.0, "cause": "fire"}}

INVESTIGATOR = {"id": "INV-0001", "label": "Investigator",
                "props": {"name": "Bob Example", "email": "bob@example.com",
                          "role": "SENIOR"}}

ALLOW = MaskingPolicy(True, True, True)
DENY = MaskingPolicy(False, False, False)


def test_classification_table():
    assert classify("Policyholder", "name") == "PII_IDENTITY"
    assert classify("Policyholder", "email") == "PII_CONTACT"
    assert classify("Investigator", "name") == "PII_IDENTITY"
    assert classify("Claim", "amount") is None          # business field
    assert classify("Claim", "status") is None
    # generic name-pattern fallback for unlisted labels (custom CSVs)
    assert classify("Record", "email") == "PII_CONTACT"
    assert classify("Record", "customer_name") == "PII_IDENTITY"


def test_redact_node_off_mode_returns_same_node(monkeypatch):
    monkeypatch.setattr(settings, "PII_MODE", "off")
    policy = MaskingPolicy.for_roles({"analyst"})
    assert redact_node(HOLDER, policy) is HOLDER  # untouched, same object


def test_redact_node_masks_pii_for_denied_role(monkeypatch):
    monkeypatch.setattr(settings, "PII_MODE", "mask")
    out = redact_node(HOLDER, DENY)
    assert out is not HOLDER
    assert out["props"]["name"] == "[REDACTED]"
    assert out["props"]["dob"] == "[REDACTED]"
    assert "***" in out["props"]["email"]
    assert out["props"]["phone"] != HOLDER["props"]["phone"]
    # non-PII business fields untouched (retrieval semantics unchanged)
    assert out["props"]["risk_score"] == 42.0
    assert out["id"] == "PH-0001"


def test_redact_node_allows_pii_reader_role(monkeypatch):
    monkeypatch.setattr(settings, "PII_MODE", "mask")
    monkeypatch.setattr(settings, "PII_READER_ROLES", "admin,auditor")
    out = redact_node(HOLDER, MaskingPolicy.for_roles({"auditor"}))
    assert out["props"]["name"] == "Alice Example"     # auditor may read
    out2 = redact_node(HOLDER, MaskingPolicy.for_roles({"analyst"}))
    assert out2["props"]["name"] == "[REDACTED]"       # analyst may not


def test_redact_claim_untouched_even_for_denied_role(monkeypatch):
    monkeypatch.setattr(settings, "PII_MODE", "mask")
    out = redact_node(CLAIM, DENY)
    assert out["props"] == CLAIM["props"]


def test_mask_value_forms():
    assert mask_value("alice@example.com", "PII_CONTACT") == "al***@example.com"
    # contact masking redacts the numeric parts (street number, phone digits)
    assert mask_value("123 Main St", "PII_CONTACT") == "### Main St"
    assert mask_value("+1 555 010 2222", "PII_CONTACT") == "+# ### ### ####"
    assert mask_value("anything", "PII_IDENTITY") == "[REDACTED]"
    assert mask_value("cancer", "PII_HEALTH") == "[REDACTED-HEALTH]"


def test_mask_text_scrubs_free_text():
    text = "Call Alice at alice@example.com or +1 555 010 2222, born 1985-03-04."
    out = mask_text(text)
    assert "alice@example.com" not in out
    assert "+1 555 010 2222" not in out
    assert "1985-03-04" not in out
    assert "[EMAIL-REDACTED]" in out and "[PHONE-REDACTED]" in out


def test_scrub_answer_off_mode_noop(monkeypatch):
    monkeypatch.setattr(settings, "PII_MODE", "off")
    assert scrub_answer("alice@example.com", MaskingPolicy.for_roles({"analyst"})) \
        == "alice@example.com"


def test_scrub_answer_mask_mode_role_blind(monkeypatch):
    monkeypatch.setattr(settings, "PII_MODE", "mask")
    out = scrub_answer("Contact alice@example.com", MaskingPolicy(True, True, True))
    assert "alice@example.com" not in out
