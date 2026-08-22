"""v2 extraction review queue + confidence scoring tests (WS-C, G17)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest
from fastapi.testclient import TestClient

from graphrag.config import settings
from graphrag.entity_extractor import (extract_entities_with_confidence,
                                       score_entity, score_extraction)
from graphrag.extraction_review import (ReviewStore, apply_review_item,
                                        partition_entities)


@pytest.fixture(autouse=True)
def _reset_review_settings():
    yield
    settings.EXTRACTION_REVIEW_ENABLED = False
    settings.EXTRACTION_CONFIDENCE_THRESHOLD = 0.7
    settings.AUTH_MODE = "none"
    settings.API_KEY = ""

# ---------------------------------------------------------------------------
# confidence scoring (deterministic)
# ---------------------------------------------------------------------------

def test_score_entity_complete_is_1():
    score, reasons = score_entity(
        "Claim", "CLM-0003",
        {"id": "CLM-0003", "claim_number": "CLM-2024-0003", "date": "2024-01-01",
         "amount": 5000.0, "status": "IN_REVIEW", "cause": "water damage"})
    assert score == 1.0 and reasons == []


def test_score_entity_missing_fields_lower():
    score, reasons = score_entity("Claim", "CLM-0003", {"id": "CLM-0003"})
    assert score < 0.7
    assert "missing_required_fields" in reasons


def test_score_entity_malformed_id_lower():
    score, reasons = score_entity("Claim", "banana", {"id": "banana"})
    assert "malformed_id" in reasons
    assert score < 0.7


def test_score_entity_generic_real_world_id():
    # real-world id schemes (CL-2024-010101) are acceptable via the fallback
    score, reasons = score_entity(
        "Claim", "CL-2024-010101",
        {"id": "CL-2024-010101", "claim_number": "x", "date": "2024-01-01",
         "amount": 1.0, "status": "OPEN", "cause": "fire"})
    assert "malformed_id" not in reasons
    assert score == 1.0


def test_score_extraction_shape():
    entities = {"Claim": {"CLM-0003": {"id": "CLM-0003", "amount": 1.0}},
                "Policy": {"POL-0001": {"id": "POL-0001", "status": "ACTIVE",
                                        "policy_number": "P", "type": "T",
                                        "start_date": "s", "end_date": "e",
                                        "premium": 1.0, "deductible": 1.0}}}
    confidence, reasons = score_extraction(entities)
    assert confidence["Claim"]["CLM-0003"] < 1.0
    assert confidence["Policy"]["POL-0001"] == 1.0
    assert "missing_required_fields" in reasons["Claim"]["CLM-0003"]


def test_extract_entities_with_confidence_additive():
    """The confidence variant must return the SAME core contract plus extras."""
    from graphrag.entity_extractor import extract_entities

    text = ("Commercial Insurance Policy\n"
            "Policy ID POL-0009\nPolicy Number CP-2024-0009\n"
            "Type COMMERCIAL_GENERAL_LIABILITY\n"
            "Term 2024-01-01 to 2024-12-31\nStatus ACTIVE\n"
            "Annual Premium $12,000\nDeductible $1,000\n"
            "Policyholder ID PH-0009\n")
    base = extract_entities(text, doc_id_hint="policy_POL-0009.pdf")
    conf = extract_entities_with_confidence(text, doc_id_hint="policy_POL-0009.pdf")
    assert conf["doc_id"] == base["doc_id"]
    assert conf["entities"] == base["entities"]
    assert conf["mode"] == base["mode"]
    assert "confidence" in conf and "reasons" in conf
    assert conf["confidence"]["Policy"]["POL-0009"] == 1.0


def test_key_aliases_parse_real_world_spellings():
    """'Policy No:', 'Loss Date:', 'Annual Premium' spellings must extract."""
    from graphrag.entity_extractor import extract_entities_heuristic

    text = ("Commercial Insurance Policy\n"
            "Policy No: CP-2024-0077\n"
            "Type: COMMERCIAL_GENERAL_LIABILITY\n"
            "Term: 2024-02-01 to 2025-01-31\n"
            "Status: ACTIVE\n"
            "Annual Premium: $9,000\n"
            "Deductible: $500\n"
            "Policyholder ID: PH-0077\n")
    result = extract_entities_heuristic(text, doc_id_hint="policy_POL-0077.pdf")
    policies = result["entities"].get("Policy", {})
    assert "POL-0077" in policies, "no policy extracted from real-world key spellings"
    pol = policies["POL-0077"]
    assert pol["policy_number"] == "CP-2024-0077"
    assert pol["premium"] == 9000.0
    assert pol["deductible"] == 500.0
    assert pol["start_date"] == "2024-02-01"   # "Term:" colon form parsed


# ---------------------------------------------------------------------------
# partition + review store
# ---------------------------------------------------------------------------

def test_partition_entities_splits_by_threshold():
    entities = {
        "Claim": {
            "CLM-0001": {"id": "CLM-0001", "claim_number": "c", "date": "d",
                         "amount": 1.0, "status": "OPEN", "cause": "fire"},
            "CLM-0002": {"id": "CLM-0002"},   # incomplete -> held
        },
    }
    confidence = {"Claim": {"CLM-0001": 1.0, "CLM-0002": 0.4}}
    to_apply, held = partition_entities(entities, confidence, 0.7)
    assert list(to_apply["Claim"]) == ["CLM-0001"]
    assert [h["entity_id"] for h in held] == ["CLM-0002"]
    assert held[0]["confidence"] == 0.4


def test_review_store_lifecycle(tmp_path):
    store = ReviewStore(tmp_path / "review.db")
    rid, created = store.submit("DOC-1", "a.pdf", "Claim", "CLM-0002",
                                {"id": "CLM-0002"}, 0.4, ["missing_required_fields"])
    assert created is True
    item = store.get(rid)
    assert item["status"] == "pending" and item["confidence"] == 0.4
    assert item["reasons"] == ["missing_required_fields"]
    assert store.summary() == {"pending": 1, "approved": 0, "rejected": 0}

    # re-submitting the same (doc, entity) updates instead of duplicating
    rid2, created2 = store.submit("DOC-1", "a.pdf", "Claim", "CLM-0002",
                                  {"id": "CLM-0002", "amount": 9.0}, 0.5)
    assert rid2 == rid and created2 is False
    assert store.get(rid)["confidence"] == 0.5
    assert store.summary()["pending"] == 1

    # approve -> decided, metrics-visible state change
    decided = store.decide(rid, "approved", "user-1")
    assert decided["status"] == "approved" and decided["decided_by"] == "user-1"
    assert store.summary() == {"pending": 0, "approved": 1, "rejected": 0}
    # already decided -> no re-decision
    assert store.decide(rid, "rejected") is None

    # list filtering
    assert [i["id"] for i in store.list(status="approved")] == [rid]
    assert store.list(status="pending") == []


def test_review_store_reject_and_invalid_decision(tmp_path):
    store = ReviewStore(tmp_path / "review.db")
    rid, _ = store.submit("D", None, "Policy", "POL-9", {"id": "POL-9"}, 0.3)
    assert store.decide(rid, "rejected")["status"] == "rejected"
    with pytest.raises(ValueError, match="approved or rejected"):
        store.decide("whatever", "maybe")
    assert store.get("missing") is None
    assert store.decide("missing", "approved") is None


# ---------------------------------------------------------------------------
# apply_review_item (approve path → CDC + snapshot merge) — scripted driver
# ---------------------------------------------------------------------------

class _Cursor:
    def __init__(self, rows=()):
        self._rows = list(rows)

    def single(self):
        return self._rows[0] if self._rows else None

    def data(self):
        return self._rows

    def __iter__(self):
        return iter(self._rows)


class _Tx:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def run(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return _Cursor()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Session:
    def __init__(self, tx):
        self._tx = tx
        self.calls = tx.calls

    def run(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return _Cursor()

    def begin_transaction(self):
        return self._tx

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Driver:
    def __init__(self):
        self.tx = _Tx()

    def session(self, **kw):
        return _Session(self.tx)

    def close(self):
        pass  # lifespan shutdown calls driver.close()


def test_apply_review_item_runs_cdc_add_and_snapshot_merge():
    driver = _Driver()
    item = {"doc_id": "DOC-9", "source_file": "x.pdf", "label": "Claim",
            "entity_id": "CLM-0009",
            "props": {"id": "CLM-0009", "amount": 100.0, "status": "OPEN"},
            "confidence": 0.9}
    stats = apply_review_item(driver, item)
    assert stats["entities_added"] == 1
    # the CDC add ran (MERGE with the entity) AND the snapshot save ran with
    # the merged snapshot (entity included)
    queries = " ".join(q for q, _ in driver.tx.calls)
    assert "MERGE (n:Claim" in queries
    assert "DocSnapshot" in queries
    snapshot_call = next(kw for q, kw in driver.tx.calls
                         if "DocSnapshot" in q and "entities_json" in kw)
    snapshot = json.loads(snapshot_call["entities_json"])
    assert snapshot["Claim"]["CLM-0009"]["amount"] == 100.0


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@pytest.fixture()
def review_store(tmp_path, monkeypatch):
    import graphrag.api_server as api

    s = ReviewStore(tmp_path / "api-review.db")
    monkeypatch.setattr(api, "get_review_store", lambda: s)
    return s


def _token(roles, secret="x" * 40):
    import jwt as pyjwt
    return pyjwt.encode({"sub": "u", "roles": roles}, secret, algorithm="HS256")


def test_review_endpoints_flow(monkeypatch, review_store):
    import graphrag.api_server as api

    monkeypatch.setattr(settings, "AUTH_MODE", "none")
    monkeypatch.setattr(settings, "API_KEY", "")
    rid, _ = review_store.submit("DOC-1", "a.pdf", "Claim", "CLM-0002",
                                 {"id": "CLM-0002"}, 0.4,
                                 ["missing_required_fields"])

    # approve applies through apply_review_item (patched to a recorder)
    applied = {"called": False}

    def fake_apply(driver, item):
        applied["called"] = True
        applied["entity"] = item["entity_id"]
        return {"entities_added": 1, "update_time_ms": 5.0}

    monkeypatch.setattr(api, "apply_review_item", fake_apply)
    with TestClient(api.app) as client:
        listing = client.get("/api/v1/review").json()
        assert listing["summary"]["pending"] == 1
        assert listing["items"][0]["id"] == rid

        resp = client.post(f"/api/v1/review/{rid}/approve")
        assert resp.status_code == 200
        assert applied["called"] and applied["entity"] == "CLM-0002"
        assert resp.json()["item"]["status"] == "approved"
        assert client.get("/api/v1/review").json()["summary"] == \
            {"pending": 0, "approved": 1, "rejected": 0}

        # re-approve an already-decided item -> 409
        assert client.post(f"/api/v1/review/{rid}/approve").status_code == 409
        # unknown item -> 404
        assert client.post("/api/v1/review/missing/approve").status_code == 404


def test_review_reject_and_role_gates(monkeypatch, review_store):
    import graphrag.api_server as api

    rid, _ = review_store.submit("D", None, "Policy", "POL-9", {"id": "POL-9"},
                                 0.3)

    monkeypatch.setattr(settings, "AUTH_MODE", "jwt")
    monkeypatch.setattr(settings, "JWT_SECRET", "x" * 40)
    monkeypatch.setattr(settings, "JWT_ISSUER", "")
    monkeypatch.setattr(settings, "JWT_AUDIENCE", "")
    with TestClient(api.app) as client:
        auditor = {"Authorization": f"Bearer {_token(['auditor'])}"}
        analyst = {"Authorization": f"Bearer {_token(['analyst'])}"}
        # viewing is open to all roles
        assert client.get("/api/v1/review", headers=auditor).status_code == 200
        # decisions are analyst/admin only — auditor gets 403
        assert client.post(f"/api/v1/review/{rid}/reject",
                           headers=auditor).status_code == 403
        resp = client.post(f"/api/v1/review/{rid}/reject", headers=analyst)
        assert resp.status_code == 200
        assert resp.json()["item"]["status"] == "rejected"
        assert review_store.summary()["rejected"] == 1


# ---------------------------------------------------------------------------
# upload path: mixed-confidence extraction -> partial apply + hold
# ---------------------------------------------------------------------------

def test_upload_holds_low_confidence_when_review_enabled(monkeypatch,
                                                         review_store):
    import graphrag.api_server as api

    monkeypatch.setattr(settings, "AUTH_MODE", "none")
    monkeypatch.setattr(settings, "API_KEY", "")
    monkeypatch.setattr(settings, "EXTRACTION_REVIEW_ENABLED", True)
    monkeypatch.setattr(settings, "EXTRACTION_CONFIDENCE_THRESHOLD", 0.7)

    monkeypatch.setattr(api, "extract_text_from_pdf", lambda b: "pdf text")
    monkeypatch.setattr(
        api, "extract_entities_with_confidence",
        lambda text, doc_id_hint=None, mode=None: {
            "doc_id": "POL-0077",
            "mode": "heuristic",
            "entities": {
                "Policy": {"POL-0077": {
                    "id": "POL-0077", "policy_number": "CP-2024-0077",
                    "type": "T", "start_date": "s", "end_date": "e",
                    "premium": 1.0, "deductible": 1.0, "status": "ACTIVE"}},
                "Policyholder": {"PH-0077": {"id": "PH-0077"}},
            },
            "confidence": {"Policy": {"POL-0077": 1.0},
                           "Policyholder": {"PH-0077": 0.2}},
            "reasons": {"Policyholder": {"PH-0077":
                                         ["missing_required_fields"]}},
        })

    driver = _Driver()
    with TestClient(api.app) as client:
        api.app.state.driver = driver  # lifespan driver is replaced for the test
        resp = client.post("/api/v1/upload",
                           files={"file": ("policy.pdf", b"%PDF-1.4 fake",
                                           "application/pdf")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["review"]["held"] == 1            # PH-0077 held
    assert body["review"]["applied_entities"] == 1  # POL-0077 applied
    assert review_store.summary()["pending"] == 1
    held = review_store.list(status="pending")[0]
    assert held["entity_id"] == "PH-0077"
    assert held["reasons"] == ["missing_required_fields"]
    # the applied policy DID reach the CDC path (MERGE ran)
    assert any("MERGE (n:Policy" in q for q, _ in driver.tx.calls)
