"""v2 banking domain tests — adapters + end-to-end pipeline (scripted driver).

Proves the second domain plugs into the EXISTING pipeline unchanged:
retrieval, re-ranking, pruning, audit, evals all work on banking data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest

from graphrag.config import settings
from graphrag.query_pipeline import run_query

SAMPLE = ROOT / "data" / "samples" / "banking.json"

# ---------------------------------------------------------------------------
# adapter tests (pure, no DB)
# ---------------------------------------------------------------------------

def test_banking_dataset_well_formed():
    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
    counts = data["counts"]
    assert counts["customers"] == 60
    assert counts["transactions"] == 400
    assert counts["aml_alerts"] == 18
    # every join field points at a real record
    account_ids = {a["id"] for a in data["accounts"]}
    customer_ids = {c["id"] for c in data["customers"]}
    txn_ids = {t["id"] for t in data["transactions"]}
    for a in data["accounts"]:
        assert a["customer_id"] in customer_ids
    for t in data["transactions"]:
        assert t["account_id"] in account_ids
    for d in data["disputes"]:
        assert d["account_id"] in account_ids
        assert d["transaction_id"] in txn_ids
    for aml in data["aml_alerts"]:
        assert aml["account_id"] in account_ids
    # unique merchant anchor: exactly one transaction uses it
    merchant_txns = [t for t in data["transactions"]
                     if t["merchant"] == data["unique_merchant"]]
    assert len(merchant_txns) == 1
    assert merchant_txns[0]["account_id"] == data["unique_merchant_account"]


def test_adapt_banking_builds_graph_contract():
    from scripts.ingest_banking_dataset import adapt_banking

    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
    nodes, rels = adapt_banking(data)
    assert sum(len(v) for v in nodes.values()) == \
        data["counts"]["customers"] + data["counts"]["accounts"] \
        + data["counts"]["transactions"] + data["counts"]["disputes"] \
        + data["counts"]["aml_alerts"]
    # join fields are stripped from node props
    acc = next(n for n in nodes["Account"] if n["id"] == "ACC-0035")
    assert "customer_id" not in acc["props"]
    rel_types = {r[4] for r in rels}
    assert {"HOLDS", "POSTED", "HAS_DISPUTE", "ABOUT", "HAS_ALERT"} <= rel_types
    # counts: disputes 30 -> HAS_DISPUTE 30 + ABOUT 30
    assert sum(1 for r in rels if r[4] == "HAS_DISPUTE") == 30
    assert sum(1 for r in rels if r[4] == "POSTED") == 400


# ---------------------------------------------------------------------------
# end-to-end pipeline on a scripted banking graph
# ---------------------------------------------------------------------------

NODES = {
    "ACC-0035": ("Account", {"id": "ACC-0035", "account_number": "IB-9",
                             "type": "CHECKING", "status": "ACTIVE",
                             "balance": 42000.0, "currency": "USD"}),
    "CUST-0001": ("Customer", {"id": "CUST-0001", "name": "Aarav Sharma",
                               "risk_tier": "LOW"}),
    "TXN-000042": ("Transaction", {"id": "TXN-000042",
                                   "transaction_id": "TRX00000042",
                                   "type": "POS", "amount": 250.0,
                                   "merchant": "Aurora Rare Goods Co.",
                                   "status": "POSTED"}),
    "DSP-0001": ("Dispute", {"id": "DSP-0001", "dispute_id": "DSP-2026-00001",
                             "reason": "goods not received",
                             "status": "OPEN", "amount": 250.0}),
    "AML-0001": ("AMLAlert", {"id": "AML-0001", "alert_id": "AML-2026-00001",
                              "reason": "money laundering pattern",
                              "severity": "HIGH", "status": "OPEN",
                              "amount": 42000.0}),
}
EDGES = [
    ("CUST-0001", "HOLDS", "ACC-0035"),
    ("ACC-0035", "POSTED", "TXN-000042"),
    ("ACC-0035", "HAS_DISPUTE", "DSP-0001"),
    ("DSP-0001", "ABOUT", "TXN-000042"),
    ("ACC-0035", "HAS_ALERT", "AML-0001"),
]


class _Node:
    def __init__(self, props):
        self._props = props

    def __getitem__(self, k):
        return self._props[k]

    def items(self):
        return self._props.items()

    def keys(self):
        return self._props.keys()


class _Cursor:
    def __init__(self, query, kwargs):
        self.query, self.kwargs = query, kwargs
        self._rows = self._resolve()

    def _resolve(self):
        q = self.query
        # UNWIND scans first: their RETURN lines share substrings with the
        # node-fetch query ("labels(n) AS labels, n" ⊂ "…n.id AS id")
        # global numeric scan: "accounts over $X" style thresholds
        if "UNWIND $pairs" in q:
            pairs = self.kwargs.get("pairs") or []
            direction = ">=" if ">= $threshold" in q else "<="
            threshold = float(self.kwargs.get("threshold", 0))
            out = []
            for nid, (label, props_dict) in NODES.items():
                for prop, owner in pairs:
                    if owner != label or prop not in props_dict:
                        continue
                    value = float(props_dict[prop])
                    if (direction == ">=" and value >= threshold) or \
                            (direction == "<=" and value <= threshold):
                        out.append((nid, label))
                        break
            return [{"labels": [label], "id": nid}
                    for nid, label in out[: self.kwargs.get("limit", 5)]]
        # global keyword scan: match "tok CONTAINS prop" over banking props
        if "UNWIND $tokens" in q:
            tokens = self.kwargs.get("tokens") or []
            props = self.kwargs.get("props") or []
            out = []
            for nid, (label, props_dict) in NODES.items():
                hits = 0
                for tok in tokens:
                    for prop in props:
                        value = str(props_dict.get(prop, ""))
                        if tok.lower() in value.lower():
                            hits += 1
                if hits:
                    out.append((nid, label, hits))
            out.sort(key=lambda t: t[2], reverse=True)
            return [{"labels": [label], "id": nid} for nid, label, _ in
                    out[: self.kwargs.get("limit", 5)]]
        if "MATCH (d:Dataset)" in q:
            return [{"name": "banking", "rev": 0}]
        if "labels(n) AS labels, n.id AS id" in q:
            eid = self.kwargs.get("id")
            return ([{"labels": [NODES[eid][0]], "id": eid}]
                    if eid in NODES else [])
        if "labels(n) AS labels, n" in q:
            ids = self.kwargs.get("ids") or []
            return [{"labels": [NODES[i][0]], "n": _Node(NODES[i][1])}
                    for i in ids if i in NODES]
        if "MATCH (n)<-[r]-(m)" in q:
            frontier = self.kwargs.get("frontier") or []
            return [{"src": src, "type": rel, "dst": dst}
                    for src, rel, dst in EDGES if dst in frontier]
        if "-[r]->" in q:
            frontier = self.kwargs.get("frontier") or []
            return [{"src": src, "type": rel, "dst": dst}
                    for src, rel, dst in EDGES if src in frontier]
        return []

    def data(self):
        return self._rows

    def single(self):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


class _Session:
    def run(self, query, **kwargs):
        return _Cursor(query, kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Driver:
    def session(self, **kw):
        return _Session()


@pytest.fixture(autouse=True)
def _isolated_audit(tmp_path):
    """Never write the repo's audit trail during these tests."""
    from graphrag import query_pipeline as qp
    from graphrag.traversal_logger import AuditStore

    store = AuditStore(tmp_path / "audit.jsonl")
    old = qp.audit_store
    qp.audit_store = store
    yield store
    qp.audit_store = old
    settings.AUDIT_ENABLED = True
    settings.PII_MODE = "off"
    settings.TENANT_MODE = "off"


def _run(query, **kwargs):
    return run_query(_Driver(), query, answer_mode="extractive", **kwargs)


def test_banking_id_lookup_end_to_end():
    result = _run("What is the status of account ACC-0035?")
    assert "ACC-0035" in result["retrieval"]["seeds"]
    assert "ACC-0035" in result["pruned"]["kept"]
    assert result["retrieval"]["node_count"] >= 1


def test_banking_dispute_neighborhood_keyword():
    # id anchor + neighborhood keyword filter ("dispute" finds DSP-0001)
    result = _run("Is there a dispute on account ACC-0035?")
    kept = set(result["pruned"]["kept"])
    assert "ACC-0035" in kept
    assert "DSP-0001" in kept


def test_banking_who_holds_account():
    result = _run("Who holds account ACC-0035?")
    kept = set(result["pruned"]["kept"])
    assert "CUST-0001" in kept


def test_banking_aml_id_lookup():
    result = _run("What is the status of AML alert AML-0001?")
    assert "AML-0001" in result["pruned"]["kept"]


def test_banking_aml_neighborhood_paraphrase_keywords():
    # anchor + "money laundering" keywords filter within the neighborhood
    result = _run("Which customers have been flagged for possible money "
                  "laundering on account ACC-0035?")
    kept = set(result["pruned"]["kept"])
    assert "CUST-0001" in kept
    assert "AML-0001" in kept


def test_banking_merchant_paraphrase_no_id():
    # NO entity id in the query — global keyword seeding on the unique merchant
    result = _run("Which account posted a payment to Aurora Rare Goods Co.?")
    kept = set(result["pruned"]["kept"])
    assert "ACC-0035" in kept
    assert "TXN-000042" in kept
    # the keyword-seeded node (the merchant's transaction) is the seed
    assert "TXN-000042" in result["retrieval"]["seeds"]


def test_banking_balance_threshold():
    result = _run("Which accounts have a balance over $40,000.00?")
    kept = set(result["pruned"]["kept"])
    assert "ACC-0035" in kept  # balance 42000 >= 40000


def test_banking_negative_probe_refuses():
    result = _run("What is the status of account ACC-99999?")
    assert result["retrieval"]["node_count"] == 0
    low = result["answer"].lower()
    assert "no relevant context" in low or "not determinable" in low


def test_banking_query_is_audited_and_attributed(_isolated_audit):
    from graphrag.identity import UserIdentity

    identity = UserIdentity("bank-analyst", frozenset({"analyst"}), "bank-a")
    settings.TENANT_MODE = "column"
    result = run_query(_Driver(), "Who holds account ACC-0035?",
                       identity=identity, answer_mode="extractive")
    assert result["user"]["subject"] == "bank-analyst"
    record = _isolated_audit.recent(1)[0]
    assert record["user"]["subject"] == "bank-analyst"
    assert record["retrieval"]["seeds"] == ["ACC-0035"]
    assert _isolated_audit.verify()["valid"]


def test_banking_session_wiring():
    from graphrag import sessions
    meta = sessions.get_session_meta("banking_demo")
    assert meta is not None and meta["kind"] == "banking"
    assert sessions.session_for_marker("banking") == "banking_demo"
    cmd = sessions._seed_command(meta)
    assert "ingest_banking_dataset.py" in " ".join(cmd)
    assert "--reset" in cmd
