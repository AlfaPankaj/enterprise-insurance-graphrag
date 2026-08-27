"""v2 PII encryption-at-rest tests (Fernet, write paths, read path)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest

from graphrag.config import settings
from graphrag.pii import (decrypt_node, decrypt_value, encrypt_node,
                          encrypt_value, encryption_enabled)

KEY = "k" * 32


@pytest.fixture(autouse=True)
def _clear_key():
    yield
    settings.PII_ENCRYPTION_KEY = ""


HOLDER = {"id": "PH-0001", "label": "Policyholder",
          "props": {"name": "Alice Example", "email": "alice@example.com",
                    "risk_score": 42.0}}
CLAIM = {"id": "CLM-0003", "label": "Claim",
         "props": {"amount": 12000.0, "cause": "fire"}}


# ---------------------------------------------------------------------------
# value-level crypto
# ---------------------------------------------------------------------------

def test_roundtrip_string_and_numeric(monkeypatch):
    monkeypatch.setattr(settings, "PII_ENCRYPTION_KEY", KEY)
    for value in ("Alice Example", 42.5, True):
        token = encrypt_value(value)
        assert token.startswith("enc:v1:")
        assert decrypt_value(token) == str(value)


def test_encrypt_idempotent(monkeypatch):
    monkeypatch.setattr(settings, "PII_ENCRYPTION_KEY", KEY)
    once = encrypt_value("secret")
    assert encrypt_value(once) == once


def test_decrypt_passthrough_for_plaintext(monkeypatch):
    monkeypatch.setattr(settings, "PII_ENCRYPTION_KEY", KEY)
    assert decrypt_value("plain value") == "plain value"
    assert decrypt_value(123) == "123"


def test_tampered_token_raises(monkeypatch):
    monkeypatch.setattr(settings, "PII_ENCRYPTION_KEY", KEY)
    token = encrypt_value("secret")
    tampered = token[:-4] + ("AAAA" if not token.endswith("AAAA") else "BBBB")
    from cryptography.fernet import InvalidToken

    with pytest.raises(InvalidToken):
        decrypt_value(tampered)


def test_wrong_key_cannot_decrypt(monkeypatch):
    monkeypatch.setattr(settings, "PII_ENCRYPTION_KEY", KEY)
    token = encrypt_value("secret")
    monkeypatch.setattr(settings, "PII_ENCRYPTION_KEY", "x" * 32)
    from cryptography.fernet import InvalidToken

    with pytest.raises(InvalidToken):
        decrypt_value(token)


def test_passphrase_key_derivation_is_stable(monkeypatch):
    monkeypatch.setattr(settings, "PII_ENCRYPTION_KEY", "my-passphrase")
    a = encrypt_value("x")
    # a fresh call with the same passphrase must produce the same key
    assert decrypt_value(a) == "x"
    b = encrypt_value("x")
    assert decrypt_value(b) == "x"


# ---------------------------------------------------------------------------
# node-level helpers
# ---------------------------------------------------------------------------

def test_encrypt_node_classified_fields_only(monkeypatch):
    monkeypatch.setattr(settings, "PII_ENCRYPTION_KEY", KEY)
    out = encrypt_node(HOLDER)
    assert out["props"]["name"].startswith("enc:v1:")
    assert out["props"]["email"].startswith("enc:v1:")
    assert out["props"]["risk_score"] == 42.0          # business field plaintext
    assert decrypt_node(out)["props"]["name"] == "Alice Example"


def test_encrypt_node_noop_when_disabled():
    assert encryption_enabled() is False
    assert encrypt_node(HOLDER) is HOLDER
    assert decrypt_node(HOLDER) is HOLDER


def test_encrypt_node_noop_for_claim(monkeypatch):
    monkeypatch.setattr(settings, "PII_ENCRYPTION_KEY", KEY)
    assert encrypt_node(CLAIM) is CLAIM                   # nothing PII-classed


# ---------------------------------------------------------------------------
# write path: load_nodes encrypts before MERGE
# ---------------------------------------------------------------------------

class _RecordingRunner:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def run(self, query, **kwargs):
        self.calls.append((query, kwargs))


def test_load_nodes_encrypts_pii_props(monkeypatch):
    from scripts.seed_graph import load_nodes

    monkeypatch.setattr(settings, "PII_ENCRYPTION_KEY", KEY)
    runner = _RecordingRunner()
    load_nodes(runner, {"Policyholder": [{"id": "PH-0001", "props": {
        "name": "Alice Example", "risk_score": 12.0}}]})
    rows = runner.calls[0][1]["rows"]
    assert rows[0]["props"]["name"].startswith("enc:v1:")
    assert rows[0]["props"]["risk_score"] == 12.0        # untouched
    assert decrypt_value(rows[0]["props"]["name"]) == "Alice Example"


def test_load_nodes_plaintext_without_key():
    from scripts.seed_graph import load_nodes

    runner = _RecordingRunner()
    load_nodes(runner, {"Policyholder": [{"id": "PH-0001", "props": {
        "name": "Alice Example"}}]})
    rows = runner.calls[0][1]["rows"]
    assert rows[0]["props"]["name"] == "Alice Example"


# ---------------------------------------------------------------------------
# write path: CDC upsert encrypts before SET
# ---------------------------------------------------------------------------

class _Cursor:
    def single(self):
        return None

    def data(self):
        return []


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

    def begin_transaction(self):
        return self._tx

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Driver:
    def __init__(self, tx):
        self._tx = tx

    def session(self, **kw):
        return _Session(self._tx)


def test_cdc_upsert_encrypts_pii(monkeypatch):
    from graphrag.graph_updater import update_graph_surgically

    monkeypatch.setattr(settings, "PII_ENCRYPTION_KEY", KEY)
    tx = _Tx()
    changes = {"added": [{"label": "Policyholder", "id": "PH-9",
                          "props": {"name": "Secret Person", "risk_score": 3.0}}],
               "modified": [], "deleted": []}
    update_graph_surgically(_Driver(tx), "DOC-1", changes)
    upsert = next(c for c in tx.calls if "MERGE (n:Policyholder" in c[0])
    props = upsert[1]["props"]
    assert props["name"].startswith("enc:v1:")
    assert props["risk_score"] == 3.0
    assert decrypt_value(props["name"]) == "Secret Person"


# ---------------------------------------------------------------------------
# read path: _fetch_nodes decrypts
# ---------------------------------------------------------------------------

class _Node:
    def __init__(self, props):
        self._props = props

    def __getitem__(self, k):
        return self._props[k]

    def items(self):
        return self._props.items()

    def keys(self):
        return self._props.keys()


class _FetchCursor:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FetchSession:
    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def run(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return _FetchCursor(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_nodes_decrypts_on_read(monkeypatch):
    from graphrag.graph_retriever import _fetch_nodes

    monkeypatch.setattr(settings, "PII_ENCRYPTION_KEY", KEY)
    token = encrypt_value("Alice Example")
    session = _FetchSession(
        [{"labels": ["Policyholder"],
          "n": _Node({"id": "PH-0001", "name": token, "risk_score": 9.0})}]
    )
    nodes = _fetch_nodes(session, ["PH-0001"])
    assert nodes["PH-0001"]["props"]["name"] == "Alice Example"
    assert nodes["PH-0001"]["props"]["risk_score"] == 9.0


def test_fetch_nodes_plaintext_without_key():
    from graphrag.graph_retriever import _fetch_nodes

    session = _FetchSession(
        [{"labels": ["Claim"], "n": _Node({"id": "CLM-1", "amount": 5.0})}]
    )
    nodes = _fetch_nodes(session, ["CLM-1"])
    assert nodes["CLM-1"]["props"]["amount"] == 5.0
