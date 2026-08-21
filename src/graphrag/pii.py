"""PII classification & masking (v2 — WS-B, G2).

Insurance-graph properties are classified per (label, property); PII-classed
fields are redacted from retrieval context and answers unless the caller's
role can read them (``settings.PII_READER_ROLES``).

Modes (``settings.PII_MODE``):

* ``off``  — v1 behavior: raw fields everywhere (local dev, tests)
* ``mask`` — PII-classified fields are replaced with stable masks before the
  context is serialized for ranking/answering, and again in the final answer
  text. IDs and non-PII business fields (amounts, statuses, causes) pass
  through untouched, so retrieval/benchmark semantics do not change.

**Encryption at rest** (``settings.PII_ENCRYPTION_KEY``): PII-classified
properties are Fernet-encrypted (``enc:v1:<token>``) before they are written
to Neo4j and decrypted on read. Write paths (``graph_updater``, ``load_nodes``)
call ``encrypt_node``; the retrieval read path (``_fetch_nodes``) calls
``decrypt_node``. Values already carrying the prefix are never re-encrypted
(idempotent), and non-PII fields stay plaintext so graph queries/filters keep
working. Envelope encryption (KMS-wrapped data key) is the production upgrade
path; this is the single-key baseline.
"""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass

from graphrag.config import settings

# Classification per (label, property). Properties not listed here are NONE
# (never masked). PHI-class data (medical/health) gets its own class so
# policies can treat it more strictly later.
PII_CLASSES = ("PII_IDENTITY", "PII_CONTACT", "PII_HEALTH")

_CLASSIFIED: dict[tuple[str, str], str] = {
    # Policyholder — the PII-heavy entity
    ("Policyholder", "name"): "PII_IDENTITY",
    ("Policyholder", "dob"): "PII_IDENTITY",
    ("Policyholder", "address"): "PII_CONTACT",
    ("Policyholder", "phone"): "PII_CONTACT",
    ("Policyholder", "email"): "PII_CONTACT",
    # Investigator — internal staff PII
    ("Investigator", "name"): "PII_IDENTITY",
    ("Investigator", "email"): "PII_CONTACT",
    # Any entity carrying contact/identity-like fields (custom CSVs etc.)
    ("Policy", "policyholder_name"): "PII_IDENTITY",
}

# Generic field-name fallback for unlisted labels (custom CSV uploads):
# a property whose *name* matches these patterns is classified by pattern.
_NAME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(dob|date_of_birth|birthdate)$", re.I), "PII_IDENTITY"),
    (re.compile(r"^(name|full_name|policyholder_name|customer_name)$", re.I), "PII_IDENTITY"),
    (re.compile(r"^(address|street|city|zip|postcode)$", re.I), "PII_CONTACT"),
    (re.compile(r"^(phone|mobile|telephone|contact_no)$", re.I), "PII_CONTACT"),
    (re.compile(r"^email", re.I), "PII_CONTACT"),
]


@dataclass(frozen=True)
class MaskingPolicy:
    """Who may see which PII classes in this request."""

    may_read_identity: bool
    may_read_contact: bool
    may_read_health: bool

    @property
    def active(self) -> bool:
        return settings.PII_MODE == "mask"

    def allows(self, pii_class: str) -> bool:
        if pii_class == "PII_IDENTITY":
            return self.may_read_identity
        if pii_class == "PII_CONTACT":
            return self.may_read_contact
        if pii_class == "PII_HEALTH":
            return self.may_read_health
        return True

    @classmethod
    def for_roles(cls, roles: set[str] | None) -> "MaskingPolicy":
        """Policy for a caller's role set (None = anonymous/dev → full access
        only when PII_MODE is off; masked mode defaults to deny)."""
        roles = set(roles or ())
        if settings.PII_MODE != "mask":
            return cls(True, True, True)
        readers = {r.strip() for r in settings.PII_READER_ROLES.split(",") if r.strip()}
        can_read = bool(roles & readers)
        return cls(can_read, can_read, can_read)


def classify(label: str, prop: str) -> str | None:
    """PII class of a property (explicit table first, name patterns second)."""
    key = (label, prop)
    if key in _CLASSIFIED:
        return _CLASSIFIED[key]
    for pattern, cls in _NAME_PATTERNS:
        if pattern.fullmatch(prop):
            return cls
    return None


_EMAIL_RE = re.compile(r"(?i)\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d\s\-()]{6,}\d)(?!\d)")
_DOB_RE = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")


def mask_value(value, pii_class: str) -> str:
    """A stable, readable mask for one PII-classed value."""
    text = str(value)
    if pii_class == "PII_CONTACT":
        if "@" in text:
            local, _, domain = text.partition("@")
            return f"{local[:2]}***@{domain}" if local else "***"
        return re.sub(r"\d", "#", text)
    if pii_class == "PII_HEALTH":
        return "[REDACTED-HEALTH]"
    return "[REDACTED]"


def mask_text(text: str) -> str:
    """Best-effort scrub of PII patterns in free text (answers, descriptions).

    Used on LLM *output* where field-level classification cannot apply.
    Conservative: masks emails/phones/DOBs, leaves the rest intact.
    """
    out = _EMAIL_RE.sub("[EMAIL-REDACTED]", text)
    out = _PHONE_RE.sub("[PHONE-REDACTED]", out)
    return _DOB_RE.sub("[DOB-REDACTED]", out)


def redact_node(node: dict, policy: MaskingPolicy) -> dict:
    """Copy a node with PII-classed props masked per policy (or unchanged)."""
    if not policy.active:
        return node
    props = dict(node.get("props", {}))
    changed = False
    for prop, value in list(props.items()):
        cls = classify(node.get("label", ""), prop)
        if cls and not policy.allows(cls):
            props[prop] = mask_value(value, cls)
            changed = True
    if not changed:
        return node
    return {**node, "props": props}


def scrub_answer(text: str, policy: MaskingPolicy) -> str:
    """Redact PII patterns from a generated answer (role-blind safety net)."""
    if not policy.active:
        return text
    return mask_text(text)


# ---------------------------------------------------------------------------
# encryption at rest (Fernet; lazy import so cryptography stays optional)
# ---------------------------------------------------------------------------

_ENCRYPT_PREFIX = "enc:v1:"


def encryption_enabled() -> bool:
    """True when PII field encryption is configured."""
    return bool(settings.PII_ENCRYPTION_KEY)


def _fernet():
    from cryptography.fernet import Fernet

    return Fernet(_fernet_key())


def _fernet_key() -> bytes:
    """32-byte urlsafe-base64 key from the setting.

    Accepts a real urlsafe-base64 32-byte key directly; anything else
    (passphrase, hex) is SHA-256-derived so users can't lock themselves out
    with a malformed key.
    """
    raw = settings.PII_ENCRYPTION_KEY.encode("utf-8")
    try:
        decoded = base64.urlsafe_b64decode(raw + b"=" * (-len(raw) % 4))
        if len(decoded) == 32:
            return base64.urlsafe_b64encode(decoded)
    except Exception:
        pass
    return base64.urlsafe_b64encode(hashlib.sha256(raw).digest())


def encrypt_value(value) -> str:
    """Encrypt one PII value (idempotent; non-string values stringified)."""
    if isinstance(value, (int, float, bool)):
        value = str(value)
    text = str(value)
    if text.startswith(_ENCRYPT_PREFIX):
        return text
    token = _fernet().encrypt(text.encode("utf-8")).decode("ascii")
    return _ENCRYPT_PREFIX + token


def decrypt_value(value) -> str:
    """Decrypt an ``enc:v1:`` value; pass anything else through unchanged."""
    text = str(value)
    if not text.startswith(_ENCRYPT_PREFIX):
        return text
    return _fernet().decrypt(text[len(_ENCRYPT_PREFIX):].encode("ascii")).decode("utf-8")


def encrypt_node(node: dict) -> dict:
    """Write path: encrypt PII-classed props (no-op when encryption is off)."""
    if not encryption_enabled():
        return node
    props = dict(node.get("props", {}))
    changed = False
    for prop, value in list(props.items()):
        cls = classify(node.get("label", ""), prop)
        if cls and not str(value).startswith(_ENCRYPT_PREFIX):
            props[prop] = encrypt_value(value)
            changed = True
    return {**node, "props": props} if changed else node


def decrypt_node(node: dict) -> dict:
    """Read path: decrypt PII-classed props (no-op when encryption is off)."""
    if not encryption_enabled():
        return node
    props = dict(node.get("props", {}))
    changed = False
    for prop, value in list(props.items()):
        if classify(node.get("label", ""), prop) and str(value).startswith(_ENCRYPT_PREFIX):
            props[prop] = decrypt_value(value)
            changed = True
    return {**node, "props": props} if changed else node
