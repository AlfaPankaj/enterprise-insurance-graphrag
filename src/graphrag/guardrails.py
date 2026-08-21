"""Query guardrails (v2 — WS-B, G7).

Layered checks around the pipeline (OWASP LLM Top-10 hardening):

* **input** — instruction-injection heuristics on the query and on ingested
  document text (a malicious PDF must not steer extraction/answers);
* **output** — groundedness (does the answer name entity ids or values that
  are NOT in the retrieved context?), refusal detection, and PII echo scans.

Guardrails *flag* by default and never silently rewrite answers — the
pipeline records every finding in the result and the audit record, so a
blocking policy can be layered on without losing explainability.

Enabled via ``settings.GUARDRAILS_ENABLED`` (default off = v1 behavior).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from graphrag.config import settings
from graphrag.graph_retriever import ENTITY_ID_RE

_INJECTION_HINTS = (
    "ignore previous", "ignore all", "ignore the above", "disregard",
    "system prompt", "you are now", "act as", "jailbreak", "do anything now",
    "dan mode", "developer mode",
)

_REFUSAL_HINTS = (
    "not determinable", "cannot determine", "unable to determine",
    "no information", "does not contain", "i don't know", "insufficient",
)

_CURRENCY_RE = re.compile(r"\$\d[\d,]*")


@dataclass
class GuardrailResult:
    """Findings for one query execution. ``blocked`` drives policy."""

    injection_detected: bool = False
    injection_hits: list[str] = field(default_factory=list)
    ungrounded_ids: list[str] = field(default_factory=list)
    ungrounded_values: list[str] = field(default_factory=list)
    refusal_detected: bool = False
    pii_echoed: bool = False

    @property
    def blocked(self) -> bool:
        """A finding severe enough to refuse the answer (policy hook)."""
        return self.injection_detected

    @property
    def clean(self) -> bool:
        return not (self.injection_detected or self.ungrounded_ids
                    or self.ungrounded_values or self.pii_echoed)

    def as_dict(self) -> dict:
        return {
            "injection_detected": self.injection_detected,
            "injection_hits": self.injection_hits,
            "ungrounded_ids": self.ungrounded_ids,
            "ungrounded_values": self.ungrounded_values,
            "refusal_detected": self.refusal_detected,
            "pii_echoed": self.pii_echoed,
            "blocked": self.blocked,
        }


def scan_query(query: str) -> GuardrailResult:
    """Input screening: instruction-injection heuristics on the user query."""
    res = GuardrailResult()
    low = query.lower()
    for hint in _INJECTION_HINTS:
        if hint in low:
            res.injection_detected = True
            res.injection_hits.append(hint)
    return res


def scan_document(text: str) -> GuardrailResult:
    """Input screening for ingested document text (PDF/CSV content)."""
    return scan_query(text)


def check_groundedness(answer: str, context_text: str) -> GuardrailResult:
    """Output check: every entity id / currency value in the answer must
    exist in the retrieved context (no fabricated references)."""
    res = GuardrailResult()
    answer_ids = {i.upper() for i in ENTITY_ID_RE.findall(answer)}
    context_ids = {i.upper() for i in ENTITY_ID_RE.findall(context_text)}
    res.ungrounded_ids = sorted(answer_ids - context_ids)
    for value in _CURRENCY_RE.findall(answer):
        if value not in context_text:
            res.ungrounded_values.append(value)
    return res


def check_output(answer: str, context_text: str) -> GuardrailResult:
    """Output screening: groundedness + refusal detection + PII echo scan."""
    res = check_groundedness(answer, context_text)
    low = answer.lower()
    res.refusal_detected = any(h in low for h in _REFUSAL_HINTS)
    from graphrag.pii import mask_text
    res.pii_echoed = mask_text(answer) != answer
    return res


def run_guardrails(query: str, answer: str, context_text: str,
                   document_text: str | None = None) -> GuardrailResult:
    """Run the full check set (no-op result when guardrails are disabled)."""
    if not settings.GUARDRAILS_ENABLED:
        return GuardrailResult()
    res = scan_query(query)
    if document_text:
        doc = scan_document(document_text)
        res.injection_detected = res.injection_detected or doc.injection_detected
        res.injection_hits += [h for h in doc.injection_hits if h not in res.injection_hits]
    out = check_output(answer, context_text)
    res.ungrounded_ids = out.ungrounded_ids
    res.ungrounded_values = out.ungrounded_values
    res.refusal_detected = out.refusal_detected
    res.pii_echoed = out.pii_echoed
    return res
