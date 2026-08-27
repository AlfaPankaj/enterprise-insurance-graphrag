"""Answer-quality evaluation (v2 — WS-C, G15).

For the first time, the project measures *answer quality*, not just retrieval
fidelity. Two engines, one interface:

* **rules** — deterministic, zero-dependency proxy metrics:
  * ``faithfulness`` — every statement of the answer must be grounded in the
    retrieved context (entity ids / significant tokens present)
  * ``relevance`` — the answer addresses what the query asked: answer-type
    match (query asks for coverages → answer names coverage nodes), query
    entity-id echo, query content-word coverage
  * ``groundedness`` — no fabricated ids/values (reuses the guardrails check)
  * ``refusal`` — unanswerable queries (empty retrieval) must refuse instead
    of guessing; answerable queries must not refuse
  * ``overall`` — weighted composite (0..1)
* **llm** — a provider-based judge (same rubric, JSON output) via the v2 LLM
  layer; ``hybrid`` uses the judge when a provider is up, else the rules.

``evaluate_answer(query, answer, pruned, mode="rules")`` returns the score
dict; ``judge_answer(...)`` is the provider path. Scores are honest proxies —
they are calibrated on the golden set by ``scripts/benchmark_answer_quality.py``
and shown with the exact engine used.
"""

from __future__ import annotations

import json
import logging
import re

from graphrag.config import settings
from graphrag.guardrails import check_groundedness
from graphrag.graph_retriever import ENTITY_ID_RE, query_tokens
from graphrag.reranker import _label_prior_hits

logger = logging.getLogger("graphrag.evals")

WEIGHTS = {
    "faithfulness": 0.4,
    "relevance": 0.3,
    "groundedness": 0.2,
    "refusal": 0.1,
}

_REFUSAL_PHRASES = (
    "not determinable", "cannot determine", "unable to determine",
    "no information", "does not contain", "does not provide", "does not mention",
    "cannot confirm", "no evidence", "no indication", "not mentioned",
    "no relevant context", "no fraud flag",
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
# abbreviations whose period must not end a sentence
_ABBREV = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Inc|Ltd|Co|vs|etc)\.", re.I)


def split_statements(text: str) -> list[str]:
    """Split on sentence boundaries, protecting common abbreviations."""
    masked = _ABBREV.sub(lambda m: m.group(0).replace(".", "\u0001"), text or "")
    parts = [s.replace("\u0001", ".").strip()
             for s in _SENT_SPLIT.split(masked) if s.strip()]
    return parts


def is_refusal(text: str) -> bool:
    low = (text or "").lower()
    return any(p in low for p in _REFUSAL_PHRASES)


# ---------------------------------------------------------------------------
# deterministic metrics
# ---------------------------------------------------------------------------

def eval_groundedness(answer: str, context_text: str) -> tuple[float, list[str]]:
    """1.0 when no entity id / currency value in the answer is fabricated."""
    res = check_groundedness(answer, context_text)
    issues = res.ungrounded_ids + res.ungrounded_values
    return (0.0 if issues else 1.0), issues


def eval_faithfulness(answer: str, context_text: str) -> tuple[float, list[str]]:
    """Fraction of answer statements that are grounded in the context.

    A statement is grounded when it names an entity id present in the context
    or shares a significant token (len ≥ 4) with it. Refusal statements are
    excluded (the refusal metric owns them).
    """
    statements = split_statements(answer)
    if not statements:
        return 1.0, []
    context_ids = {i.upper() for i in ENTITY_ID_RE.findall(context_text)}
    context_words = {w for w in re.findall(r"[a-z]{4,}", context_text.lower())}
    grounded = 0
    issues: list[str] = []
    for stmt in statements:
        if is_refusal(stmt):
            continue  # not a claim
        stmt_ids = {i.upper() for i in ENTITY_ID_RE.findall(stmt)}
        words = set(re.findall(r"[a-z]{4,}", stmt.lower()))
        if stmt_ids & context_ids or words & context_words:
            grounded += 1
        else:
            issues.append(stmt[:120])
    total = len([s for s in statements if not is_refusal(s)])
    if total == 0:
        return 1.0, []
    return grounded / total, issues


_FILLER_WORDS = {
    "does", "have", "what", "which", "when", "where", "with", "that",
    "this", "there", "their", "about", "would", "could", "should",
}

def _content_words(text: str) -> set[str]:
    """Lowercase words ≥ 4 chars, minus generic fillers (schema nouns KEPT —
    "fraud"/"coverage" are strong relevance signals in answers)."""
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower())
            if w not in _FILLER_WORDS}


def eval_relevance(query: str, answer: str, pruned: dict | None = None,
                   answerable: bool = True) -> tuple[float, list[str]]:
    """Does the answer address the query? (answer-type, id echo, word coverage)."""
    labels = {n["id"]: n.get("label", "") for n in (pruned or {}).get("nodes", [])}
    answer_ids = ENTITY_ID_RE.findall(answer)
    issues: list[str] = []

    # correct refusal is the right "answer" to an unanswerable question
    if is_refusal(answer):
        return (1.0, []) if not answerable else (0.0, ["refused an answerable query"])

    # 1) answer-type match: the query asks about Coverages -> answer must name
    #    coverage nodes. Entities named by id IN THE QUERY are anchors, not the
    #    asked-for type ("coverages of claim CLM-0003" expects Coverage ids,
    #    not the Claim anchor). Binary: ≥1 expected-type id = addressed.
    anchor_labels = {labels.get(i) for i in ENTITY_ID_RE.findall(query)}
    expected_labels = _label_prior_hits(query) - {l for l in anchor_labels if l}
    if expected_labels:
        hits = sum(1 for i in answer_ids if labels.get(i) in expected_labels)
        if hits:
            type_match = 1.0
        else:
            type_match, issues = 0.0, issues + \
                [f"answer names no {'/'.join(sorted(expected_labels))} nodes"]
    else:
        type_match = 1.0

    # 2) id echo: entity ids named in the query appear in the answer
    query_ids = {i.upper() for i in ENTITY_ID_RE.findall(query)}
    id_match = (sum(1 for i in query_ids if i.upper() in {a.upper() for a in answer_ids})
                / len(query_ids)) if query_ids else 0.0

    # 3) content-word coverage (schema nouns included; prefix variants count —
    #    "flag" is covered by "flagged", "investigate" by "investigates")
    q_words = _content_words(query)
    a_words = _content_words(answer)

    def _covered(qword: str) -> bool:
        return any(qword == a or (len(qword) >= 4 and len(a) >= 4 and
                                  (a.startswith(qword) or qword.startswith(a)))
                   for a in a_words)

    coverage = (sum(1 for t in q_words if _covered(t)) / len(q_words)) if q_words else 0.0

    if expected_labels:
        # the query asked for a specific entity type: answer-type match is the
        # dominant signal (id echo alone cannot rescue a wrong-typed answer)
        relevance = 0.6 * type_match + 0.4 * coverage
    else:
        relevance = 0.5 * max(coverage, id_match) + 0.5 * coverage
    if expected_labels and relevance < 0.5:
        issues.append("answer-type mismatch for " + ",".join(sorted(expected_labels)))
    return round(min(relevance, 1.0), 4), issues


def eval_refusal(query: str, answer: str, answerable: bool) -> tuple[float, list[str]]:
    """1.0 = correct refusal behavior (refuse when empty, answer when not)."""
    refused = is_refusal(answer)
    if not answerable:
        return (1.0, []) if refused else (0.0, ["hallucinated answer for an empty retrieval"])
    return (0.0, ["refused an answerable query"]) if refused else (1.0, [])


def evaluate_answer(query: str, answer: str, pruned: dict,
                    answerable: bool | None = None) -> dict:
    """Deterministic (rules) evaluation of one answer. Returns the score dict.

    ``answerable`` defaults to ``pruned["node_count"] > 0`` (retrieval found
    context); golden-set entries may override it explicitly.
    """
    context_text = pruned.get("text", "")
    if answerable is None:
        answerable = bool(pruned.get("nodes")) and pruned.get("node_count", 0) > 0
    faithfulness, faith_issues = eval_faithfulness(answer, context_text)
    relevance, rel_issues = eval_relevance(query, answer, pruned, answerable)
    groundedness, ground_issues = eval_groundedness(answer, context_text)
    refusal, refusal_issues = eval_refusal(query, answer, answerable)
    overall = round(
        WEIGHTS["faithfulness"] * faithfulness
        + WEIGHTS["relevance"] * relevance
        + WEIGHTS["groundedness"] * groundedness
        + WEIGHTS["refusal"] * refusal,
        4,
    )
    return {
        "engine": "rules",
        "faithfulness": round(faithfulness, 4),
        "relevance": relevance,
        "groundedness": groundedness,
        "refusal": refusal,
        "overall": overall,
        "answerable": answerable,
        "issues": {
            "faithfulness": faith_issues,
            "relevance": rel_issues,
            "groundedness": ground_issues,
            "refusal": refusal_issues,
        },
    }


# ---------------------------------------------------------------------------
# LLM judge (optional provider path)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = (
    "You are an answer-quality judge for an insurance knowledge-graph system.\n"
    "Score the ANSWER against the QUESTION and the retrieved CONTEXT on four "
    "rubrics, each 0.0 to 1.0:\n"
    "  faithfulness: every claim in the answer is supported by the context\n"
    "  relevance: the answer addresses the question asked\n"
    "  groundedness: the answer invents no entity ids or values absent from the context\n"
    "  refusal: if the context cannot answer the question, the answer refuses "
    "(1.0); if the context CAN answer it, the answer does not refuse (1.0)\n"
    "Return ONLY JSON: {\"faithfulness\": f, \"relevance\": r, \"groundedness\": g, "
    "\"refusal\": rf, \"explanation\": \"one sentence\"}\n\n"
    "QUESTION: {query}\n\nCONTEXT:\n{context}\n\nANSWER: {answer}\n"
)


def judge_answer(query: str, answer: str, pruned: dict) -> dict | None:
    """Provider-based rubric judge; None when no provider is up.

    Weights the same composite as the rules engine, then falls back to
    ``evaluate_answer`` when the judge output is malformed.
    """
    from graphrag.llm.factory import get_provider

    provider = get_provider("auto")
    if provider is None:
        return None
    context = pruned.get("text", "")[:6000]
    prompt = (_JUDGE_PROMPT
              .replace("{query}", query)
              .replace("{context}", context)
              .replace("{answer}", answer[:2000]))
    try:
        result = provider.generate(prompt, max_tokens=256, temperature=0.0)
        raw = result.text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").removeprefix("json")
        scores = json.loads(raw)
        for key in ("faithfulness", "relevance", "groundedness", "refusal"):
            scores[key] = float(min(max(float(scores[key]), 0.0), 1.0))
    except Exception as exc:  # noqa: BLE001 - judge must never crash the eval
        logger.warning("LLM judge failed (%s), falling back to rules", exc)
        return None
    overall = round(sum(WEIGHTS[k] * scores[k] for k in WEIGHTS), 4)
    return {
        "engine": f"llm:{provider.name}:{result.model}",
        "faithfulness": scores["faithfulness"],
        "relevance": scores["relevance"],
        "groundedness": scores["groundedness"],
        "refusal": scores["refusal"],
        "overall": overall,
        "answerable": bool(pruned.get("nodes")),
        "explanation": scores.get("explanation", ""),
        "issues": {},
    }


def evaluate_answer_hybrid(query: str, answer: str, pruned: dict,
                           answerable: bool | None = None,
                           prefer_llm: bool = False) -> dict:
    """LLM judge when requested+available, deterministic rules otherwise."""
    if prefer_llm:
        judged = judge_answer(query, answer, pruned)
        if judged is not None:
            return judged
    return evaluate_answer(query, answer, pruned, answerable=answerable)
