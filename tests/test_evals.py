"""v2 answer-quality evaluation engine tests (deterministic rules + judge)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest

from graphrag.evals import (eval_faithfulness, eval_groundedness, eval_refusal,
                            eval_relevance, evaluate_answer,
                            evaluate_answer_hybrid, judge_answer,
                            split_statements)

CONTEXT = ("[Claim] CLM-0003 status=IN_REVIEW amount=5000.0\n"
           "[FraudFlag] FRD-0007 severity=MEDIUM reason=amount mismatch\n"
           "(CLM-0003)-[:FRAUD_DETECTED]->(FRD-0007)")

PRUNED_FRAUD = {
    "nodes": [
        {"id": "CLM-0003", "label": "Claim", "props": {"status": "IN_REVIEW"}},
        {"id": "FRD-0007", "label": "FraudFlag",
         "props": {"severity": "MEDIUM", "reason": "amount mismatch"}},
    ],
    "kept": ["CLM-0003", "FRD-0007"],
    "text": CONTEXT,
    "node_count": 2,
}

PRUNED_EMPTY = {"nodes": [], "kept": [], "text": "", "node_count": 0}


# ---------------------------------------------------------------------------
# statement splitting + refusal detection
# ---------------------------------------------------------------------------

def test_split_statements():
    assert split_statements("Yes. It is flagged.") == ["Yes.", "It is flagged."]
    assert split_statements("") == []


def test_faithfulness_rewards_grounded_answer():
    answer = ("Yes — claim CLM-0003 is flagged. Flag FRD-0007 has MEDIUM "
              "severity for amount mismatch.")
    score, issues = eval_faithfulness(answer, CONTEXT)
    assert score == 1.0 and issues == []


def test_faithfulness_flags_ungrounded_statement():
    answer = ("Yes — claim CLM-0003 is flagged. The claimant is Mr. Banerjee.")
    score, issues = eval_faithfulness(answer, CONTEXT)
    assert score == pytest.approx(0.5)   # first sentence grounded, second not
    assert "Banerjee" in " ".join(issues)


def test_faithfulness_ignores_refusal_sentences():
    answer = ("Not determinable from the retrieved context.")
    assert eval_faithfulness(answer, CONTEXT)[0] == 1.0


def test_groundedness_detects_fabrication():
    good, issues = eval_groundedness("CLM-0003 is flagged (FRD-0007).", CONTEXT)
    assert good == 1.0 and not issues
    bad, issues = eval_groundedness("CLM-9999 is flagged.", CONTEXT)
    assert bad == 0.0 and "CLM-9999" in issues


def test_relevance_type_and_id_echo():
    score, _ = eval_relevance("Does claim CLM-0003 have a fraud flag?",
                              "Yes — CLM-0003 is flagged (FRD-0007, MEDIUM).",
                              PRUNED_FRAUD)
    assert score >= 0.7          # type-correct + id echo (honest proxy floor)
    # the full extractive fraud format scores higher (mentions "fraud")
    score2, _ = eval_relevance(
        "Does claim CLM-0003 have a fraud flag?",
        "Yes — 1 fraud flag(s) found for claims CLM-0003: "
        "FRD-0007 (severity=MEDIUM).",
        PRUNED_FRAUD)
    assert score2 > score
    # wrong answer type: asks about coverages, answer names only claims
    score, _ = eval_relevance("Which coverages apply to claim CLM-0003?",
                              "CLM-0003 is IN_REVIEW.", PRUNED_FRAUD)
    assert score < 0.5


def test_refusal_behavior():
    # empty retrieval: refusing scores 1, hallucinating scores 0
    assert eval_refusal("q", "Not determinable from the retrieved context.",
                        answerable=False)[0] == 1.0
    assert eval_refusal("q", "It is definitely fraud.", answerable=False)[0] == 0.0
    # answerable query: answering scores 1, refusing scores 0
    assert eval_refusal("q", "CLM-0003 is flagged.", answerable=True)[0] == 1.0
    assert eval_refusal("q", "Cannot determine.", answerable=True)[0] == 0.0


def test_evaluate_answer_composite():
    res = evaluate_answer(
        "Does claim CLM-0003 have a fraud flag?",
        "Yes — CLM-0003 is flagged (FRD-0007, MEDIUM severity).",
        PRUNED_FRAUD)
    assert res["engine"] == "rules"
    for key in ("faithfulness", "relevance", "groundedness", "refusal", "overall"):
        assert 0.0 <= res[key] <= 1.0
    assert res["faithfulness"] == 1.0
    assert res["groundedness"] == 1.0


def test_evaluate_answer_negative_probe_refusal():
    res = evaluate_answer(
        "Does claim CLM-99999 have a fraud flag?",
        "No relevant context found for this query.", PRUNED_EMPTY)
    assert res["refusal"] == 1.0
    assert res["overall"] >= 0.9


# ---------------------------------------------------------------------------
# LLM judge path
# ---------------------------------------------------------------------------

class _FakeResponse:
    status_code = 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class _FakeProvider:
    name = "openai"

    def __init__(self, payload):
        self._payload = payload
        self.model = "gpt-4o-mini"

    def generate(self, prompt, **kwargs):
        from graphrag.llm.base import LLMResult
        return LLMResult(text=json.dumps(self._payload), provider=self.name,
                         model=self.model)


def test_judge_answer_parses_rubric(monkeypatch):
    monkeypatch.setattr("graphrag.llm.factory.get_provider",
                        lambda mode: _FakeProvider({
                            "faithfulness": 1.0, "relevance": 0.9,
                            "groundedness": 1.0, "refusal": 1.0,
                            "explanation": "fully grounded"}))
    judged = judge_answer("Is CLM-0003 flagged?", "Yes.", PRUNED_FRAUD)
    assert judged is not None
    assert judged["engine"].startswith("llm:openai")
    assert judged["faithfulness"] == 1.0
    assert judged["overall"] == pytest.approx(0.97)


def test_judge_answer_falls_back_to_none_when_down(monkeypatch):
    monkeypatch.setattr("graphrag.llm.factory.get_provider", lambda mode: None)
    assert judge_answer("q", "a", PRUNED_FRAUD) is None
    # hybrid path degrades to rules
    res = evaluate_answer_hybrid("q", "a", PRUNED_FRAUD, prefer_llm=True)
    assert res["engine"] == "rules"


def test_judge_answer_malformed_json_returns_none(monkeypatch):
    monkeypatch.setattr("graphrag.llm.factory.get_provider",
                        lambda mode: _FakeProvider({"not": "json"}))
    # judge returns None on malformed payload -> hybrid falls back
    res = evaluate_answer_hybrid("q", "a", PRUNED_FRAUD, prefer_llm=True)
    assert res["engine"] == "rules"


# ---------------------------------------------------------------------------
# golden set sanity
# ---------------------------------------------------------------------------

def test_golden_set_is_well_formed():
    golden = json.loads(
        (ROOT / "data" / "benchmarks" / "golden_questions.json").read_text())
    assert golden["count"] >= 40
    categories = {q["category"] for q in golden["questions"]}
    assert {"fraud", "status", "coverage", "negative", "paraphrase"} <= categories
    for q in golden["questions"]:
        assert q["query"] and "category" in q
        assert "answerable" in q and "expected_ids" in q
    # negative probes must be marked unanswerable
    for q in golden["questions"]:
        if q["category"] == "negative":
            assert q["answerable"] is False
