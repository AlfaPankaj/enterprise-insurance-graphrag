"""Answer generator tests — deterministic extractive + mocked Ollama paths."""

import pytest

from graphrag.answer_generator import (ModelNotFoundError, extractive_answer,
                                       generate_answer, ollama_available)
from graphrag.config import settings

PRUNED = {
    "nodes": [
        {"id": "CLM-0003", "label": "Claim",
         "props": {"status": "IN_REVIEW", "amount": 5000.0}},
        {"id": "FRD-CLM-0003", "label": "FraudFlag",
         "props": {"severity": "MEDIUM", "reason": "amount mismatch"}},
        {"id": "INV-0001", "label": "Investigator",
         "props": {"name": "Alice", "role": "SENIOR_INVESTIGATOR"}},
    ],
    "edges": [
        {"source": "CLM-0003", "type": "FRAUD_DETECTED", "target": "FRD-CLM-0003"},
        {"source": "CLM-0003", "type": "INVESTIGATES_CLAIM", "target": "INV-0001"},
    ],
    "kept": ["CLM-0003", "FRD-CLM-0003"],
    "text": "[Claim] CLM-0003 status=IN_REVIEW\n"
            "[FraudFlag] FRD-CLM-0003 severity=MEDIUM\n"
            "(CLM-0003)-[:FRAUD_DETECTED]->(FRD-CLM-0003)",
}


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            from httpx import HTTPStatusError
            from httpx import Request as _Req
            from httpx import Response as _Resp
            req = _Req("POST", "http://localhost:11434/api/generate")
            resp = _Resp(self.status_code, request=req, json=self._payload)
            raise HTTPStatusError(str(self.status_code), request=req, response=resp)


# ---------------------------------------------------------------------------
# extractive (default, deterministic)
# ---------------------------------------------------------------------------

def test_extractive_fraud_answer():
    out = generate_answer("Does claim CLM-0003 have a fraud flag?", PRUNED,
                          mode="extractive")
    assert out["mode"] == "extractive"
    assert out["model"] is None
    assert "FRD-CLM-0003" in out["answer"]
    assert "MEDIUM" in out["answer"]


def test_extractive_investigator_answer():
    pruned = {**PRUNED, "kept": ["INV-0001"]}
    out = extractive_answer("Who investigates claim CLM-0003?", pruned)
    assert "Alice" in out["answer"] and "INV-0001" in out["answer"]


def test_extractive_no_context():
    out = extractive_answer("random query", {"nodes": [], "edges": [], "kept": []})
    assert "No relevant context" in out["answer"]


# ---------------------------------------------------------------------------
# auto mode — falls back when Ollama is down
# ---------------------------------------------------------------------------

def test_auto_falls_back_when_ollama_down(monkeypatch):
    monkeypatch.setattr("graphrag.answer_generator.ollama_available",
                        lambda: False)
    out = generate_answer("Does claim CLM-0003 have a fraud flag?", PRUNED,
                          mode="auto")
    assert out["mode"] == "extractive"  # graceful fallback


def test_auto_falls_back_on_llm_error(monkeypatch):
    monkeypatch.setattr("graphrag.answer_generator.ollama_available",
                        lambda: True)
    monkeypatch.setattr("graphrag.answer_generator.httpx.post",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    out = generate_answer("Does claim CLM-0003 have a fraud flag?", PRUNED,
                          mode="auto")
    assert out["mode"] == "extractive"


def test_auto_falls_back_on_malformed_reply(monkeypatch):
    monkeypatch.setattr("graphrag.answer_generator.ollama_available",
                        lambda: True)
    monkeypatch.setattr("graphrag.answer_generator.httpx.post",
                        lambda *a, **k: _FakeResponse({"response": ""}))
    out = generate_answer("any query", PRUNED, mode="auto")
    assert out["mode"] == "extractive"  # empty reply -> fallback


# ---------------------------------------------------------------------------
# llm mode — mocked success + required failure
# ---------------------------------------------------------------------------

def test_llm_mode_uses_ollama(monkeypatch):
    monkeypatch.setattr("graphrag.answer_generator.ollama_available",
                        lambda: True)
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["model"] = json["model"]
        captured["prompt"] = json["prompt"]
        captured["stream"] = json["stream"]
        captured["options"] = json["options"]
        return _FakeResponse({"response": "Yes — FRD-CLM-0003 is flagged (MEDIUM)."})

    monkeypatch.setattr("graphrag.answer_generator.httpx.post", fake_post)
    out = generate_answer("Does claim CLM-0003 have a fraud flag?", PRUNED,
                          mode="llm")
    assert out["mode"] == "llm"
    assert out["model"] == settings.ANSWER_MODEL
    assert "FRD-CLM-0003" in out["answer"]
    # prompt must contain the question, the fenced context, and the rules
    assert "Does claim CLM-0003 have a fraud flag?" in captured["prompt"]
    assert "FRD-CLM-0003" in captured["prompt"]
    assert "BEGIN CONTEXT" in captured["prompt"] and "END CONTEXT" in captured["prompt"]
    assert "11434" in captured["url"]
    # request payload contract: non-streaming + capped generation
    assert captured["stream"] is False
    assert captured["options"]["num_predict"] > 0


def test_llm_payload_contract(monkeypatch):
    """POST payload matches the Ollama /api/generate contract."""
    monkeypatch.setattr("graphrag.answer_generator.ollama_available", lambda: True)
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(json)
        return _FakeResponse({"response": "ok"})

    monkeypatch.setattr("graphrag.answer_generator.httpx.post", fake_post)
    generate_answer("q", PRUNED, mode="llm")
    assert captured["model"] == settings.ANSWER_MODEL
    assert captured["stream"] is False
    assert captured["options"]["temperature"] == 0.2
    assert captured["options"]["num_predict"] == 512


def test_auto_falls_back_on_http_500(monkeypatch):
    """A non-200 reply (raise_for_status) must trigger the extractive fallback."""
    monkeypatch.setattr("graphrag.answer_generator.ollama_available", lambda: True)
    monkeypatch.setattr("graphrag.answer_generator.httpx.post",
                        lambda *a, **k: _FakeResponse({"error": "boom"}, status=500))
    out = generate_answer("Does claim CLM-0003 have a fraud flag?", PRUNED,
                          mode="auto")
    assert out["mode"] == "extractive"
    assert "fallback_reason" in out
    assert "FRD-CLM-0003" in out["answer"]


def test_prompt_survives_braces_in_context(monkeypatch):
    """Graph text containing {} must not crash prompt rendering (.format bug)."""
    monkeypatch.setattr("graphrag.answer_generator.ollama_available", lambda: True)
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["prompt"] = json["prompt"]
        return _FakeResponse({"response": "ok"})

    monkeypatch.setattr("graphrag.answer_generator.httpx.post", fake_post)
    braced = {**PRUNED, "text": "[Claim] CLM-0003 description={json: {nested: true}}"}
    out = generate_answer("q with {braces}", braced, mode="llm")
    assert out["mode"] == "llm"
    assert "{json: {nested: true}}" in captured["prompt"]  # braces preserved


def test_llm_mode_requires_ollama(monkeypatch):
    monkeypatch.setattr("graphrag.answer_generator.ollama_available",
                        lambda: False)
    with pytest.raises(RuntimeError, match="requires Ollama"):
        generate_answer("q", PRUNED, mode="llm")


def test_llm_mode_model_not_found_raises_actionable(monkeypatch):
    """404 'model not found' must raise a message telling the user to pull it."""
    monkeypatch.setattr("graphrag.answer_generator.ollama_available", lambda: True)
    monkeypatch.setattr("graphrag.answer_generator.httpx.post",
                        lambda *a, **k: _FakeResponse(
                            {"error": f"model '{settings.ANSWER_MODEL}' not found"},
                            status=404))
    with pytest.raises(ModelNotFoundError, match=settings.ANSWER_MODEL):
        generate_answer("q", PRUNED, mode="llm")


def test_auto_model_not_found_falls_back_with_reason(monkeypatch):
    """In auto mode, a missing model degrades to extractive WITH a reason."""
    monkeypatch.setattr("graphrag.answer_generator.ollama_available", lambda: True)
    monkeypatch.setattr("graphrag.answer_generator.httpx.post",
                        lambda *a, **k: _FakeResponse(
                            {"error": f"model '{settings.ANSWER_MODEL}' not found"},
                            status=404))
    out = generate_answer("Does claim CLM-0003 have a fraud flag?", PRUNED,
                          mode="auto")
    assert out["mode"] == "extractive"
    assert "ollama pull" in out["fallback_reason"]
    assert "FRD-CLM-0003" in out["answer"]


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        generate_answer("q", PRUNED, mode="bogus")


def test_ollama_available_probe(monkeypatch):
    # reachable server
    monkeypatch.setattr("graphrag.answer_generator.httpx.get",
                        lambda *a, **k: _FakeResponse({}, status=200))
    assert ollama_available() is True
    # unreachable -> probe fails closed
    monkeypatch.setattr("graphrag.answer_generator.httpx.get",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("refused")))
    assert ollama_available() is False


def test_negative_probe_cached(monkeypatch):
    """A failed probe is remembered — repeated checks skip the network call."""
    from graphrag import answer_generator as ag
    ag._PROBE_CACHE.clear()
    calls = {"n": 0}

    def failing_get(*a, **k):
        calls["n"] += 1
        raise OSError("refused")

    monkeypatch.setattr("graphrag.answer_generator.httpx.get", failing_get)
    assert ollama_available() is False
    assert ollama_available() is False   # served from cache
    assert ollama_available() is False
    assert calls["n"] == 1, "probe must not repeat while negative cache is fresh"
    ag._PROBE_CACHE.clear()
