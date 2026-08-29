"""Opt-in model warm-up controls preserve selected model semantics."""

from __future__ import annotations

from graphrag.config import settings
from graphrag.warmup import clear_warmup_state, warm_query_models


class _Response:
    def raise_for_status(self):
        return None


def test_ollama_warmup_uses_selected_model_and_keepalive(monkeypatch):
    from graphrag import warmup

    calls: list[dict] = []
    monkeypatch.setattr(settings, "QUERY_MODEL_WARMUP_ENABLED", True)
    monkeypatch.setattr(settings, "ANSWER_MODE", "auto")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "ANSWER_MODEL", "selected-answer-model")
    monkeypatch.setattr(settings, "RERANKER_MODE", "lexical")
    monkeypatch.setattr(settings, "OLLAMA_KEEP_ALIVE", "17m")
    monkeypatch.setattr(
        warmup,
        "request_post",
        lambda _url, **kwargs: (calls.append(kwargs["json"]) or _Response()),
    )
    clear_warmup_state()
    assert warm_query_models()["ollama"] == "ready"
    assert warm_query_models()["ollama"] == "ready"
    assert calls == [{
        "model": "selected-answer-model",
        "prompt": "",
        "stream": False,
        "keep_alive": "17m",
    }]
    clear_warmup_state()


def test_cross_encoder_warmup_loads_cached_reranker(monkeypatch):
    from graphrag import reranker as reranker_module

    class _FakeCrossEncoder:
        loaded = 0

        def _load(self):
            _FakeCrossEncoder.loaded += 1

    instance = _FakeCrossEncoder()
    monkeypatch.setattr(settings, "QUERY_MODEL_WARMUP_ENABLED", True)
    monkeypatch.setattr(settings, "ANSWER_MODE", "extractive")
    monkeypatch.setattr(settings, "RERANKER_MODE", "cross-encoder")
    monkeypatch.setattr(reranker_module, "CrossEncoderReranker", _FakeCrossEncoder)
    monkeypatch.setattr(reranker_module, "make_reranker", lambda _mode: instance)
    clear_warmup_state()
    assert warm_query_models()["reranker"] == "ready"
    assert warm_query_models()["reranker"] == "ready"
    assert _FakeCrossEncoder.loaded == 1
    clear_warmup_state()
