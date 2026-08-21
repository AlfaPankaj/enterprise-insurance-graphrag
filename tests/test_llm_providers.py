"""v2 multi-provider LLM layer tests (mocked HTTP — no servers needed)."""

from __future__ import annotations

import json

import pytest

from graphrag.config import settings
from graphrag.llm import ModelNotFoundError, ProviderError
from graphrag.llm.factory import clear_probe_cache, get_provider
from graphrag.llm.ollama import OllamaProvider
from graphrag.llm.openai_compat import OpenAICompatProvider
from graphrag.llm.pricing import estimate_cost


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200, url: str = "http://x"):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            from httpx import HTTPStatusError, Request, Response
            req = Request("POST", "http://x")
            resp = Response(self.status_code, request=req, json=self._payload)
            raise HTTPStatusError(str(self.status_code), request=req, response=resp)

    @property
    def text(self):
        return json.dumps(self._payload)


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------

def test_ollama_generate_contract(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _FakeResponse({"response": "Yes, flagged.",
                              "prompt_eval_count": 42, "eval_count": 7})

    monkeypatch.setattr("graphrag.llm.ollama.httpx.post", fake_post)
    result = OllamaProvider("http://localhost:11434").generate(
        "q", model="llama3.2:3b", max_tokens=512, temperature=0.2)
    assert result.text == "Yes, flagged."
    assert result.provider == "ollama"
    assert result.input_tokens == 42 and result.output_tokens == 7
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["options"]["num_predict"] == 512
    assert "11434" in captured["url"]


def test_ollama_model_missing_raises_actionable(monkeypatch):
    monkeypatch.setattr(
        "graphrag.llm.ollama.httpx.post",
        lambda *a, **k: _FakeResponse({"error": "model 'nope' not found"}, status=404),
    )
    with pytest.raises(ModelNotFoundError, match="ollama pull nope"):
        OllamaProvider().generate("q", model="nope")


def test_ollama_empty_answer_is_provider_error(monkeypatch):
    monkeypatch.setattr("graphrag.llm.ollama.httpx.post",
                        lambda *a, **k: _FakeResponse({"response": ""}))
    with pytest.raises(ProviderError, match="empty"):
        OllamaProvider().generate("q")


# ---------------------------------------------------------------------------
# OpenAICompatProvider
# ---------------------------------------------------------------------------

def test_openai_generate_maps_usage_and_cost(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, params=None, timeout=None):
        captured.update({"url": url, "payload": json, "headers": headers,
                         "params": params})
        return _FakeResponse({
            "choices": [{"message": {"content": "The answer."}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        })

    monkeypatch.setattr("graphrag.llm.openai_compat.httpx.post", fake_post)
    provider = OpenAICompatProvider(base_url="https://api.openai.com/v1",
                                    api_key="sk-test", model="gpt-4o-mini")
    result = provider.generate("prompt")
    assert result.text == "The answer."
    assert result.provider == "openai"
    assert result.input_tokens == 100 and result.output_tokens == 20
    # gpt-4o-mini list price: 0.00015 in / 0.0006 out per 1k
    assert result.cost_usd == pytest.approx(100 / 1000 * 0.00015
                                            + 20 / 1000 * 0.0006)
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["payload"]["messages"][0]["content"] == "prompt"


def test_openai_azure_payload(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, params=None, timeout=None):
        captured.update({"url": url, "payload": json, "params": params})
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("graphrag.llm.openai_compat.httpx.post", fake_post)
    monkeypatch.setattr(settings, "OPENAI_API_VERSION", "2024-06-01")
    try:
        provider = OpenAICompatProvider(
            base_url="https://res.openai.azure.com/openai/deployments/gpt4o",
            api_key="az-key", model="gpt4o")
        result = provider.generate("p", json_mode=True)
    finally:
        monkeypatch.setattr(settings, "OPENAI_API_VERSION", "")
    assert result.text == "ok"
    # Azure: deployment URL kept as-is, api-version as a query param
    assert "openai.azure.com/openai/deployments/gpt4o" in captured["url"]
    assert captured["params"] == {"api-version": "2024-06-01"}
    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_openai_model_not_found(monkeypatch):
    monkeypatch.setattr("graphrag.llm.openai_compat.httpx.post",
                        lambda *a, **k: _FakeResponse({"error": "not found"},
                                                      status=404))
    with pytest.raises(ModelNotFoundError, match="gpt-4o-mini"):
        OpenAICompatProvider(base_url="https://x", api_key="k").generate("p")


def test_openai_unconfigured_raises():
    with pytest.raises(ProviderError, match="not configured"):
        OpenAICompatProvider(base_url="", api_key="").generate("p")


def test_estimate_cost_unknown_model_uses_env(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1K_INPUT", 0.01)
    monkeypatch.setattr(settings, "LLM_PRICE_PER_1K_OUTPUT", 0.03)
    cost = estimate_cost("totally-unknown-model", 1000, 500)
    assert cost == pytest.approx(0.01 + 0.015)


# ---------------------------------------------------------------------------
# factory — fallback chain
# ---------------------------------------------------------------------------

def test_factory_auto_prefers_openai_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test")
    clear_probe_cache()
    provider = get_provider("auto")
    assert provider is not None and provider.name == "openai"


def test_factory_auto_falls_back_to_ollama(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "")
    monkeypatch.setattr("graphrag.llm.factory._probe",
                        lambda p: p.name == "ollama")
    clear_probe_cache()
    provider = get_provider("auto")
    assert provider is not None and provider.name == "ollama"


def test_factory_auto_none_when_nothing_up(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "")
    monkeypatch.setattr("graphrag.llm.factory._probe", lambda p: False)
    clear_probe_cache()
    assert get_provider("auto") is None


def test_factory_openai_mode_requires_config(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "")
    with pytest.raises(RuntimeError, match="OPENAI_BASE_URL"):
        get_provider("openai")


def test_factory_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown"):
        get_provider("bogus")


def test_negative_probe_cache_skips_repeat_probes(monkeypatch):
    from graphrag.llm import factory

    class _Down:
        name = "down-provider"
        calls = 0

        def available(self):
            _Down.calls += 1
            return False

    clear_probe_cache()
    assert factory._probe(_Down()) is False
    assert factory._probe(_Down()) is False   # second call: cached, no network
    assert _Down.calls == 1
    clear_probe_cache()
