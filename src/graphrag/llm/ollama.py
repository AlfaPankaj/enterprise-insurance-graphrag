"""Ollama backend — local /api/generate.

Kept contract-compatible with the v1 call site (answer_generator's mocked
tests POST to ``{LLAMA_API_URL}/api/generate`` with ``{model, prompt, stream:
False, options: {temperature, num_predict}}``), so this provider re-implements
exactly that wire format while returning the richer ``LLMResult``.
"""

from __future__ import annotations

import logging
import time

import httpx

from graphrag.config import settings
from graphrag.llm.base import LLMResult, ModelNotFoundError, ProviderError

logger = logging.getLogger("graphrag.llm.ollama")


def _raise_actionable_error(response: httpx.Response, model: str) -> None:
    """Turn a bare httpx error into an actionable one (404 = model not pulled)."""
    body = ""
    try:
        body = response.json().get("error", "")
    except Exception:
        pass
    if response.status_code == 404 and "not found" in body.lower():
        raise ModelNotFoundError(
            f"Ollama model '{model}' is not installed. Pull it first with: "
            f"ollama pull {model}"
        ) from None
    response.raise_for_status()


class OllamaProvider:
    """Local Ollama server speaking ``/api/generate``."""

    name = "ollama"

    _UNSET = object()

    def __init__(self, base_url=_UNSET):
        self._base_url = base_url

    @property
    def base_url(self) -> str:
        url = settings.LLAMA_API_URL if self._base_url is self._UNSET else (self._base_url or "")
        return url.rstrip("/")

    def available(self) -> bool:
        try:
            return httpx.get(f"{self.base_url}/api/tags", timeout=2).status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, *, model: str | None = None,
                 max_tokens: int = 512, temperature: float = 0.2,
                 json_mode: bool = False) -> LLMResult:
        model = model or settings.ANSWER_MODEL or settings.LLAMA_MODEL
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        start = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate", json=payload,
                timeout=settings.LLM_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama unreachable at {self.base_url}: {exc}") from exc
        _raise_actionable_error(response, model)
        data = response.json()
        text = (data.get("response") or "").strip()
        if not text:
            raise ProviderError("Ollama returned an empty answer")
        return LLMResult(
            text=text,
            provider=self.name,
            model=model,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            raw=data,
        )
