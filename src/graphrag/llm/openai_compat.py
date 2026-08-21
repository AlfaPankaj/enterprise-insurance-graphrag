"""OpenAI-compatible Chat Completions backend (v2).

One implementation covers every provider that speaks ``POST /chat/completions``:

* OpenAI / Azure OpenAI (``OPENAI_API_VERSION`` switches Azure URL/header shape)
* self-hosted vLLM / TGI / Ollama's ``/v1`` compatibility layer
* Together, Groq, Fireworks, OpenRouter, …

Token usage and cost come from the provider's ``usage`` block when present
(cost = input/output tokens × configured per-1k prices, see
``graphrag.llm.pricing``); when absent, token accounting falls back to the
tiktoken proxy like v1.
"""

from __future__ import annotations

import json
import logging
import time

import httpx

from graphrag.config import settings
from graphrag.llm.base import LLMResult, ModelNotFoundError, ProviderError
from graphrag.llm.pricing import estimate_cost

logger = logging.getLogger("graphrag.llm.openai")


def _chat_url(base: str) -> str:
    """Azure deployments embed the model in the URL path; OpenAI-compatible
    services use a fixed /chat/completions."""
    base = base.rstrip("/")
    if settings.OPENAI_API_VERSION:
        # Azure: {base}/chat/completions?api-version=… (base already holds the deployment)
        return base
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


class OpenAICompatProvider:
    """Chat-Completions backend for OpenAI and compatible gateways."""

    name = "openai"

    _UNSET = object()  # sentinel: "use live settings"

    def __init__(self, base_url=_UNSET, api_key=_UNSET, model=_UNSET):
        self._base_url = base_url
        self._api_key = api_key
        self._model = model

    @property
    def base_url(self) -> str:
        return settings.OPENAI_BASE_URL if self._base_url is self._UNSET else (self._base_url or "")

    @property
    def api_key(self) -> str:
        return settings.OPENAI_API_KEY if self._api_key is self._UNSET else (self._api_key or "")

    @property
    def model(self) -> str:
        return settings.OPENAI_MODEL if self._model is self._UNSET else (self._model or "")

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def available(self) -> bool:
        if not self.configured:
            return False
        if not self.api_key and not settings.OPENAI_API_VERSION:
            # keyless OpenAI-compatible endpoints (local vLLM/Ollama /v1) are
            # still usable; a real probe requires a call we do not want to pay
            # for — treat configured + keyless as available and let the call
            # surface auth errors. A configured Azure endpoint without a key
            # can never work, so report unavailable.
            if "openai.azure.com" in self.base_url or ".azure." in self.base_url:
                return False
            return True
        return True

    def generate(self, prompt: str, *, model: str | None = None,
                 max_tokens: int = 512, temperature: float = 0.2,
                 json_mode: bool = False) -> LLMResult:
        if not self.configured:
            raise ProviderError(
                "OpenAI-compatible provider is not configured "
                "(set OPENAI_BASE_URL / OPENAI_MODEL)"
            )
        model = model or self.model
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        params = {"api-version": settings.OPENAI_API_VERSION} if settings.OPENAI_API_VERSION else None
        payload: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        try:
            response = httpx.post(
                _chat_url(self.base_url), json=payload, headers=headers,
                params=params, timeout=settings.LLM_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"OpenAI-compatible endpoint unreachable ({self.base_url}): {exc}"
            ) from exc

        if response.status_code == 404:
            raise ModelNotFoundError(
                f"model '{model}' not found on {self.base_url} — check OPENAI_MODEL "
                f"(and the deployment name for Azure)"
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"chat completions failed ({response.status_code}): "
                f"{response.text[:300]}"
            )
        data = response.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"unexpected chat-completions payload: {exc}") from exc
        text = text.strip()
        if not text:
            raise ProviderError("provider returned an empty answer")

        usage = data.get("usage") or {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        cost = estimate_cost(model, input_tokens, output_tokens)
        return LLMResult(
            text=text,
            provider=self.name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            raw=data,
        )

    def stream(self, prompt: str, *, model: str | None = None,
               max_tokens: int = 512, temperature: float = 0.2,
               json_mode: bool = False):
        """Yield ``{"type": "delta", "text"}`` events, then ``{"type": "done", ...}``.

        SSE parsing: every ``data: {...}`` line carries a delta chunk; the
        final event carries the assembled text, model, usage and cost
        (``[DONE]`` terminates the stream).
        """
        if not self.configured:
            raise ProviderError(
                "OpenAI-compatible provider is not configured "
                "(set OPENAI_BASE_URL / OPENAI_MODEL)"
            )
        model = model or self.model
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        params = {"api-version": settings.OPENAI_API_VERSION} if settings.OPENAI_API_VERSION else None
        payload: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        timeout = httpx.Timeout(settings.LLM_TIMEOUT_S, read=settings.LLM_TIMEOUT_S)
        try:
            with httpx.stream("POST", _chat_url(self.base_url), json=payload,
                              headers=headers, params=params, timeout=timeout) as response:
                if response.status_code == 404:
                    raise ModelNotFoundError(
                        f"model '{model}' not found on {self.base_url} — check "
                        f"OPENAI_MODEL (and the deployment name for Azure)"
                    )
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", errors="replace")[:300]
                    raise ProviderError(
                        f"chat completions failed ({response.status_code}): {body}"
                    )
                full: list[str] = []
                input_tokens = output_tokens = None
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_line = line[len("data:"):].strip()
                    if data_line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_line)
                    except ValueError:
                        continue
                    choices = chunk.get("choices") or []
                    if choices:
                        content = ((choices[0].get("delta") or {}).get("content")) or ""
                        if content:
                            full.append(content)
                            yield {"type": "delta", "text": content}
                    usage = chunk.get("usage")
                    if usage:
                        input_tokens = usage.get("prompt_tokens")
                        output_tokens = usage.get("completion_tokens")
                if not full:
                    raise ProviderError("provider returned an empty stream")
                yield {
                    "type": "done",
                    "text": "".join(full),
                    "provider": self.name,
                    "model": model,
                    "usage": {"input_tokens": input_tokens,
                              "output_tokens": output_tokens},
                    "cost_usd": estimate_cost(model, input_tokens, output_tokens),
                }
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"OpenAI-compatible endpoint unreachable ({self.base_url}): {exc}"
            ) from exc
