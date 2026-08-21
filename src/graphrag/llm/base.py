"""Provider contracts shared by every backend (v2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ProviderError(RuntimeError):
    """A provider call failed (network, HTTP, or a rejected payload)."""


class ModelNotFoundError(ProviderError):
    """The configured model does not exist on the provider.

    Raised with an actionable message (e.g. "pull it first with: ollama pull X").
    """


@dataclass
class LLMResult:
    """One completed generation, provider-agnostic.

    ``usage`` / ``cost_usd`` are None when the provider does not report them —
    callers then fall back to the tiktoken proxy for token accounting.
    """

    text: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class LLMProvider:
    """Interface every backend implements.

    ``generate(prompt, *, model, max_tokens, temperature, json_mode)`` returns
    an ``LLMResult`` or raises ``ProviderError``. ``available()`` is a cheap
    probe used for fallback-chain resolution; negative results may be cached
    by the caller.
    """

    name = "base"

    def available(self) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def generate(self, prompt: str, *, model: str | None = None,
                 max_tokens: int = 512, temperature: float = 0.2,
                 json_mode: bool = False) -> LLMResult:  # pragma: no cover
        raise NotImplementedError
