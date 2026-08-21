"""Multi-provider LLM layer (v2 — WS-A).

One interface, many backends:

* ``OllamaProvider``            — local /api/generate (dev, air-gapped)
* ``OpenAICompatProvider``      — any OpenAI-compatible Chat Completions API
                                  (Azure OpenAI, vLLM, Together, Groq, Ollama's
                                  own /v1 compat endpoint, …)

``get_provider()`` resolves the configured provider (``settings.LLM_PROVIDER``)
with a fallback chain (``auto`` → OpenAI-compatible when configured, else
Ollama, else None) and returns ``LLMResult`` objects that carry token usage and
estimated cost — the inputs for the v2 cost dashboard and audit trail.
"""

from __future__ import annotations

from graphrag.llm.base import LLMResult, ModelNotFoundError, ProviderError
from graphrag.llm.factory import configured_providers, get_provider
from graphrag.llm.ollama import OllamaProvider
from graphrag.llm.openai_compat import OpenAICompatProvider

__all__ = [
    "LLMResult",
    "ProviderError",
    "ModelNotFoundError",
    "OllamaProvider",
    "OpenAICompatProvider",
    "get_provider",
    "configured_providers",
]
