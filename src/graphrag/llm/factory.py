"""Provider resolution + fallback chain (v2).

``get_provider()`` answers "which backend answers the next call?":

* ``settings.LLM_PROVIDER == "openai"``   → OpenAICompatProvider (raise if unconfigured)
* ``settings.LLM_PROVIDER == "ollama"``   → OllamaProvider
* ``settings.LLM_PROVIDER == "auto"``     → OpenAI-compatible when configured
  AND probed-ok, else Ollama when probed-ok, else None (caller falls back to
  the deterministic extractive path)

Successful and failed probes are cached per backend for a short TTL, so the
latency-critical query path avoids repeated provider health round trips.
"""

from __future__ import annotations

import logging
import threading
import time

from graphrag.config import settings
from graphrag.llm.base import LLMProvider
from graphrag.llm.ollama import OllamaProvider
from graphrag.llm.openai_compat import OpenAICompatProvider

logger = logging.getLogger("graphrag.llm")

_PROBE_CACHE: dict[str, tuple[bool, float]] = {}
_LOCK = threading.Lock()

_openai = OpenAICompatProvider()
_ollama = OllamaProvider()


def _probe(provider: LLMProvider) -> bool:
    """``available()`` with a short success/failure cache per endpoint."""
    now = time.monotonic()
    base_url = str(getattr(provider, "base_url", "") or "").rstrip("/")
    cache_key = f"{provider.name}|{base_url}"
    with _LOCK:
        cached = _PROBE_CACHE.get(cache_key)
        ttl = max(0.0, float(settings.LLM_PROBE_TTL_S))
        if cached and now - cached[1] < ttl:
            return cached[0]
        try:
            ok = provider.available()
        except Exception:
            ok = False
        _PROBE_CACHE[cache_key] = (ok, now)
        return ok


def provider_available(provider: LLMProvider) -> bool:
    """Public cached availability check for ingestion/query integrations."""
    return _probe(provider)


def _openai_configured() -> bool:
    return bool(settings.OPENAI_BASE_URL)


def configured_providers() -> list[str]:
    """Provider names this deployment is configured to use, best-first."""
    out: list[str] = []
    if _openai_configured():
        out.append("openai")
    out.append("ollama")
    return out


def get_provider(mode: str | None = None) -> LLMProvider | None:
    """Resolve the provider for the next call; None = no provider usable.

    ``mode`` overrides ``settings.LLM_PROVIDER`` (same vocabulary as v1's
    answer/extraction modes: "auto" | "ollama" | "openai").
    """
    mode = mode or settings.LLM_PROVIDER
    if mode == "openai":
        if not _openai_configured():
            raise RuntimeError(
                "LLM_PROVIDER=openai but OPENAI_BASE_URL is not set "
                "(see .env.example)"
            )
        return _openai
    if mode == "ollama":
        return _ollama
    if mode != "auto":
        raise ValueError(f"unknown LLM provider mode: {mode!r}")
    # auto: OpenAI-compatible first (configured + probed), then Ollama
    if _openai_configured() and _probe(_openai):
        return _openai
    if _probe(_ollama):
        return _ollama
    return None


def clear_probe_cache() -> None:
    """Reset the probe cache (tests)."""
    with _LOCK:
        _PROBE_CACHE.clear()
