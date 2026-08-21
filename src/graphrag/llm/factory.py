"""Provider resolution + fallback chain (v2).

``get_provider()`` answers "which backend answers the next call?":

* ``settings.LLM_PROVIDER == "openai"``   → OpenAICompatProvider (raise if unconfigured)
* ``settings.LLM_PROVIDER == "ollama"``   → OllamaProvider
* ``settings.LLM_PROVIDER == "auto"``     → OpenAI-compatible when configured
  AND probed-ok, else Ollama when probed-ok, else None (caller falls back to
  the deterministic extractive path)

Failed probes are cached per backend (short TTL) exactly like v1's negative
probe cache, so the latency-critical query path never pays a dead probe twice.
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

_PROBE_TTL_S = 15.0
_PROBE_CACHE: dict[str, tuple[bool, float]] = {}
_LOCK = threading.Lock()

_openai = OpenAICompatProvider()
_ollama = OllamaProvider()


def _probe(provider: LLMProvider) -> bool:
    """``available()`` with a negative-result cache (per backend name)."""
    now = time.monotonic()
    with _LOCK:
        cached = _PROBE_CACHE.get(provider.name)
        if cached and not cached[0] and now - cached[1] < _PROBE_TTL_S:
            return False
    try:
        ok = provider.available()
    except Exception:
        ok = False
    with _LOCK:
        _PROBE_CACHE[provider.name] = (ok, now)
    return ok


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
