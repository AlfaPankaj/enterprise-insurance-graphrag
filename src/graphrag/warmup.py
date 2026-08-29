"""Opt-in query-model warm-up without changing inference semantics."""

from __future__ import annotations

import logging
import threading

import httpx

from graphrag.config import settings
from graphrag.http_client import request_post

logger = logging.getLogger("graphrag.warmup")
_LOCK = threading.Lock()
_WARMED: set[tuple[str, ...]] = set()


def _ollama_signature() -> tuple[str, ...]:
    return (
        "ollama",
        settings.LLAMA_API_URL,
        settings.ANSWER_MODEL or settings.LLAMA_MODEL,
        settings.OLLAMA_KEEP_ALIVE,
    )


def warm_query_models() -> dict[str, str]:
    """Preload the configured Ollama and cross-encoder models once.

    Ollama's empty-prompt generate request is its documented preload path: it
    loads the exact answer model and applies ``keep_alive`` but emits no answer.
    Cross-encoder warm-up only performs the same lazy model load that the first
    rerank would otherwise trigger. Failures are logged and never alter runtime
    provider fallback or ranking behavior.
    """
    if not settings.QUERY_MODEL_WARMUP_ENABLED:
        return {"status": "disabled"}

    report: dict[str, str] = {}
    answer_mode = settings.ANSWER_MODE
    provider_mode = settings.LLM_PROVIDER
    warm_ollama = provider_mode == "ollama" or (
        provider_mode == "auto" and not settings.OPENAI_BASE_URL
    )
    if answer_mode in {"auto", "llm"} and warm_ollama:
        signature = _ollama_signature()
        with _LOCK:
            should_warm = signature not in _WARMED
        if should_warm:
            model = signature[2]
            try:
                response = request_post(
                    f"{settings.LLAMA_API_URL.rstrip('/')}/api/generate",
                    json={
                        "model": model,
                        "prompt": "",
                        "stream": False,
                        "keep_alive": settings.OLLAMA_KEEP_ALIVE,
                    },
                    timeout=settings.LLM_TIMEOUT_S,
                )
                response.raise_for_status()
                with _LOCK:
                    _WARMED.add(signature)
                report["ollama"] = "ready"
            except (httpx.HTTPError, OSError) as exc:
                report["ollama"] = "unavailable"
                logger.info("Ollama warm-up skipped: %s", exc)
        else:
            report["ollama"] = "ready"

    if settings.RERANKER_MODE in {"auto", "cross-encoder"}:
        signature = ("cross-encoder", settings.CROSS_ENCODER_MODEL)
        with _LOCK:
            should_warm = signature not in _WARMED
        if should_warm:
            try:
                from graphrag.reranker import CrossEncoderReranker, make_reranker

                reranker = make_reranker(settings.RERANKER_MODE)
                if isinstance(reranker, CrossEncoderReranker):
                    reranker._load()
                with _LOCK:
                    _WARMED.add(signature)
                report["reranker"] = "ready"
            except Exception as exc:  # noqa: BLE001 - warm-up is best effort
                report["reranker"] = "unavailable"
                logger.info("reranker warm-up skipped: %s", exc)
        else:
            report["reranker"] = "ready"

    return report or {"status": "nothing_to_warm"}


def clear_warmup_state() -> None:
    """Reset process-local warm-up state (tests)."""
    with _LOCK:
        _WARMED.clear()
