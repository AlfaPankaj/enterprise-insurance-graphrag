"""Answer generation — Ollama LLM with a deterministic extractive fallback.

``generate_answer(query, pruned, subgraph, mode)`` produces the final natural-
language answer for a query:

  * **extractive** (default) — rule-based, deterministic, zero-dependency
    answers assembled from the pruned context (fast, reproducible, and the
    fallback for everything below).
  * **auto** — try the Ollama LLM first; on *any* failure (server down, model
    missing, timeout, malformed reply) fall back to extractive.
  * **llm** — require Ollama; raise if the LLM path fails.

The LLM is a prompt-only call to Ollama's ``/api/generate`` (same client
pattern as ``entity_extractor``). It never sees the whole graph — only the
token-optimized pruned context — so the cost story stays intact: the
re-ranker/pruner already cut the context, the LLM just reads what survived.

Both paths return ``{"answer": str, "mode": "extractive"|"llm",
"model": str|None, "llm_ms": int|None}``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx

from graphrag.config import settings
from graphrag.llm import ModelNotFoundError as ProviderModelNotFoundError
from graphrag.llm.base import ProviderError
from graphrag.llm.factory import get_provider
from graphrag.llm.openai_compat import OpenAICompatProvider

logger = logging.getLogger("graphrag.answer_generator")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_FILE = PROJECT_ROOT / "prompts" / "answer_prompts.txt"

VALID_MODES = ("extractive", "auto", "llm")

# Negative-probe cache: when Ollama is down, re-probing on every query adds
# seconds of latency to the auto/fallback path. Remember a failed probe for a
# short window instead (a fresh success probe still happens after the TTL).
_PROBE_TTL_S = 15
_PROBE_CACHE: dict[str, tuple[bool, float]] = {}

# The prompt uses {query}/{context} so graph-derived text containing braces
# cannot crash .format()/string interpolation (real claim text can hold JSON).
# The context is swapped via a sentinel so a query containing "{context}"
# cannot clobber the insertion.
_QUERY_TOKEN = "{query}"
_CONTEXT_TOKEN = "{context}"
_CONTEXT_SENTINEL = "\x00GRAPH_CONTEXT\x00"


# ---------------------------------------------------------------------------
# deterministic extractive answers (the fallback)
# ---------------------------------------------------------------------------

def extractive_answer(query: str, pruned: dict) -> dict:
    """Rule-based answer from the pruned context (no LLM required)."""
    nodes = pruned["nodes"]
    by_id = {n["id"]: n for n in nodes}
    lower = query.lower()

    # fraud questions -> list fraud flags present in the pruned context
    if "fraud" in lower:
        flags = [n for n in nodes if n["label"] == "FraudFlag"]
        if flags:
            claim_ids = sorted({e["source"] for e in pruned["edges"]
                                if e["type"] == "FRAUD_DETECTED"})
            summary = "; ".join(
                f"{f['id']} (severity={f['props'].get('severity')}, "
                f"reason={f['props'].get('reason')})" for f in flags
            )
            return {"answer": f"Yes — {len(flags)} fraud flag(s) found for "
                              f"{'claims ' + ', '.join(claim_ids) if claim_ids else 'the claim(s)'}: {summary}",
                    "mode": "extractive", "model": None, "llm_ms": None}

    # investigator questions -> who investigates the mentioned claim
    if any(w in lower for w in ("investigat", "who handles", "assigned")):
        invs = [n for n in nodes if n["label"] == "Investigator"]
        if invs:
            return {"answer": "Assigned investigator(s): " +
                              ", ".join(f"{i['id']} ({i['props'].get('name')}, "
                                        f"{i['props'].get('role')})" for i in invs),
                    "mode": "extractive", "model": None, "llm_ms": None}

    # coverage questions -> list coverage ids/limits
    if "coverage" in lower or "cover" in lower:
        covs = [n for n in nodes if n["label"] == "Coverage"]
        if covs:
            return {"answer": ", ".join(
                f"{c['id']} ({c['props'].get('category')}, limit=${c['props'].get('limit'):,.0f})"
                if isinstance(c['props'].get('limit'), (int, float)) else c["id"]
                for c in covs) or "No coverages found",
                "mode": "extractive", "model": None, "llm_ms": None}

    # fallback: the top-ranked node, plus the number of context nodes
    top = by_id.get(pruned["kept"][0]) if pruned["kept"] else None
    if top:
        return {"answer": f"Most relevant context: {top['id']} "
                          f"({top['label']}) — {len(nodes)} nodes retained.",
                "mode": "extractive", "model": None, "llm_ms": None}
    return {"answer": "No relevant context found for this query.",
            "mode": "extractive", "model": None, "llm_ms": None}


# ---------------------------------------------------------------------------
# LLM path (Ollama)
# ---------------------------------------------------------------------------

def ollama_available() -> bool:
    """Probe Ollama's /api/tags; negative results are cached for a short TTL.

    ``entity_extractor`` has an equivalent raw probe for the ingestion path;
    this one adds a failure cache because it sits on the latency-critical
    query path — every auto-mode query would otherwise pay the probe cost.
    """
    url = settings.LLAMA_API_URL
    now = time.monotonic()
    cached = _PROBE_CACHE.get(url)
    if cached and not cached[0] and now - cached[1] < _PROBE_TTL_S:
        return False  # still in the negative-cache window
    try:
        ok = httpx.get(f"{url}/api/tags", timeout=1).status_code == 200
    except Exception:
        ok = False
    _PROBE_CACHE[url] = (ok, now)
    return ok


def _load_prompt() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def _render_prompt(query: str, context: str) -> str:
    """Fill the prompt template — brace-safe (graph text may contain {})."""
    template = _load_prompt()
    template = template.replace(_CONTEXT_TOKEN, _CONTEXT_SENTINEL)
    template = template.replace(_QUERY_TOKEN, query)
    return template.replace(_CONTEXT_SENTINEL, context)


class ModelNotFoundError(RuntimeError):
    """The configured Ollama model is not installed on the server."""


def _raise_actionable_ollama_error(response: httpx.Response, model: str) -> None:
    """Turn a bare httpx error into an actionable one.

    Ollama answers ``404 {\"error\": \"model '<name>' not found\"}`` when the
    model was never pulled — a raw HTTPStatusError tells the user nothing.
    Any other status propagates unchanged.
    """
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


def _generate_with_llm(query: str, pruned: dict, timeout: float = 90.0) -> dict:
    """Call Ollama /api/generate with the pruned context; raise on failure."""
    model = settings.ANSWER_MODEL or settings.LLAMA_MODEL
    prompt = _render_prompt(query, pruned["text"])
    start = time.perf_counter()
    response = httpx.post(
        f"{settings.LLAMA_API_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": settings.ANSWER_MAX_TOKENS},
        },
        timeout=timeout,
    )
    _raise_actionable_ollama_error(response, model)
    payload = response.json()
    if not payload.get("response", "").strip():
        raise ValueError("Ollama returned an empty answer")
    return {
        "answer": payload["response"].strip(),
        "mode": "llm",
        "model": model,
        "llm_ms": round((time.perf_counter() - start) * 1000),
    }


# ---------------------------------------------------------------------------
# OpenAI-compatible path (v2 multi-provider layer)
# ---------------------------------------------------------------------------

def _generate_with_openai(query: str, pruned: dict) -> dict:
    """Answer via the OpenAI-compatible Chat Completions provider.

    Raises on failure (the caller decides the fallback policy). Retries
    transient provider errors up to ``settings.LLM_MAX_RETRIES``.
    """
    provider = OpenAICompatProvider()
    prompt = _render_prompt(query, pruned["text"])
    last_exc: Exception | None = None
    for attempt in range(settings.LLM_MAX_RETRIES + 1):
        try:
            result = provider.generate(
                prompt,
                model=settings.ANSWER_MODEL or settings.OPENAI_MODEL,
                max_tokens=settings.ANSWER_MAX_TOKENS,
                temperature=0.2,
            )
            return {
                "answer": result.text,
                "mode": "llm",
                "model": result.model,
                "llm_ms": result.latency_ms,
                "provider": result.provider,
                "usage": {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                },
                "cost_usd": result.cost_usd,
            }
        except (ProviderError, ProviderModelNotFoundError) as exc:
            last_exc = exc
            if attempt >= settings.LLM_MAX_RETRIES:
                break
            logger.warning("provider retry %d/%d after: %s", attempt + 1,
                           settings.LLM_MAX_RETRIES, exc)
    assert last_exc is not None
    raise last_exc


def _openai_answer_path() -> bool:
    """True when the OpenAI-compatible provider should answer this query.

    ``LLM_PROVIDER=openai`` forces it; ``auto`` prefers it only when
    configured (a configured-but-dead gateway falls through to Ollama —
    the probe is cached, so the dead gateway costs one failed probe per TTL).
    """
    if settings.LLM_PROVIDER == "openai":
        return True
    if settings.LLM_PROVIDER == "auto" and settings.OPENAI_BASE_URL:
        provider = get_provider("auto")
        return provider is not None and provider.name == "openai"
    return False


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def generate_answer(query: str, pruned: dict, mode: str | None = None) -> dict:
    """Answer the query — LLM when requested/available, extractive otherwise.

    ``mode`` in {"extractive", "auto", "llm"}; defaults to
    ``settings.ANSWER_MODE``. The LLM path resolves through the v2
    multi-provider layer: OpenAI-compatible gateway first (when configured),
    Ollama second, deterministic extractive last. ``auto`` attaches
    ``fallback_reason`` to the result when it degraded to extractive, so
    callers can explain the fallback.
    """
    mode = mode or settings.ANSWER_MODE
    if mode not in VALID_MODES:
        raise ValueError(f"unknown answer mode: {mode!r} (expected one of {VALID_MODES})")

    if mode == "llm":
        # hard requirement: a provider must be reachable AND reply — any
        # failure propagates to the caller (the pipeline/API surface reports it)
        if _openai_answer_path():
            return _generate_with_openai(query, pruned)
        if not ollama_available():
            raise RuntimeError(
                f"answer mode 'llm' requires Ollama at {settings.LLAMA_API_URL}"
            )
        return _generate_with_llm(query, pruned)

    if mode == "auto":
        # v2 provider chain: OpenAI-compatible (configured) → Ollama → extractive
        if _openai_answer_path():
            try:
                return _generate_with_openai(query, pruned)
            except Exception as exc:  # noqa: BLE001 - auto must never crash the query
                logger.warning("OpenAI-compatible answer unavailable (%s), "
                               "trying Ollama", exc)
        if ollama_available():
            try:
                return _generate_with_llm(query, pruned)
            except Exception as exc:  # noqa: BLE001 - auto must never crash the query
                # auto -> fall back to the deterministic extractor, but say why so
                # the UI/API can show "why is this extractive?" instead of guessing
                logger.warning("LLM answer unavailable, using extractive: %s", exc)
                out = extractive_answer(query, pruned)
                out["fallback_reason"] = str(exc)[:200]
                return out

    return extractive_answer(query, pruned)
