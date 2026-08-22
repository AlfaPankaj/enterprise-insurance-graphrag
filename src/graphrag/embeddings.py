"""Embedding providers (v2 — WS-C hybrid retrieval, G16).

Pluggable text-embedding backends, same philosophy as the LLM layer:

* ``OpenAICompatEmbedder`` — any OpenAI-compatible ``POST /embeddings``
  (OpenAI, Azure OpenAI, vLLM, Together, …)
* ``OllamaEmbedder`` — local ``/api/embeddings`` (needs an embedding model
  pulled, e.g. ``ollama pull nomic-embed-text``)
* ``HashEmbedder`` — **deterministic feature hashing** (zero dependencies,
  no API, no model): 256-dim signed unit vectors over token n-grams. Strong
  enough to power lexical-ish semantic seed fallback and hybrid re-ranking
  in demos/tests; real embeddings take over whenever a provider is up.

``get_embedder(mode)`` resolves ``auto`` → OpenAI-compatible when configured,
else Ollama when probed, else the hash fallback. ``embed(texts)`` always
returns unit vectors (list[list[float]]), so cosine similarity is a dot
product.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading

import httpx

from graphrag.config import settings

logger = logging.getLogger("graphrag.embeddings")

_DIM = 256
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


class EmbeddingError(RuntimeError):
    """An embedding call failed (network/HTTP)."""


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two unit (or arbitrary) vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class HashEmbedder:
    """Deterministic n-gram feature-hashing embedder (zero dependencies)."""

    name = "hash"

    def available(self) -> bool:
        return True

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text.lower()) for text in texts]

    @staticmethod
    def _one(text: str) -> list[float]:
        vec = [0.0] * _DIM
        tokens = _TOKEN_RE.findall(text)
        if not tokens:
            tokens = ["<empty>"]
        for i, tok in enumerate(tokens):
            gram = tok
            h = hashlib.blake2b(gram.encode(), digest_size=8).digest()
            sign = 1.0 if h[0] & 1 else -1.0
            bucket = int.from_bytes(h, "big") % _DIM
            vec[bucket] += sign
            if i + 1 < len(tokens):  # bigram signal
                gram2 = tokens[i] + "_" + tokens[i + 1]
                h2 = hashlib.blake2b(gram2.encode(), digest_size=8).digest()
                sign2 = 1.0 if h2[0] & 1 else -1.0
                bucket2 = int.from_bytes(h2, "big") % _DIM
                vec[bucket2] += sign2
        return _normalize(vec)


class OpenAICompatEmbedder:
    """OpenAI-compatible /embeddings backend."""

    name = "openai"

    _UNSET = object()

    def __init__(self, base_url=_UNSET, api_key=_UNSET, model=_UNSET):
        self._base_url = base_url
        self._api_key = api_key
        self._model = model

    @property
    def base_url(self) -> str:
        return settings.OPENAI_BASE_URL if self._base_url is self._UNSET \
            else (self._base_url or "")

    @property
    def api_key(self) -> str:
        return settings.OPENAI_API_KEY if self._api_key is self._UNSET \
            else (self._api_key or "")

    @property
    def model(self) -> str:
        return settings.EMBEDDING_MODEL if self._model is self._UNSET \
            else (self._model or "")

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def available(self) -> bool:
        return self.configured and bool(self.api_key or "azure" not in self.base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.configured:
            raise EmbeddingError("OpenAI-compatible embeddings not configured "
                                 "(set OPENAI_BASE_URL / EMBEDDING_MODEL)")
        url = self.base_url.rstrip("/")
        if not url.endswith("/embeddings"):
            url = f"{url}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        params = {"api-version": settings.OPENAI_API_VERSION} \
            if settings.OPENAI_API_VERSION else None
        try:
            response = httpx.post(
                url, json={"model": self.model, "input": texts},
                headers=headers, params=params, timeout=settings.LLM_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"embeddings endpoint unreachable: {exc}") from exc
        if response.status_code >= 400:
            raise EmbeddingError(
                f"embeddings failed ({response.status_code}): {response.text[:200]}")
        data = response.json().get("data") or []
        vectors = [d.get("embedding") or [] for d in data]
        if len(vectors) != len(texts):
            raise EmbeddingError("embeddings endpoint returned a mismatched count")
        return [_normalize(v) for v in vectors]


class OllamaEmbedder:
    """Local Ollama /api/embeddings backend."""

    name = "ollama"

    _UNSET = object()

    def __init__(self, base_url=_UNSET, model=_UNSET):
        self._base_url = base_url
        self._model = model

    @property
    def base_url(self) -> str:
        return settings.LLAMA_API_URL if self._base_url is self._UNSET \
            else (self._base_url or "")

    @property
    def model(self) -> str:
        return settings.EMBEDDING_OLLAMA_MODEL if self._model is self._UNSET \
            else (self._model or "")

    def available(self) -> bool:
        try:
            return httpx.get(f"{self.base_url}/api/tags", timeout=2).status_code == 200
        except Exception:
            return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = httpx.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": texts[0] if len(texts) == 1 else texts},
                timeout=settings.LLM_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama embeddings unreachable: {exc}") from exc
        if response.status_code >= 400:
            raise EmbeddingError(
                f"ollama embeddings failed ({response.status_code}): "
                f"{response.text[:200]} — is '{self.model}' pulled?")
        payload = response.json()
        if "embeddings" in payload:  # batched response
            return [_normalize(v) for v in payload["embeddings"]]
        return [_normalize(payload.get("embedding") or [])]


# process-wide singletons (same pattern as the LLM providers)
_openai_emb = OpenAICompatEmbedder()
_ollama_emb = OllamaEmbedder()
_hash_emb = HashEmbedder()


def get_embedder(mode: str | None = None):
    """Resolve the embedding backend; ``auto`` degrades gracefully.

    auto → OpenAI-compatible when configured → Ollama when probed → hash.
    The hash fallback means hybrid retrieval always has embeddings available.
    """
    mode = mode or settings.EMBEDDING_PROVIDER
    if mode == "openai":
        if not _openai_emb.configured:
            raise EmbeddingError("EMBEDDING_PROVIDER=openai but OPENAI_BASE_URL "
                                 "is not set")
        return _openai_emb
    if mode == "ollama":
        return _ollama_emb
    if mode == "hash":
        return _hash_emb
    if mode != "auto":
        raise ValueError(f"unknown embedding provider: {mode!r}")
    if _openai_emb.configured:
        return _openai_emb
    if _ollama_emb.available():
        return _ollama_emb
    return _hash_emb


def embed_texts(texts: list[str], mode: str | None = None) -> list[list[float]]:
    """Embed ``texts`` with the resolved backend (never raises for ``auto``)."""
    embedder = get_embedder(mode)
    try:
        return embedder.embed(texts)
    except EmbeddingError:
        if mode in ("openai", "ollama"):
            raise
        logger.warning("embedding provider %s failed, using hash fallback",
                       embedder.name)
        return _hash_emb.embed(texts)
