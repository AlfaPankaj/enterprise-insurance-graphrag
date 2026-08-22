"""v2 embedding provider tests (hash fallback, provider resolution, mocks)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import math

import pytest

from graphrag.config import settings
from graphrag.embeddings import (EmbeddingError, HashEmbedder, cosine,
                                 embed_texts, get_embedder)


# ---------------------------------------------------------------------------
# hash embedder (zero-dependency backbone)
# ---------------------------------------------------------------------------

def test_hash_embedder_deterministic_and_unit():
    e = HashEmbedder()
    a = e.embed(["water damage claim"])[0]
    b = e.embed(["water damage claim"])[0]
    assert a == b                          # deterministic
    assert len(a) == 256
    assert math.isclose(sum(x * x for x in a), 1.0, abs_tol=1e-6)  # unit


def test_hash_embedder_related_texts_similar():
    e = HashEmbedder()
    v1 = e.embed(["claim CLM-0003 water damage"])[0]
    v2 = e.embed(["claim CLM-0003 water damage"])[0]
    v3 = e.embed(["policy endorsement premium adjustment"])[0]
    assert cosine(v1, v2) == pytest.approx(1.0)
    assert cosine(v1, v3) < cosine(v1, v2)


def test_hash_embedder_handles_empty_and_batches():
    e = HashEmbedder()
    vecs = e.embed(["", "a b c", "x y z"])
    assert len(vecs) == 3
    assert all(len(v) == 256 for v in vecs)
    assert math.isclose(sum(x * x for x in vecs[0]), 1.0, abs_tol=1e-6)


def test_cosine_basics():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([1.0, 0.0], [0.0, 0.0]) == pytest.approx(0.0)  # no NaN


# ---------------------------------------------------------------------------
# provider resolution
# ---------------------------------------------------------------------------

def test_get_embedder_hash_by_default(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "auto")
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "")
    monkeypatch.setattr("graphrag.embeddings._ollama_emb.available",
                        lambda: False)
    assert get_embedder("auto").name == "hash"


def test_get_embedder_openai_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1")
    assert get_embedder("auto").name == "openai"
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "")


def test_get_embedder_openai_mode_requires_config(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "")
    with pytest.raises(EmbeddingError, match="OPENAI_BASE_URL"):
        get_embedder("openai")


def test_get_embedder_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown embedding provider"):
        get_embedder("bogus")


def test_embed_texts_falls_back_to_hash(monkeypatch):
    from graphrag import embeddings as emb

    def broken(texts):
        raise EmbeddingError("down")

    monkeypatch.setattr(emb._openai_emb, "embed", broken)
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1")
    vecs = embed_texts(["hello world"], mode="auto")
    assert len(vecs) == 1 and len(vecs[0]) == 256   # hash fallback served
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "")


# ---------------------------------------------------------------------------
# mocked HTTP backends
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    @property
    def text(self):
        return str(self._payload)


def test_openai_embed_contract(monkeypatch):
    from graphrag.embeddings import OpenAICompatEmbedder

    captured = {}

    def fake_post(url, json=None, headers=None, params=None, timeout=None):
        captured.update(url=url, json=json, headers=headers)
        return _FakeResponse({"data": [{"embedding": [1.0, 0.0]},
                                       {"embedding": [0.0, 1.0]}]})

    monkeypatch.setattr("graphrag.embeddings.httpx.post", fake_post)
    emb = OpenAICompatEmbedder(base_url="https://api.openai.com/v1",
                               api_key="sk", model="text-embedding-3-small")
    vecs = emb.embed(["a", "b"])
    assert len(vecs) == 2
    assert captured["url"] == "https://api.openai.com/v1/embeddings"
    assert captured["json"]["model"] == "text-embedding-3-small"
    assert captured["headers"]["Authorization"] == "Bearer sk"
    assert math.isclose(vecs[0][0], 1.0, abs_tol=1e-6)  # normalized unit


def test_openai_embed_mismatched_count_raises(monkeypatch):
    from graphrag.embeddings import OpenAICompatEmbedder

    monkeypatch.setattr("graphrag.embeddings.httpx.post",
                        lambda *a, **k: _FakeResponse({"data": [{"embedding": [1.0]}]}))
    with pytest.raises(EmbeddingError, match="mismatched"):
        OpenAICompatEmbedder(base_url="https://x", api_key="k").embed(["a", "b"])


def test_ollama_embed_contract(monkeypatch):
    from graphrag.embeddings import OllamaEmbedder

    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(url=url, json=json)
        return _FakeResponse({"embedding": [1.0, 0.0, 0.0]})

    monkeypatch.setattr("graphrag.embeddings.httpx.post", fake_post)
    vec = OllamaEmbedder(base_url="http://localhost:11434",
                         model="nomic-embed-text").embed(["text"])[0]
    assert captured["url"].endswith("/api/embeddings")
    assert captured["json"]["model"] == "nomic-embed-text"
    assert len(vec) == 3
