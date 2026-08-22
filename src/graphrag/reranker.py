"""Re-rank retrieved graph nodes by relevance to the query (Shot 2).

Two interchangeable backends with the same ``rank(query, nodes)`` interface:

  * **cross-encoder** — a neural ``CrossEncoder`` (sentence-transformers),
    scoring ``(query, node_text)`` pairs. Real relevance ranking, model
    downloaded on first use.
  * **lexical** — pure-Python Okapi **BM25** over node texts plus an
    entity-id match bonus. Zero-dependency fallback (fast, reproducible
    tests; no model download needed).

``make_reranker(mode)`` resolves ``auto`` → cross-encoder when importable and
loadable, else lexical. Both return ``[(node, score)]`` sorted best-first.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import re
import threading
from collections import Counter

from graphrag.config import settings
from graphrag.graph_retriever import ENTITY_ID_RE, query_tokens, serialize_node

logger = logging.getLogger("graphrag.reranker")

_TOKEN_RE = re.compile(r"[a-zA-Z]+")

# Answer-type hints: query phrases -> node labels. A small lexical prior added
# on top of the neural score (hybrid retrieval) — it guarantees the answer type
# is not starved out by the token budget.
_LABEL_HINTS: list[tuple[tuple[str, ...], str]] = [
    (("coverage", "coverages", "covers"), "Coverage"),
    (("fraud", "flag", "flagged"), "FraudFlag"),
    (("investigat", "handled by", "assigned"), "Investigator"),
    (("endorsement", "endorsements"), "Endorsement"),
    (("policy", "policies"), "Policy"),
    (("claim", "claims"), "Claim"),
    (("policyholder", "holder", "held by"), "Policyholder"),
]
_LABEL_PRIOR = 2.5


def _label_prior_hits(query: str) -> set[str]:
    """Labels the query explicitly asks about (e.g. "coverages" -> Coverage)."""
    q = query.lower()
    hits = set()
    for phrases, label in _LABEL_HINTS:
        if any(p in q for p in phrases):
            hits.add(label)
    return hits


def _node_text(node: dict) -> str:
    return serialize_node(node)


# ---------------------------------------------------------------------------
# BM25 (Okapi) — pure-Python, zero dependencies
# ---------------------------------------------------------------------------

_K1 = 1.5   # term-frequency saturation
_B = 0.75   # length-normalization strength (Lucene defaults)
_ID_BONUS = 5.0  # entity-id match: guarantees an explicitly-named id ranks #1


def _bm25_idf(n_docs: int, df: int) -> float:
    """Robertson-Sparck Jones IDF with +1 smoothing (never negative)."""
    return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


def _prefix_variants(query_tok: str, vocab: set[str]) -> set[str]:
    """Exact term or shared-prefix (len >= 4) variants — covers singular/plural
    and morphological pairs (coverages/coverage, investigates/investigator)
    without a full stemmer."""
    if not vocab:
        return {query_tok}
    variants = {t for t in vocab if t == query_tok
                or (len(query_tok) >= 4 and len(t) >= 4
                    and (t.startswith(query_tok) or query_tok.startswith(t)))}
    return variants or {query_tok}


# ---------------------------------------------------------------------------
# lexical backend (deterministic fallback)
# ---------------------------------------------------------------------------

class LexicalReranker:
    """Okapi BM25 over node texts, plus an entity-id match bonus.

        score = BM25(query, node_text) + 5.0 * (query names this entity id)

    BM25 provides length normalization (a verbose policy node cannot outrank a
    terse claim summary just for being longer) and term-frequency saturation
    (a keyword repeated 10x does not dominate the ranking). The entity-id
    bonus guarantees an explicitly-named policy/claim number jumps to rank 1,
    while BM25 orders the remaining context nodes by term relevance.
    """

    name = "lexical"

    def rank(self, query: str, nodes: list[dict]) -> list[tuple[dict, float]]:
        if not nodes:
            return []
        q_tokens = query_tokens(query)
        q_ids = set(re.findall(ENTITY_ID_RE, query.upper()))

        docs = [_TOKEN_RE.findall(_node_text(n).lower()) for n in nodes]
        n_docs = len(docs)
        avgdl = (sum(len(d) for d in docs) / n_docs) or 1.0  # avoid div0 on empty docs

        vocab = {t for d in docs for t in d}
        df = Counter(t for d in docs for t in set(d))
        variants = {qt: _prefix_variants(qt, vocab) for qt in q_tokens}

        scored: list[tuple[dict, float]] = []
        for node, tokens in zip(nodes, docs):
            tf = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for qt in q_tokens:
                variant_df = max((df.get(t, 0) for t in variants[qt]), default=0)
                idf = _bm25_idf(n_docs, variant_df)
                # one contribution per query token: the strongest variant match
                # (max, not sum — "claim" + "claimant" is one piece of evidence)
                best = max((tf.get(t, 0) for t in variants[qt]), default=0)
                if not best:
                    continue
                denom = best + _K1 * (1 - _B + _B * (doc_len / avgdl))
                score += idf * (best * (_K1 + 1)) / denom
            id_bonus = _ID_BONUS if node["id"].upper() in q_ids else 0.0
            scored.append((node, score + id_bonus))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored


# ---------------------------------------------------------------------------
# cross-encoder backend (neural)
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """sentence-transformers CrossEncoder; model loads lazily on first rank."""

    name = "cross-encoder"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.CROSS_ENCODER_MODEL
        self._model = None

    _load_lock = threading.Lock()  # shared: one load even under concurrent requests

    def _load(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self.model_name)

    def rank(self, query: str, nodes: list[dict]) -> list[tuple[dict, float]]:
        """Neural ranking + answer-type prior; falls back to lexical on failure."""
        try:
            self._load()
            pairs = [(query, _node_text(n)) for n in nodes]
            scores = self._model.predict(pairs)
            label_hits = _label_prior_hits(query)
            scored = [(node, float(s) + (_LABEL_PRIOR if node["label"] in label_hits else 0.0))
                      for node, s in zip(nodes, scores)]
            scored.sort(key=lambda t: t[1], reverse=True)
            return scored
        except Exception:
            logger.warning("cross-encoder unavailable (%s), falling back to lexical",
                           self.model_name, exc_info=True)
            self.name = "lexical"  # permanent fallback for this instance
            return LexicalReranker().rank(query, nodes)


# ---------------------------------------------------------------------------
# hybrid backend (v2 — WS-C): RRF fusion of lexical + semantic + proximity
# ---------------------------------------------------------------------------

class HybridReranker:
    """Reciprocal-Rank-Fusion of three signals over the retrieved subgraph:

      * **lexical** — the zero-dependency BM25 rank (always available)
      * **semantic** — cosine similarity to the query embedding, over a
        revision-cached in-memory vector index of the graph node texts
        (embedding provider: OpenAI-compatible → Ollama → deterministic hash)
      * **proximity** — hop distance from the seed nodes (nodes carry
        ``_dist``, injected by the pipeline)

    Degrades gracefully: without a readable graph revision (no vector index)
    or with no proximity info it simply returns the lexical ranking, so the
    hybrid path can never break a query.
    """

    name = "hybrid"
    _RRF_K = 60.0

    def __init__(self, driver=None):
        self.driver = driver
        self._lexical = LexicalReranker()

    @staticmethod
    def _rrf(ranked_lists: list[list[str]]) -> dict[str, float]:
        fused: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, node_id in enumerate(ranked):
                fused[node_id] = fused.get(node_id, 0.0) + \
                    1.0 / (HybridReranker._RRF_K + rank + 1)
        return fused

    def rank(self, query: str, nodes: list[dict]) -> list[tuple[dict, float]]:
        if not nodes:
            return []
        node_by_id = {n["id"]: n for n in nodes}

        # 1) lexical
        lexical = self._lexical.rank(query, nodes)
        lex_ids = [n["id"] for n, _ in lexical]

        # 2) semantic (optional: vector index over node texts)
        vec_ids: list[str] = []
        store = None
        if self.driver is not None:
            try:
                from graphrag.vector_store import build_vector_store
                store = build_vector_store(self.driver)
            except Exception:  # noqa: BLE001 - hybrid must never break a query
                store = None
        if store is not None and len(store):
            try:
                from graphrag.embeddings import cosine, embed_texts
                texts = [serialize_node(n) for n in nodes]
                query_vec = embed_texts([query])[0]
                scored = sorted(
                    ((node_by_id[n["id"]], cosine(query_vec, vec))
                     for n, vec in zip(nodes, embed_texts(texts))),
                    key=lambda t: t[1], reverse=True,
                )
                vec_ids = [n["id"] for n, _ in scored]
            except Exception:  # noqa: BLE001
                vec_ids = []

        # 3) proximity (hop distance from seeds, injected as _dist)
        prox_ids: list[str] = []
        dist = {n["id"]: n.get("props", {}).get("_dist")
                for n in nodes}
        if any(d is not None for d in dist.values()):
            prox_ids = [n["id"] for n in sorted(
                nodes, key=lambda n: (dist[n["id"]]
                                      if dist[n["id"]] is not None else 10 ** 6))]

        fused = self._rrf([lst for lst in (lex_ids, vec_ids, prox_ids) if lst])
        if not fused:
            return lexical
        # answer-type prior (parity with the cross-encoder path)
        label_hits = _label_prior_hits(query)
        for n in nodes:
            if n["label"] in label_hits:
                fused[n["id"]] = fused.get(n["id"], 0.0) + 0.01 * _LABEL_PRIOR
        scored = sorted(((node_by_id[nid], score)
                         for nid, score in fused.items()),
                        key=lambda t: t[1], reverse=True)
        return scored


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def cross_encoder_available() -> bool:
    """Check for sentence-transformers WITHOUT importing it.

    A real import pulls in torch + transformers (~7s, and transformers 5.x
    triggers Streamlit's module watcher to scan optional deps like torchvision).
    ``find_spec`` answers "is it installed?" with zero side effects — the heavy
    import happens lazily on the first ``rank()`` instead.
    """
    return importlib.util.find_spec("sentence_transformers") is not None


# One cross-encoder per process: the model weights are heavy to load, so a
# cached instance is reused across queries (load stays lazy — first rank() only).
_cross_encoder_cache: CrossEncoderReranker | None = None


def make_reranker(mode: str | None = None, driver=None):
    """Build the configured reranker; ``auto`` prefers cross-encoder.

    ``hybrid`` builds the v2 RRF fusion reranker (lexical + semantic +
    proximity); ``driver`` lets it build its revision-cached vector index.
    """
    global _cross_encoder_cache
    mode = mode or settings.RERANKER_MODE
    if mode == "hybrid":
        return HybridReranker(driver)
    if mode == "cross-encoder" or (mode == "auto" and cross_encoder_available()):
        if _cross_encoder_cache is None:
            try:
                _cross_encoder_cache = CrossEncoderReranker()
            except Exception:
                _cross_encoder_cache = None
                if mode == "cross-encoder":
                    raise
        if _cross_encoder_cache is not None:
            return _cross_encoder_cache
    return LexicalReranker()


def rank_nodes_by_relevance(query: str, nodes: list[dict], mode: str | None = None):
    """Rank ``nodes`` by relevance to ``query``; returns [(node, score)] best-first."""
    return make_reranker(mode).rank(query, nodes)
