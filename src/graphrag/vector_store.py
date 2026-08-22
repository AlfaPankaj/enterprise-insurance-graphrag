"""In-memory vector store + semantic seed fallback (v2 — WS-C, G16).

A flat, brute-force cosine index over serialized node texts — deliberately
simple: the retrieved subgraphs are small and the zero-dependency ethos wins
over ANN complexity. ``build_vector_store(driver)`` scans the graph once
(tenant-scoped, capped at ``settings.VECTOR_INDEX_MAX_NODES``) and caches the
index **per dataset revision** (same invalidation signal as the answer cache:
any write bumps ``(:Dataset).rev``).

``semantic_seeds(session, query, store, k)`` turns a query embedding into
seed node ids — used by the retriever only when lexical/id seeding finds
nothing (paraphrase queries without ids/keywords), so established retrieval
semantics never change.
"""

from __future__ import annotations

import threading

from graphrag.config import settings
from graphrag.embeddings import cosine, embed_texts
from graphrag.graph_retriever import serialize_node, tenant_predicate

# revision-keyed cache: (dataset, rev) -> (node_count, VectorStore)
_STORE_CACHE: dict[tuple, "VectorStore"] = {}
_CACHE_LOCK = threading.Lock()


class VectorStore:
    """Flat index: texts + unit vectors; brute-force top-k by cosine."""

    def __init__(self):
        self._texts: list[str] = []
        self._vectors: list[list[float]] = []
        self._ids: list[str] = []
        self._labels: list[str] = []
        self._index: dict[str, int] = {}
        self._lock = threading.Lock()

    def add(self, node_id: str, label: str, text: str, vector: list[float]) -> None:
        with self._lock:
            self._ids.append(node_id)
            self._labels.append(label)
            self._texts.append(text)
            self._vectors.append(vector)
            self._index[node_id] = len(self._ids) - 1

    def __len__(self) -> int:
        return len(self._ids)

    def search(self, query_vector: list[float], k: int = 5,
               exclude: set[str] | None = None) -> list[tuple[str, str, float]]:
        """Top-k (node_id, label, cosine) matches for a query vector."""
        exclude = exclude or set()
        scored: list[tuple[str, str, float]] = []
        with self._lock:
            for i, vec in enumerate(self._vectors):
                if self._ids[i] in exclude:
                    continue
                scored.append((self._ids[i], self._labels[i], cosine(query_vector, vec)))
        scored.sort(key=lambda t: t[2], reverse=True)
        return scored[:k]


def clear_vector_cache() -> None:
    """Drop cached indexes (tests / explicit refresh)."""
    with _CACHE_LOCK:
        _STORE_CACHE.clear()


def _scan_nodes(driver, limit: int):
    """Stream every node id/label/serialized text (tenant-scoped)."""
    with driver.session() as session:
        tp = tenant_predicate("n")
        rows = session.run(
            f"MATCH (n) WHERE NOT 'Dataset' IN labels(n) AND {tp} "
            "RETURN labels(n) AS labels, n LIMIT $limit",
            limit=limit,
            tenant=None,
        )
        for row in rows:
            node = row["n"]
            props = {k: v for k, v in dict(node).items()
                     if k != "id" and not isinstance(v, (dict, list))}
            node_id = node["id"] if "id" in dict(node) else props.get("id")
            if not node_id:
                continue
            yield {"id": str(node_id), "label": row["labels"][0],
                   "props": props}


def build_vector_store(driver, revision: tuple | None = None,
                       force: bool = False) -> "VectorStore | None":
    """Build (or fetch from the revision-keyed cache) the vector index.

    Returns None when the dataset revision is unreadable or the graph scan
    fails — callers fall back to non-semantic retrieval.
    """
    from graphrag.cache import graph_revision
    from graphrag.tracing import start_span

    if revision is None:
        revision = graph_revision(driver)
    if revision is None:
        return None
    with _CACHE_LOCK:
        if not force and revision in _STORE_CACHE:
            return _STORE_CACHE[revision]

    store = VectorStore()
    texts: list[str] = []
    nodes: list[tuple[str, str, str]] = []
    with start_span("graphrag.vector.build", {"dataset": revision[0]}):
        try:
            for node in _scan_nodes(driver, settings.VECTOR_INDEX_MAX_NODES):
                text = serialize_node(node)
                texts.append(text)
                nodes.append((node["id"], node["label"], text))
        except Exception:  # noqa: BLE001 - semantic index must never break queries
            return None
    if not texts:
        return None
    vectors = embed_texts(texts)
    for (node_id, label, text), vector in zip(nodes, vectors):
        store.add(node_id, label, text, vector)
    with _CACHE_LOCK:
        _STORE_CACHE[revision] = store
        # bound the cache: drop the oldest revision keys beyond a few
        while len(_STORE_CACHE) > 4:
            _STORE_CACHE.pop(next(iter(_STORE_CACHE)))
    return store


def semantic_seeds(session, query: str, store: VectorStore,
                   k: int = 3) -> list[dict]:
    """Query-embedding → seed nodes (``kind="semantic"``).

    Used only when id/keyword/numeric seeding produced nothing. Raises
    nothing: an embedding failure yields no seeds.
    """
    try:
        vectors = embed_texts([query])
    except Exception:  # noqa: BLE001
        return []
    return [{"id": node_id, "label": label, "kind": "semantic"}
            for node_id, label, _score in store.search(vectors[0], k=k)]
