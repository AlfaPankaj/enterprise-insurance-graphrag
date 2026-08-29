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
from itertools import islice

from graphrag.config import settings
from graphrag.embeddings import cosine, embed_texts, get_embedder
from graphrag.graph_retriever import serialize_node, tenant_predicate

# revision-keyed cache: (dataset, rev) -> (node_count, VectorStore)
_STORE_CACHE: dict[tuple, VectorStore] = {}
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


class Neo4jVectorStore:
    """Thin query facade over Neo4j's native HNSW vector index."""

    def __init__(self, driver, index_name: str, tenant_id: str | None = None,
                 count: int = 0):
        if not index_name.replace("_", "").isalnum():
            raise ValueError("unsafe Neo4j vector index name")
        self.driver = driver
        self.index_name = index_name
        self.tenant_id = tenant_id
        self.count = count

    def __len__(self) -> int:
        return self.count

    def search(self, query_vector: list[float], k: int = 5,
               exclude: set[str] | None = None) -> list[tuple[str, str, float]]:
        exclude = exclude or set()
        with self.driver.session() as session:
            tp = tenant_predicate("node")
            rows = session.run(
                f"CALL db.index.vector.queryNodes($index, $candidate_k, $vector) "
                f"YIELD node, score WHERE {tp} AND NOT node.id IN $exclude "
                "RETURN node.id AS id, labels(node) AS labels, score "
                "ORDER BY score DESC LIMIT $k",
                index=self.index_name, candidate_k=min(max(k * 20, 100), 10_000),
                vector=query_vector, exclude=list(exclude), k=k,
                tenant=(self.tenant_id if self.tenant_id
                        and settings.TENANT_MODE == "column" else None),
            ).data()
        out = []
        for row in rows:
            labels = [label for label in row["labels"] if label != "Searchable"]
            out.append((row["id"], labels[0] if labels else "Record",
                        float(row["score"])))
        return out


def clear_vector_cache() -> None:
    """Drop cached store facades (native vectors remain durable in Neo4j)."""
    with _CACHE_LOCK:
        _STORE_CACHE.clear()


def _build_pgvector_store(revision: tuple, nodes, vectors,
                          cache_key: tuple | None = None,
                          force: bool = False):
    """Open pgvector and initialize it only when no incremental index exists."""
    from graphrag.postgres_backend import PgVectorStore  # optional dep

    dim = len(vectors[0]) if vectors else 384
    store = PgVectorStore(dataset=revision[0], dim=dim)
    reused = not force and len(store) > 0
    if not reused:
        store.rebuild([(nid, label, text, vec)
                       for (nid, label, text), vec in zip(nodes, vectors)])
    with _CACHE_LOCK:
        _STORE_CACHE[cache_key or revision] = store
        while len(_STORE_CACHE) > 4:
            _STORE_CACHE.pop(next(iter(_STORE_CACHE)))
    return store, reused


def _build_pgvector_streaming(driver, revision: tuple, tenant_id: str | None,
                              cache_key: tuple, force: bool = False):
    """Build pgvector in bounded batches without applying the memory-store cap."""
    iterator = (_scan_nodes(driver, 2_147_483_647, tenant_id=tenant_id)
                if tenant_id else _scan_nodes(driver, 2_147_483_647))
    store = None
    batch_size = max(1, min(int(settings.BATCH_SIZE), 512))
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        rows = [(node["id"], node["label"], serialize_node(node)) for node in batch]
        vectors = embed_texts([row[2] for row in rows])
        if not vectors:
            break
        if store is None:
            store, reused = _build_pgvector_store(
                revision, rows, vectors, cache_key=cache_key, force=force
            )
            if reused:
                break
        else:
            store.upsert([
                (node_id, label, text, vector)
                for (node_id, label, text), vector in zip(rows, vectors)
            ])
    return store


def _scan_nodes(driver, limit: int, tenant_id: str | None = None):
    """Stream capped nodes for the demo memory/pgvector backends."""
    with driver.session() as session:
        tp = tenant_predicate("n")
        rows = session.run(
            f"MATCH (n) WHERE NOT 'Dataset' IN labels(n) AND {tp} "
            "RETURN labels(n) AS labels, n LIMIT $limit",
            limit=limit,
            tenant=(tenant_id if tenant_id and settings.TENANT_MODE == "column"
                    else None),
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


def _embedding_model_key() -> str:
    provider = get_embedder()
    model = getattr(provider, "model", None) or provider.name
    return f"{provider.name}:{model}"


def _native_pending_batch(driver, tenant_id: str | None, model_key: str,
                          limit: int) -> list[dict]:
    with driver.session() as session:
        tp = tenant_predicate("n")
        rows = session.run(
            f"MATCH (n) WHERE NOT 'Dataset' IN labels(n) AND {tp} "
            "AND n.id IS NOT NULL AND (n.embedding IS NULL OR n.embedding_model<>$model) "
            "RETURN elementId(n) AS element_id, labels(n) AS labels, n LIMIT $limit",
            tenant=(tenant_id if tenant_id and settings.TENANT_MODE == "column" else None),
            model=model_key, limit=limit,
        )
        out = []
        for row in rows:
            node = row["n"]
            props = {key: value for key, value in dict(node).items()
                     if key not in {"id", "embedding"}
                     and not isinstance(value, (dict, list))}
            labels = [label for label in row["labels"] if label != "Searchable"]
            out.append({
                "element_id": row["element_id"], "id": str(node["id"]),
                "label": labels[0] if labels else "Record", "props": props,
            })
        return out


def _ensure_native_vector_index(driver, dimension: int) -> None:
    name = settings.NEO4J_VECTOR_INDEX
    if not name.replace("_", "").isalnum():
        raise ValueError("unsafe NEO4J_VECTOR_INDEX")
    with driver.session() as session:
        session.run(
            f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
            "FOR (n:Searchable) ON (n.embedding) OPTIONS {indexConfig: {"
            f"`vector.dimensions`: {int(dimension)}, "
            "`vector.similarity_function`: 'cosine'}}"
        ).consume()


def _build_neo4j_store(driver, revision: tuple, tenant_id: str | None,
                        force: bool = False):
    """Incrementally fill durable native vectors; never materialize the graph."""
    import logging

    model_key = _embedding_model_key()
    embedded = 0
    index_ready = False
    batch_size = max(1, min(int(settings.BATCH_SIZE), 256))
    try:
        if force:
            with driver.session() as session:
                session.run(
                    f"MATCH (n:Searchable) WHERE {tenant_predicate('n')} "
                    "REMOVE n.embedding_model",
                    tenant=(tenant_id if tenant_id and settings.TENANT_MODE == "column"
                            else None),
                ).consume()
        while True:
            pending = _native_pending_batch(driver, tenant_id, model_key, batch_size)
            if not pending:
                break
            texts = [serialize_node(node) for node in pending]
            vectors = embed_texts(texts)
            if not vectors:
                break
            if not index_ready:
                _ensure_native_vector_index(driver, len(vectors[0]))
                index_ready = True
            rows = [{"element_id": node["element_id"], "vector": vector}
                    for node, vector in zip(pending, vectors)]
            with driver.session() as session:
                session.run(
                    "UNWIND $rows AS row MATCH (n) WHERE elementId(n)=row.element_id "
                    "SET n:Searchable, n.embedding=row.vector, n.embedding_model=$model",
                    rows=rows, model=model_key,
                ).consume()
            embedded += len(rows)
        with driver.session() as session:
            tp = tenant_predicate("n")
            row = session.run(
                f"MATCH (n:Searchable) WHERE {tp} AND n.embedding_model=$model "
                "RETURN count(n) AS count",
                tenant=(tenant_id if tenant_id and settings.TENANT_MODE == "column" else None),
                model=model_key,
            ).single()
        count = int(row["count"]) if row else embedded
        if not count:
            return None
        store = Neo4jVectorStore(driver, settings.NEO4J_VECTOR_INDEX,
                                 tenant_id=tenant_id, count=count)
        with _CACHE_LOCK:
            _STORE_CACHE[revision] = store
        return store
    except Exception as exc:  # noqa: BLE001 - optional backend compatibility boundary
        logging.getLogger("graphrag.vector").warning(
            "Neo4j native vector store unavailable (%s)", exc
        )
        return None


def update_native_embeddings(driver, entities: list[dict],
                             tenant_id: str | None = None,
                             dataset_name: str | None = None,
                             replace: bool = False) -> int:
    """Incrementally update vectors for CDC upserts after their graph commit."""
    backend = (settings.VECTOR_BACKEND or "memory").strip().lower()
    if backend not in {"neo4j", "pgvector"} \
            or not settings.VECTOR_INCREMENTAL_ENABLED or not entities:
        return 0
    model_key = _embedding_model_key()
    texts = []
    valid = []
    for entity in entities:
        props = entity.get("new_props", entity.get("props", {}))
        valid.append(entity)
        texts.append(serialize_node({"id": entity["id"], "label": entity["label"],
                                     "props": props}))
    vectors = embed_texts(texts)
    if backend == "pgvector":
        from graphrag.cache import graph_revision
        from graphrag.postgres_backend import PgVectorStore

        revision = graph_revision(driver, tenant_id) if dataset_name is None else None
        resolved_dataset = dataset_name or (revision[0] if revision else None)
        if not resolved_dataset:
            return 0
        dataset = (f"{tenant_id}:{resolved_dataset}"
                   if tenant_id else resolved_dataset)
        store = PgVectorStore(dataset=dataset, dim=len(vectors[0]))
        entries = [
            (entity["id"], entity["label"], text, vector)
            for entity, text, vector in zip(valid, texts, vectors)
        ]
        try:
            if replace:
                store.rebuild(entries)
            else:
                store.upsert(entries)
        finally:
            store.close()
        clear_vector_cache()
        return len(valid)

    _ensure_native_vector_index(driver, len(vectors[0]))
    updated = 0
    with driver.session() as session:
        for entity, vector in zip(valid, vectors):
            if tenant_id and settings.TENANT_MODE == "column":
                result = session.run(
                    f"MATCH (n:{entity['label']} {{id:$id, tenant_id:$tenant}}) "
                    "SET n:Searchable, n.embedding=$vector, n.embedding_model=$model "
                    "RETURN count(n) AS count", id=entity["id"], tenant=tenant_id,
                    vector=vector, model=model_key,
                ).single()
            else:
                result = session.run(
                    f"MATCH (n:{entity['label']} {{id:$id}}) "
                    "SET n:Searchable, n.embedding=$vector, n.embedding_model=$model "
                    "RETURN count(n) AS count", id=entity["id"], vector=vector,
                    model=model_key,
                ).single()
            updated += int(result["count"]) if result else 0
    clear_vector_cache()
    return updated


def delete_vector_embeddings(driver, node_ids: list[str],
                             tenant_id: str | None = None,
                             dataset_name: str | None = None) -> int:
    """Remove deleted CDC entities from external pgvector storage."""
    if (settings.VECTOR_BACKEND or "memory").strip().lower() != "pgvector" \
            or not node_ids:
        return 0
    from graphrag.cache import graph_revision
    from graphrag.postgres_backend import PgVectorStore

    revision = graph_revision(driver, tenant_id) if dataset_name is None else None
    resolved_dataset = dataset_name or (revision[0] if revision else None)
    if not resolved_dataset:
        return 0
    dataset = f"{tenant_id}:{resolved_dataset}" if tenant_id else resolved_dataset
    # Dimension is irrelevant to DELETE but table construction requires it.
    dimension = len(embed_texts(["dimension probe"])[0])
    store = PgVectorStore(dataset=dataset, dim=dimension)
    try:
        store.delete(node_ids)
    finally:
        store.close()
    clear_vector_cache()
    return len(node_ids)


def build_vector_store(driver, revision: tuple | None = None,
                       force: bool = False, tenant_id: str | None = None) -> VectorStore | Neo4jVectorStore | None:
    """Build (or fetch from the revision-keyed cache) the vector index.

    Returns None when the dataset revision is unreadable or the graph scan
    fails — callers fall back to non-semantic retrieval.
    """
    from graphrag.cache import graph_revision
    from graphrag.tracing import start_span

    if revision is None:
        revision = graph_revision(driver, tenant_id)
    if revision is None:
        return None
    cache_key = (*revision, tenant_id or "")
    with _CACHE_LOCK:
        if not force and cache_key in _STORE_CACHE:
            return _STORE_CACHE[cache_key]

    backend = (settings.VECTOR_BACKEND or "memory").strip().lower()
    if backend == "neo4j":
        return _build_neo4j_store(driver, cache_key, tenant_id, force=force)
    if backend == "pgvector":
        try:
            pg_revision = (
                f"{tenant_id}:{revision[0]}" if tenant_id else revision[0],
                revision[1],
            )
            return _build_pgvector_streaming(
                driver, pg_revision, tenant_id, cache_key, force=force
            )
        except Exception as exc:  # noqa: BLE001 - fall back, never break queries
            import logging
            logging.getLogger("graphrag.vector").warning(
                "pgvector store unavailable (%s) — using in-memory index", exc
            )

    store = VectorStore()
    texts: list[str] = []
    nodes: list[tuple[str, str, str]] = []
    with start_span("graphrag.vector.build", {"dataset": revision[0]}):
        try:
            max_nodes = int(settings.VECTOR_INDEX_MAX_NODES)
            scanned = (_scan_nodes(driver, max_nodes + 1, tenant_id=tenant_id)
                       if tenant_id else _scan_nodes(driver, max_nodes + 1))
            for index, node in enumerate(scanned):
                if index >= max_nodes:
                    import logging
                    logging.getLogger("graphrag.vector").warning(
                        "in-memory vector index truncated at %d nodes; semantic "
                        "recall is incomplete. Use VECTOR_BACKEND=neo4j or pgvector.",
                        max_nodes,
                    )
                    break
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
        _STORE_CACHE[cache_key] = store
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
