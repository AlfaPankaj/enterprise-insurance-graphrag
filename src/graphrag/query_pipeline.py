"""Query pipeline — retrieve → re-rank → prune → answer (Shot 2).

``run_query(driver, query)`` executes the full token-optimized retrieval flow
and returns baseline vs optimized token counts, savings %, and an answer —
through the v2 multi-provider chain with a deterministic extractive fallback.
Phase 4 adds full lineage logging; v2 adds identity/tenant attribution, PII
masking, guardrails, an **answer cache**, and **streaming**:

* ``run_query(...)`` — one-shot; consults the cache first
  (``settings.CACHE_ENABLED``), keyed by query + pipeline parameters + tenant
  + PII scope + dataset revision (any graph write bumps the revision, so
  cached answers can never survive a mutation). Cache hits are re-audited
  (fresh audit id, ``cached: true``) — the trail stays complete.
* ``stream_query(...)`` — the same pipeline as a generator: yields ``status`` /
  ``meta`` / ``delta`` / ``done`` / ``blocked`` events (SSE endpoint + live
  UI). Live token streaming is disabled when PII masking is active for the caller (the
  answer is delivered as one buffered delta instead).
"""

from __future__ import annotations

import copy
import time
import uuid

from graphrag.answer_generator import generate_answer, stream_answer
from graphrag.cache import (
    build_cache_key,
    graph_revision,
    query_cache,
    runtime_cache_signature,
)
from graphrag.config import settings
from graphrag.context_pruner import prune_context
from graphrag.graph_retriever import retrieve_subgraph, serialize_subgraph
from graphrag.guardrails import run_guardrails
from graphrag.identity import UserIdentity
from graphrag.pii import MaskingPolicy, redact_node, scrub_answer
from graphrag.prometheus import (
    cache_hits_total,
    cache_misses_total,
    llm_cost_total,
    llm_fallbacks_total,
    token_savings,
)
from graphrag.reranker import _label_prior_hits, make_reranker
from graphrag.token_counter import count_tokens
from graphrag.tracing import start_span
from graphrag.traversal_logger import audit_store, build_audit_record

_BLOCKED_ANSWER = ("Query blocked by guardrail policy "
                   "(instruction injection detected).")


def _query_hash(query: str) -> str:
    """Short hash for span attributes (never log raw query text)."""
    import hashlib
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]


def _pii_scope(policy: MaskingPolicy) -> str:
    """Cache-key scope of the effective PII policy: off | full | restricted."""
    if not policy.active:
        return "off"
    if policy.allows("PII_IDENTITY") and policy.allows("PII_CONTACT"):
        return "full"
    return "restricted"


def _prepare(driver, query: str, max_hops: int, token_budget: int,
             reranker_mode: str | None, identity: UserIdentity | None) -> dict:
    """Shared retrieval → re-rank → prune half of the pipeline.

    Returns the context dict consumed by ``_finalize`` — includes the
    (PII-masked) subgraph, ranking, pruned context, baseline tokens, savings,
    per-stage timings, and the resolved tenant/PII policy.
    """
    t0 = time.perf_counter()
    tenant_id = identity.tenant_id if identity else None
    scoped_tenant = tenant_id if settings.TENANT_MODE == "column" else None
    pii_policy = MaskingPolicy.for_roles(set(identity.roles) if identity else None)
    qhash = _query_hash(query)

    with start_span("graphrag.retrieve", {"query_hash": qhash,
                                          "max_hops": max_hops,
                                          "tenant": scoped_tenant or ""}):
        subgraph = retrieve_subgraph(driver, query, max_hops, tenant_id=scoped_tenant)
    # PII masking: rank/prune/answer only ever see the policy-applied view
    if pii_policy.active:
        subgraph["nodes"] = [redact_node(n, pii_policy) for n in subgraph["nodes"]]
    baseline_text = serialize_subgraph(subgraph)
    baseline_tokens = count_tokens(baseline_text)
    t1 = time.perf_counter()

    hybrid_mode = (reranker_mode or settings.RERANKER_MODE) == "hybrid"
    store = None
    if hybrid_mode:
        # v2 semantic seed fallback: when id/keyword/numeric seeding found
        # nothing, embed the query and seed from the revision-cached vector
        # index (paraphrase queries). Builds the index lazily, once per
        # dataset revision; pure-Cypher retrieval is untouched otherwise.
        try:
            from graphrag.vector_store import build_vector_store
            store = build_vector_store(driver, tenant_id=scoped_tenant)
        except Exception:  # noqa: BLE001 - hybrid degrades to lexical
            store = None
        if store is not None and not subgraph["seeds"]:
            try:
                from graphrag.vector_store import semantic_seeds
                with start_span("graphrag.semantic_seeds",
                                {"query_hash": qhash, "store_size": len(store)}):
                    with driver.session() as session:
                        extra = semantic_seeds(session, query, store, k=3)
                if extra:
                    from graphrag.graph_retriever import retrieve_subgraph as _rs
                    subgraph = _rs(driver, query, max_hops,
                                   tenant_id=scoped_tenant,
                                   vector_store=store)
                    if pii_policy.active:
                        subgraph["nodes"] = [redact_node(n, pii_policy)
                                             for n in subgraph["nodes"]]
                    baseline_text = serialize_subgraph(subgraph)
                    baseline_tokens = count_tokens(baseline_text)
            except Exception:  # noqa: BLE001
                pass

    # Reuse the exact revision/tenant-scoped store resolved above. Hybrid
    # ranking still computes the same query/document vectors and RRF scores;
    # this only removes a redundant store-resolution round trip.
    reranker = make_reranker(reranker_mode, driver=driver, vector_store=store)

    # edge-aware ranking text: each node carries its direct neighbors + edge
    # types, so the scorer can connect "coverage COV-0017" to the claim's policy
    neighbors: dict[str, list[str]] = {}
    for e in subgraph["edges"]:
        neighbors.setdefault(e["source"], []).append(f"{e['type']}: {e['target']}")
        neighbors.setdefault(e["target"], []).append(f"{e['type']}: {e['source']}")
    # hop distance from seeds (hybrid proximity signal)
    dist: dict[str, int] = {s["id"]: 0 for s in subgraph["seeds"]}
    frontier = list(dist)
    while frontier:
        nxt: list[str] = []
        for e in subgraph["edges"]:
            for a, b in ((e["source"], e["target"]), (e["target"], e["source"])):
                if a in frontier and b not in dist:
                    dist[b] = dist[a] + 1
                    nxt.append(b)
        frontier = nxt
    rank_nodes = []
    for n in subgraph["nodes"]:
        ctx = neighbors.get(n["id"])
        if ctx:
            copy = dict(n)
            copy["props"] = {**n["props"], "_ctx": "neighbors: " + ", ".join(sorted(ctx)),
                             "_dist": dist.get(n["id"])}
            rank_nodes.append(copy)
        else:
            rank_nodes.append(n)

    with start_span("graphrag.rerank", {"query_hash": qhash,
                                        "nodes": len(rank_nodes)}):
        ranked = reranker.rank(query, rank_nodes)
    t2 = time.perf_counter()
    # map scores back onto the ORIGINAL (un-enriched) nodes, so the pruned
    # context serializes identically to the baseline — token accounting stays
    # consistent and savings are never inflated by ranking hints.
    node_by_id = {n["id"]: n for n in subgraph["nodes"]}
    ranked = [(node_by_id[enode["id"]], score) for enode, score in ranked]

    # protect the local neighborhood: seeds + direct neighbors always survive
    # pruning (the answer to a graph question lives within 1 hop of the entity)
    seed_ids = {s["id"] for s in subgraph["seeds"]}
    label_by_id = {n["id"]: n["label"] for n in subgraph["nodes"]}
    protected = set(seed_ids)
    for e in subgraph["edges"]:
        if e["source"] in seed_ids:
            protected.add(e["target"])
        if e["target"] in seed_ids:
            protected.add(e["source"])
    # answer-type protection: nodes whose label matches what the query asks
    # about (e.g. "coverages" -> Coverage) survive up to 2 hops from a seed —
    # "Which coverages apply to claim CLM-0106?" must not prune the coverages.
    # Claim/Policy are excluded: they are the neighborhood backbone, already
    # covered by 1-hop seed protection — protecting them would re-expand every
    # sibling claim and defeat the token budget.
    answer_labels = _label_prior_hits(query) - {"Claim", "Policy"}
    if answer_labels:
        for e in subgraph["edges"]:
            other = None
            if e["source"] in protected and label_by_id.get(e["target"]) in answer_labels:
                other = e["target"]
            elif e["target"] in protected and label_by_id.get(e["source"]) in answer_labels:
                other = e["source"]
            if other:
                protected.add(other)

    with start_span("graphrag.prune", {"query_hash": qhash,
                                       "budget": token_budget,
                                       "baseline_tokens": baseline_tokens}):
        pruned = prune_context(ranked, token_budget, edges=subgraph["edges"],
                               protected_ids=sorted(protected))
    t3 = time.perf_counter()

    # no-result queries must not report "100% savings" — nothing was retrieved
    savings = 0.0 if (subgraph["node_count"] == 0 or baseline_tokens == 0) else \
        round((1 - pruned["tokens"] / baseline_tokens) * 100, 2)

    return {
        "query": query,
        "max_hops": max_hops,
        "token_budget": token_budget,
        "reranker_mode": reranker_mode,
        "subgraph": subgraph,
        "ranked": ranked,
        "pruned": pruned,
        "baseline_tokens": baseline_tokens,
        "savings": savings,
        "reranker": reranker,
        "scoped_tenant": scoped_tenant,
        "pii_policy": pii_policy,
        "identity": identity,
        "t0": t0, "t1": t1, "t2": t2, "t3": t3,
    }


def _finalize(ctx: dict, answer: dict) -> dict:
    """Shared answer post-processing + audit + result assembly.

    Applies the PII safety net, guardrail checks (with blocked-answer
    enforcement), token/timing accounting, and persists the audit record —
    then returns the full result dict.
    """
    query = ctx["query"]
    pruned = ctx["pruned"]
    pii_policy = ctx["pii_policy"]
    t4 = time.perf_counter()

    if pii_policy.active:
        answer = {**answer, "answer": scrub_answer(answer["answer"], pii_policy)}
    guardrails = run_guardrails(query, answer["answer"], pruned["text"])
    if guardrails.blocked:
        answer = {"answer": _BLOCKED_ANSWER, "mode": "blocked", "model": None,
                  "fallback_reason": "guardrail: " + ", ".join(guardrails.injection_hits)}

    tokens = {
        "before": ctx["baseline_tokens"],
        "after": pruned["tokens"],
        "savings_percent": max(ctx["savings"], 0.0),
    }
    timings = {
        "retrieval_ms": round((ctx["t1"] - ctx["t0"]) * 1000, 2),
        "rerank_ms": round((ctx["t2"] - ctx["t1"]) * 1000, 2),
        "prune_ms": round((ctx["t3"] - ctx["t2"]) * 1000, 2),
        "answer_ms": round((t4 - ctx["t3"]) * 1000, 2),
        "total_ms": round((t4 - ctx["t0"]) * 1000, 2),
    }
    if ctx.get("_ttft_ms") is not None:
        timings["time_to_first_token_ms"] = ctx["_ttft_ms"]
    if ctx.get("_generation_ttft_ms") is not None:
        timings["generation_time_to_first_token_ms"] = ctx["_generation_ttft_ms"]

    # --- explainability (Shot 3): build + persist the audit record ---
    audit_record = build_audit_record(
        query=query, subgraph=ctx["subgraph"], ranked=ctx["ranked"], pruned=pruned,
        tokens=tokens, answer=answer["answer"], answer_mode=answer["mode"],
        answer_model=answer.get("model"), reranker=ctx["reranker"].name,
        max_hops=ctx["max_hops"],
        t0=ctx["t0"], t1=ctx["t1"], t2=ctx["t2"], t3=ctx["t3"], t4=t4,
        user=ctx["identity"].as_dict() if ctx["identity"] else None,
        tenant_id=ctx["scoped_tenant"],
        answer_provider=answer.get("provider"),
        usage=answer.get("usage"),
        cost_usd=answer.get("cost_usd"),
    )
    if ctx.get("_ttft_ms") is not None:
        audit_record["timings_ms"]["time_to_first_token_ms"] = ctx["_ttft_ms"]
    if ctx.get("_generation_ttft_ms") is not None:
        audit_record["timings_ms"]["generation_time_to_first_token_ms"] = \
            ctx["_generation_ttft_ms"]
    if settings.AUDIT_ENABLED:
        audit_store.append(audit_record)
    ctx["_audit_record"] = audit_record

    traversal = {
        "audit_id": audit_record["audit_id"],
        "nodes_visited": audit_record["traversal"]["nodes_visited"],
        "edges_traversed": audit_record["traversal"]["edges_traversed"],
        "paths": audit_record["traversal"]["paths"],
        "cypher": audit_record["cypher"],
        "timings_ms": timings,
    }

    # v2 observability: cost + fallback + savings metrics
    if answer.get("cost_usd") is not None:
        llm_cost_total.inc(answer["cost_usd"])
    if answer.get("fallback_reason"):
        llm_fallbacks_total.inc()
    token_savings.observe(max(ctx["savings"], 0.0) / 100.0)

    return {
        "query": query,
        "answer": answer["answer"],
        "answer_mode": answer["mode"],
        "answer_model": answer.get("model"),
        "answer_fallback": answer.get("fallback_reason"),
        "answer_provider": answer.get("provider"),
        "usage": answer.get("usage"),
        "cost_usd": answer.get("cost_usd"),
        "guardrails": guardrails.as_dict(),
        "user": ctx["identity"].as_dict() if ctx["identity"] else None,
        "tenant_id": ctx["scoped_tenant"],
        "reranker": ctx["reranker"].name,
        "tokens": tokens,
        "retrieval": {
            "seeds": [s["id"] for s in ctx["subgraph"]["seeds"]],
            "node_count": ctx["subgraph"]["node_count"],
            "edge_count": ctx["subgraph"]["edge_count"],
        },
        "pruned": {
            "node_count": pruned["node_count"],
            "kept": pruned["kept"],
            "dropped": pruned["dropped"],
            "dropped_count": pruned["dropped_count"],
            "budget": ctx["token_budget"],
        },
        "traversal": traversal,
        "execution_time_ms": round((t4 - ctx["t0"]) * 1000, 2),
        "time_to_first_token_ms": ctx.get("_ttft_ms"),
    }


def _cache_key(ctx_signature: dict) -> str | None:
    """Cache key for the query + pipeline + data signature; None = don't cache."""
    if not settings.CACHE_ENABLED:
        return None
    rev = graph_revision(ctx_signature["driver"], ctx_signature.get("tenant"))
    if rev is None:
        return None  # revision unreadable (DB down / no marker) -> skip caching
    return build_cache_key(
        query=ctx_signature["query"],
        max_hops=ctx_signature["max_hops"],
        token_budget=ctx_signature["token_budget"],
        reranker_mode=ctx_signature["reranker_mode"] or settings.RERANKER_MODE,
        answer_mode=ctx_signature["answer_mode"] or settings.ANSWER_MODE,
        tenant=ctx_signature["tenant"] or "",
        pii_scope=ctx_signature["pii_scope"],
        include_context=ctx_signature.get("include_context", False),
        runtime=runtime_cache_signature(
            reranker_mode=ctx_signature["reranker_mode"],
            answer_mode=ctx_signature["answer_mode"],
        ),
        dataset=rev[0],
        rev=rev[1],
    )


def _serve_cache_hit(key: str, query: str, entry: dict | None = None,
                     time_to_first_token_ms: float | None = None) -> dict:
    """Re-audit + return a cached result (fresh audit id, cached flags)."""
    t_hit0 = time.perf_counter()
    entry = entry or query_cache.get(key)
    if entry is None:  # defensive: a very short TTL may expire between calls
        raise KeyError(f"cache entry expired: {key}")
    hit_ms = round((time.perf_counter() - t_hit0) * 1000, 2)
    result = copy.deepcopy(entry["result"])

    # fresh audit event: same content, new id/timestamp, cached marker
    new_audit = copy.deepcopy(entry["audit"])
    new_audit["audit_id"] = uuid.uuid4().hex[:12]
    new_audit["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    new_audit["cached"] = True
    new_audit.pop("record_hash", None)
    audit_timings = dict(new_audit.get("timings_ms", {}))
    audit_timings.pop("time_to_first_token_ms", None)
    audit_timings.pop("generation_time_to_first_token_ms", None)
    audit_timings["cache_lookup_ms"] = hit_ms
    if time_to_first_token_ms is not None:
        audit_timings["time_to_first_token_ms"] = time_to_first_token_ms
    new_audit["timings_ms"] = audit_timings
    if settings.AUDIT_ENABLED:
        audit_store.append(new_audit)

    result["cached"] = True
    result["cached_execution_ms"] = hit_ms
    result["cached_original_execution_ms"] = result.get("execution_time_ms")
    result["execution_time_ms"] = hit_ms
    result["traversal"] = {**result["traversal"], "audit_id": new_audit["audit_id"]}
    result["cached_original_time_to_first_token_ms"] = result.get(
        "time_to_first_token_ms"
    )
    result["time_to_first_token_ms"] = time_to_first_token_ms
    cached_timings = dict(result["traversal"].get("timings_ms", {}))
    cached_timings.pop("time_to_first_token_ms", None)
    cached_timings.pop("generation_time_to_first_token_ms", None)
    cached_timings["cache_lookup_ms"] = hit_ms
    if time_to_first_token_ms is not None:
        cached_timings["time_to_first_token_ms"] = time_to_first_token_ms
    result["traversal"]["timings_ms"] = cached_timings
    if entry.get("context") is not None:
        result["context"] = entry["context"]
    cache_hits_total.inc()
    return result


def run_query(driver, query: str, max_hops: int | None = None,
              token_budget: int | None = None, reranker_mode: str | None = None,
              answer_mode: str | None = None,
              identity: UserIdentity | None = None,
              include_context: bool = False) -> dict:
    """Full token-optimized retrieval pipeline for a natural-language query.

    Returns the answer + token savings, and — for explainability (Shot 3) —
    the full traversal lineage (nodes/edges visited, Cypher used, per-stage
    timings). Every call is appended to the audit trail store.

    v2 additions: ``identity`` (a ``UserIdentity``) drives tenant scoping,
    PII masking, and audit attribution; guardrail findings are recorded in
    ``result["guardrails"]`` and enforced (refusal) when enabled; and with
    ``settings.CACHE_ENABLED`` an in-process cache answers repeated queries
    instantly (cache hits carry ``cached: true`` and a fresh audit record).

    ``answer_mode`` in {"extractive", "auto", "llm"} — default
    ``settings.ANSWER_MODE`` (extractive = deterministic; auto = provider
    chain with extractive fallback; llm = require a provider).

    ``include_context`` (v2, benchmarks/evals): attach the pruned context
    text + node list to the result as ``result["context"]`` — off by default
    to keep the API payload lean.
    """
    max_hops = max_hops or settings.MAX_HOPS
    token_budget = token_budget or settings.MAX_TOKENS

    tenant_id = identity.tenant_id if identity else None
    scoped_tenant = tenant_id if settings.TENANT_MODE == "column" else None
    pii_policy = MaskingPolicy.for_roles(set(identity.roles) if identity else None)

    # ---- v2 answer cache ----
    key = _cache_key({
        "driver": driver,
        "query": query,
        "max_hops": max_hops,
        "token_budget": token_budget,
        "reranker_mode": reranker_mode,
        "answer_mode": answer_mode,
        "tenant": scoped_tenant,
        "pii_scope": _pii_scope(pii_policy),
        "include_context": include_context,
    })
    entry = query_cache.get(key) if key is not None else None
    if entry is not None:
        return _serve_cache_hit(key, query, entry)
    if key is not None:
        cache_misses_total.inc()

    ctx = _prepare(driver, query, max_hops, token_budget, reranker_mode, identity)
    with start_span("graphrag.answer", {"query_hash": _query_hash(query),
                                        "mode": answer_mode or settings.ANSWER_MODE}):
        answer = generate_answer(query, ctx["pruned"], mode=answer_mode)
    result = _finalize(ctx, answer)

    requested_answer_mode = answer_mode or settings.ANSWER_MODE
    if key is not None and _cacheable_result(result, requested_answer_mode):
        query_cache.put(key, {
            "result": result,
            "audit": ctx["_audit_record"],
            "context": _context_payload(ctx) if include_context else None,
        })
    if include_context:
        result["context"] = _context_payload(ctx)
    return result


def _cacheable_result(result: dict, requested_answer_mode: str) -> bool:
    """Do not retain availability-driven or interrupted fallback answers."""
    if result.get("answer_mode") == "blocked":
        return True
    if result.get("answer_fallback"):
        return False
    if requested_answer_mode == "auto" and result.get("answer_mode") != "llm":
        return False
    return True


def _context_payload(ctx: dict) -> dict:
    """Pruned context text + nodes (for benchmarks/evals)."""
    return {
        "text": ctx["pruned"].get("text", ""),
        "nodes": ctx["pruned"].get("nodes", []),
    }


def stream_query(driver, query: str, max_hops: int | None = None,
                 token_budget: int | None = None, reranker_mode: str | None = None,
                 answer_mode: str | None = None,
                 identity: UserIdentity | None = None):
    """The query pipeline as an event generator (SSE endpoint + live UI).

    ``status`` events are additive progress/latency metadata; ``meta``,
    ``delta``, ``done``, and ``blocked`` retain their established contracts.
    Cache hits are re-audited and delivered through the same stream protocol.
    """
    stream_t0 = time.perf_counter()
    max_hops = max_hops or settings.MAX_HOPS
    token_budget = token_budget or settings.MAX_TOKENS

    yield {"type": "status", "stage": "cache_lookup", "state": "started"}
    tenant_id = identity.tenant_id if identity else None
    scoped_tenant = tenant_id if settings.TENANT_MODE == "column" else None
    pii_policy = MaskingPolicy.for_roles(set(identity.roles) if identity else None)
    key = _cache_key({
        "driver": driver,
        "query": query,
        "max_hops": max_hops,
        "token_budget": token_budget,
        "reranker_mode": reranker_mode,
        "answer_mode": answer_mode,
        "tenant": scoped_tenant,
        "pii_scope": _pii_scope(pii_policy),
        "include_context": False,
    })
    entry = query_cache.get(key) if key is not None else None
    if entry is not None:
        ttft_ms = round((time.perf_counter() - stream_t0) * 1000, 2)
        result = _serve_cache_hit(
            key, query, entry, time_to_first_token_ms=ttft_ms
        )
        yield {"type": "status", "stage": "cache_lookup", "state": "hit",
               "elapsed_ms": ttft_ms}
        if result.get("answer_mode") == "blocked":
            yield {"type": "blocked", "result": result}
            return
        yield {
            "type": "meta",
            "streaming": True,
            "cached": True,
            "retrieval": result.get("retrieval", {}),
            "reranker": result.get("reranker"),
        }
        yield {"type": "status", "stage": "generation", "state": "first_token",
               "time_to_first_token_ms": ttft_ms, "cached": True}
        yield {"type": "delta", "text": result["answer"], "cached": True}
        yield {"type": "done", "result": result}
        return
    if key is not None:
        cache_misses_total.inc()

    yield {"type": "status", "stage": "retrieval", "state": "started"}
    ctx = _prepare(driver, query, max_hops, token_budget, reranker_mode, identity)
    pruned = ctx["pruned"]
    pii_policy = ctx["pii_policy"]
    stage_timings = {
        "retrieval_ms": round((ctx["t1"] - ctx["t0"]) * 1000, 2),
        "rerank_ms": round((ctx["t2"] - ctx["t1"]) * 1000, 2),
        "prune_ms": round((ctx["t3"] - ctx["t2"]) * 1000, 2),
    }
    yield {"type": "status", "stage": "retrieval", "state": "completed",
           "timings_ms": stage_timings}
    yield {
        "type": "meta",
        "streaming": not pii_policy.active,
        "cached": False,
        "retrieval": {
            "seeds": [s["id"] for s in ctx["subgraph"]["seeds"]],
            "node_count": ctx["subgraph"]["node_count"],
            "edge_count": ctx["subgraph"]["edge_count"],
            "baseline_tokens": ctx["baseline_tokens"],
            "pruned_tokens": pruned["tokens"],
        },
        "reranker": ctx["reranker"].name,
        "timings_ms": stage_timings,
    }

    # Input guardrail before any answer token leaves. Output checks still run
    # in _finalize, exactly as on the buffered path.
    from graphrag.guardrails import scan_query
    input_guard = scan_query(query) if settings.GUARDRAILS_ENABLED else None
    if input_guard is not None and input_guard.blocked:
        blocked = {
            "answer": _BLOCKED_ANSWER,
            "mode": "blocked",
            "model": None,
            "fallback_reason": "guardrail: " + ", ".join(input_guard.injection_hits),
        }
        result = _finalize(ctx, blocked)
        if key is not None:
            query_cache.put(key, {"result": result, "audit": ctx["_audit_record"],
                                  "context": None})
        yield {"type": "blocked", "result": result}
        return

    generation_t0 = time.perf_counter()
    yield {"type": "status", "stage": "generation", "state": "started"}
    use_live = not pii_policy.active
    first_token_recorded = False
    if use_live:
        answer_dict: dict | None = None
        with start_span("graphrag.answer.stream",
                        {"query_hash": _query_hash(query),
                         "mode": answer_mode or settings.ANSWER_MODE}):
            for ev in stream_answer(query, pruned, mode=answer_mode):
                if ev["type"] == "delta":
                    if ev.get("text") and not first_token_recorded:
                        now = time.perf_counter()
                        ctx["_ttft_ms"] = round((now - stream_t0) * 1000, 2)
                        ctx["_generation_ttft_ms"] = round(
                            (now - generation_t0) * 1000, 2
                        )
                        first_token_recorded = True
                        yield {
                            "type": "status", "stage": "generation",
                            "state": "first_token",
                            "time_to_first_token_ms": ctx["_ttft_ms"],
                            "generation_time_to_first_token_ms":
                                ctx["_generation_ttft_ms"],
                        }
                    yield {"type": "delta", "text": ev["text"]}
                else:
                    answer_dict = {
                        "answer": ev["answer"], "mode": ev["mode"],
                        "model": ev.get("model"), "provider": ev.get("provider"),
                        "usage": ev.get("usage"), "cost_usd": ev.get("cost_usd"),
                        "fallback_reason": ev.get("fallback_reason"),
                    }
        if answer_dict is None:  # pragma: no cover
            # ``stream_answer`` always terminates with a done event.
            answer_dict = generate_answer(query, pruned, mode=answer_mode)
    else:
        # PII masking active: buffer so no potentially sensitive partial text
        # can leave before the complete answer is scrubbed.
        answer_dict = generate_answer(query, pruned, mode=answer_mode)
        if pii_policy.active:
            answer_dict = {**answer_dict,
                           "answer": scrub_answer(answer_dict["answer"], pii_policy)}
        now = time.perf_counter()
        ctx["_ttft_ms"] = round((now - stream_t0) * 1000, 2)
        ctx["_generation_ttft_ms"] = round((now - generation_t0) * 1000, 2)
        first_token_recorded = True
        yield {
            "type": "status", "stage": "generation", "state": "first_token",
            "time_to_first_token_ms": ctx["_ttft_ms"],
            "generation_time_to_first_token_ms": ctx["_generation_ttft_ms"],
        }
        yield {"type": "delta", "text": answer_dict["answer"]}

    # Empty provider answers are rejected by providers, but retain a metric for
    # defensive/custom implementations that emit no non-empty delta.
    if not first_token_recorded:
        now = time.perf_counter()
        ctx["_ttft_ms"] = round((now - stream_t0) * 1000, 2)
        ctx["_generation_ttft_ms"] = round((now - generation_t0) * 1000, 2)

    result = _finalize(ctx, answer_dict)
    requested_answer_mode = answer_mode or settings.ANSWER_MODE
    if key is not None and _cacheable_result(result, requested_answer_mode):
        query_cache.put(key, {"result": result, "audit": ctx["_audit_record"],
                              "context": None})
    yield {"type": "status", "stage": "generation", "state": "completed",
           "elapsed_ms": result["traversal"]["timings_ms"]["answer_ms"]}
    yield {"type": "done", "result": result}
