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
* ``stream_query(...)`` — the same pipeline as a generator: yields ``meta`` /
  ``delta`` / ``done`` / ``blocked`` events (SSE endpoint + live UI). Live
  token streaming is disabled when PII masking is active for the caller (the
  answer is delivered as one buffered delta instead).
"""

from __future__ import annotations

import time
import uuid

from graphrag.answer_generator import generate_answer, stream_answer
from graphrag.cache import build_cache_key, graph_revision, query_cache
from graphrag.config import settings
from graphrag.context_pruner import prune_context
from graphrag.graph_retriever import retrieve_subgraph, serialize_subgraph
from graphrag.guardrails import run_guardrails
from graphrag.identity import UserIdentity
from graphrag.pii import MaskingPolicy, redact_node, scrub_answer
from graphrag.prometheus import (cache_hits_total, cache_misses_total,
                                 llm_cost_total, llm_fallbacks_total,
                                 token_savings)
from graphrag.reranker import _label_prior_hits, make_reranker
from graphrag.token_counter import count_tokens
from graphrag.traversal_logger import audit_store, build_audit_record

_BLOCKED_ANSWER = ("Query blocked by guardrail policy "
                   "(instruction injection detected).")


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

    subgraph = retrieve_subgraph(driver, query, max_hops, tenant_id=scoped_tenant)
    # PII masking: rank/prune/answer only ever see the policy-applied view
    if pii_policy.active:
        subgraph["nodes"] = [redact_node(n, pii_policy) for n in subgraph["nodes"]]
    baseline_text = serialize_subgraph(subgraph)
    baseline_tokens = count_tokens(baseline_text)
    t1 = time.perf_counter()

    reranker = make_reranker(reranker_mode)

    # edge-aware ranking text: each node carries its direct neighbors + edge
    # types, so the scorer can connect "coverage COV-0017" to the claim's policy
    neighbors: dict[str, list[str]] = {}
    for e in subgraph["edges"]:
        neighbors.setdefault(e["source"], []).append(f"{e['type']}: {e['target']}")
        neighbors.setdefault(e["target"], []).append(f"{e['type']}: {e['source']}")
    rank_nodes = []
    for n in subgraph["nodes"]:
        ctx = neighbors.get(n["id"])
        if ctx:
            copy = dict(n)
            copy["props"] = {**n["props"], "_ctx": "neighbors: " + ", ".join(sorted(ctx))}
            rank_nodes.append(copy)
        else:
            rank_nodes.append(n)

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
    }


def _cache_key(ctx_signature: dict) -> str | None:
    """Cache key for the query + pipeline + data signature; None = don't cache."""
    if not settings.CACHE_ENABLED:
        return None
    rev = graph_revision(ctx_signature["driver"])
    if rev is None:
        return None  # revision unreadable (DB down / no marker) -> skip caching
    return build_cache_key(
        query=ctx_signature["query"],
        max_hops=ctx_signature["max_hops"],
        token_budget=ctx_signature["token_budget"],
        reranker_mode=ctx_signature["reranker_mode"],
        answer_mode=ctx_signature["answer_mode"],
        tenant=ctx_signature["tenant"] or "",
        pii_scope=ctx_signature["pii_scope"],
        dataset=rev[0],
        rev=rev[1],
    )


def _serve_cache_hit(key: str, query: str) -> dict:
    """Re-audit + return a cached result (fresh audit id, cached flags)."""
    t_hit0 = time.perf_counter()
    entry = query_cache.get(key)
    hit_ms = round((time.perf_counter() - t_hit0) * 1000, 2)
    result = dict(entry["result"])

    # fresh audit event: same content, new id/timestamp, cached marker
    new_audit = dict(entry["audit"])
    new_audit["audit_id"] = uuid.uuid4().hex[:12]
    new_audit["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    new_audit["cached"] = True
    new_audit.pop("record_hash", None)
    if settings.AUDIT_ENABLED:
        audit_store.append(new_audit)

    result["cached"] = True
    result["cached_execution_ms"] = hit_ms
    result["cached_original_execution_ms"] = result.get("execution_time_ms")
    result["execution_time_ms"] = hit_ms
    result["traversal"] = {**result["traversal"], "audit_id": new_audit["audit_id"]}
    cache_hits_total.inc()
    return result


def run_query(driver, query: str, max_hops: int | None = None,
              token_budget: int | None = None, reranker_mode: str | None = None,
              answer_mode: str | None = None,
              identity: UserIdentity | None = None) -> dict:
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
    })
    if key is not None and query_cache.get(key) is not None:
        return _serve_cache_hit(key, query)
    if key is not None:
        cache_misses_total.inc()

    ctx = _prepare(driver, query, max_hops, token_budget, reranker_mode, identity)
    answer = generate_answer(query, ctx["pruned"], mode=answer_mode)
    result = _finalize(ctx, answer)

    if key is not None:
        query_cache.put(key, {"result": result, "audit": ctx["_audit_record"]})
    return result


def stream_query(driver, query: str, max_hops: int | None = None,
                 token_budget: int | None = None, reranker_mode: str | None = None,
                 answer_mode: str | None = None,
                 identity: UserIdentity | None = None):
    """The query pipeline as an event generator (v2 — SSE endpoint + live UI).

    Event sequence:
    * ``{"type": "meta", ...}`` — retrieval stats + streaming flag
    * ``{"type": "delta", "text"}`` — answer tokens (live stream, or one
      buffered delta when PII masking is active for this caller)
    * ``{"type": "done", "result": {...}}`` — the full result (same shape as
      ``run_query``) after the audit record is written
    * ``{"type": "blocked", "result": {...}}`` — guardrail refusal instead of
      an answer

    Auto mode never raises; ``llm`` mode propagates provider failures.
    """
    max_hops = max_hops or settings.MAX_HOPS
    token_budget = token_budget or settings.MAX_TOKENS
    ctx = _prepare(driver, query, max_hops, token_budget, reranker_mode, identity)
    pruned = ctx["pruned"]
    pii_policy = ctx["pii_policy"]

    yield {
        "type": "meta",
        "streaming": not pii_policy.active,
        "retrieval": {
            "seeds": [s["id"] for s in ctx["subgraph"]["seeds"]],
            "node_count": ctx["subgraph"]["node_count"],
            "edge_count": ctx["subgraph"]["edge_count"],
            "baseline_tokens": ctx["baseline_tokens"],
            "pruned_tokens": pruned["tokens"],
        },
        "reranker": ctx["reranker"].name,
    }

    # input guardrail before any token leaves (output checks still run at the end)
    from graphrag.guardrails import scan_query
    input_guard = scan_query(query) if settings.GUARDRAILS_ENABLED else None
    if input_guard is not None and input_guard.blocked:
        blocked = {"answer": _BLOCKED_ANSWER, "mode": "blocked", "model": None,
                   "fallback_reason": "guardrail: " + ", ".join(input_guard.injection_hits)}
        yield {"type": "blocked", "result": _finalize(ctx, blocked)}
        return

    use_live = not pii_policy.active
    if use_live:
        answer_dict: dict | None = None
        for ev in stream_answer(query, pruned, mode=answer_mode):
            if ev["type"] == "delta":
                yield {"type": "delta", "text": ev["text"]}
            else:
                answer_dict = {
                    "answer": ev["answer"], "mode": ev["mode"],
                    "model": ev.get("model"), "provider": ev.get("provider"),
                    "usage": ev.get("usage"), "cost_usd": ev.get("cost_usd"),
                    "fallback_reason": ev.get("fallback_reason"),
                }
        if answer_dict is None:  # pragma: no cover - stream_answer always ends with done
            answer_dict = generate_answer(query, pruned, mode=answer_mode)
    else:
        # PII masking active: buffered path so nothing sensitive streams
        answer_dict = generate_answer(query, pruned, mode=answer_mode)
        if pii_policy.active:
            answer_dict = {**answer_dict,
                           "answer": scrub_answer(answer_dict["answer"], pii_policy)}
        yield {"type": "delta", "text": answer_dict["answer"]}

    yield {"type": "done", "result": _finalize(ctx, answer_dict)}
