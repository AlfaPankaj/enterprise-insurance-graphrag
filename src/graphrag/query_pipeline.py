"""Query pipeline — retrieve → re-rank → prune → answer (Shot 2).

``run_query(driver, query)`` executes the full token-optimized retrieval flow
and returns baseline vs optimized token counts, savings %, and an answer — an
Ollama LLM answer when configured (``ANSWER_MODE=auto``) with a deterministic
extractive fallback. Phase 4 adds full lineage logging.
"""

from __future__ import annotations

import time

from graphrag.answer_generator import generate_answer
from graphrag.config import settings
from graphrag.context_pruner import prune_context
from graphrag.graph_retriever import retrieve_subgraph, serialize_subgraph
from graphrag.reranker import _label_prior_hits, make_reranker
from graphrag.token_counter import count_tokens
from graphrag.traversal_logger import audit_store, build_audit_record


def run_query(driver, query: str, max_hops: int | None = None,
              token_budget: int | None = None, reranker_mode: str | None = None,
              answer_mode: str | None = None) -> dict:
    """Full token-optimized retrieval pipeline for a natural-language query.

    Returns the answer + token savings, and — for explainability (Shot 3) —
    the full traversal lineage (nodes/edges visited, Cypher used, per-stage
    timings). Every call is appended to the audit trail store.

    ``answer_mode`` in {"extractive", "auto", "llm"} — default
    ``settings.ANSWER_MODE`` (extractive = deterministic, no Ollama needed;
    auto = try Ollama, fall back to extractive; llm = require Ollama).
    """
    max_hops = max_hops or settings.MAX_HOPS
    token_budget = token_budget or settings.MAX_TOKENS
    t0 = time.perf_counter()

    subgraph = retrieve_subgraph(driver, query, max_hops)
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

    answer = generate_answer(query, pruned, mode=answer_mode)
    t4 = time.perf_counter()

    tokens = {
        "before": baseline_tokens,
        "after": pruned["tokens"],
        "savings_percent": max(savings, 0.0),
    }
    timings = {
        "retrieval_ms": round((t1 - t0) * 1000, 2),
        "rerank_ms": round((t2 - t1) * 1000, 2),
        "prune_ms": round((t3 - t2) * 1000, 2),
        "answer_ms": round((t4 - t3) * 1000, 2),
        "total_ms": round((t4 - t0) * 1000, 2),
    }

    # --- explainability (Shot 3): build + persist the audit record ---
    audit_record = build_audit_record(
        query=query, subgraph=subgraph, ranked=ranked, pruned=pruned,
        tokens=tokens, answer=answer["answer"], answer_mode=answer["mode"],
        answer_model=answer.get("model"), reranker=reranker.name, max_hops=max_hops,
        t0=t0, t1=t1, t2=t2, t3=t3, t4=t4,
    )
    if settings.AUDIT_ENABLED:
        audit_store.append(audit_record)

    traversal = {
        "audit_id": audit_record["audit_id"],
        "nodes_visited": audit_record["traversal"]["nodes_visited"],
        "edges_traversed": audit_record["traversal"]["edges_traversed"],
        "paths": audit_record["traversal"]["paths"],
        "cypher": audit_record["cypher"],
        "timings_ms": timings,
    }

    return {
        "query": query,
        "answer": answer["answer"],
        "answer_mode": answer["mode"],
        "answer_model": answer.get("model"),
        "answer_fallback": answer.get("fallback_reason"),
        "reranker": reranker.name,
        "tokens": tokens,
        "retrieval": {
            "seeds": [s["id"] for s in subgraph["seeds"]],
            "node_count": subgraph["node_count"],
            "edge_count": subgraph["edge_count"],
        },
        "pruned": {
            "node_count": pruned["node_count"],
            "kept": pruned["kept"],
            "dropped": pruned["dropped"],
            "dropped_count": pruned["dropped_count"],
            "budget": token_budget,
        },
        "traversal": traversal,
        "execution_time_ms": round((t4 - t0) * 1000, 2),
    }
