"""Adaptive context pruning — keep only the highest-relevance nodes that fit
the token budget (Shot 2).

``prune_context(ranked, token_budget)`` greedily accepts nodes best-first while
the serialized context stays under budget, then filters the edge list down to
edges whose endpoints were both kept. Returns::

    {"nodes", "edges", "text", "tokens", "budget",
     "dropped": [ids], "kept": [ids], "node_count"}
"""

from __future__ import annotations

from graphrag.config import settings
from graphrag.graph_retriever import serialize_node
from graphrag.token_counter import count_tokens


def _render(kept: list[dict], kept_edges: list[dict]) -> tuple[str, int]:
    """Serialized pruned context — node lines + kept edges, no header (the
    caller owns the QUERY line, so the budget covers exactly this context)."""
    lines = [serialize_node(n) for n in kept]
    seen: set[tuple] = set()
    for e in kept_edges:
        key = (e["source"], e["type"], e["target"])
        if key not in seen:
            seen.add(key)
            lines.append(f"({e['source']})-[:{e['type']}]->({e['target']})")
    text = "\n".join(lines)
    return text, count_tokens(text)


def prune_context(ranked_nodes: list[tuple[dict, float]] | list[dict],
                  token_budget: int | None = None,
                  edges: list[dict] | None = None,
                  protected_ids: list[str] | None = None) -> dict:
    """Keep nodes best-first so the *serialized* context fits the token budget.

    ``protected_ids`` (e.g. the seed nodes + their direct neighbors) are always
    kept — the local neighborhood of the query's entities is authoritative and
    must not be pruned away. Everything else is greedy-selected by rank, then
    the lowest-ranked kept nodes are trimmed until the text fits the budget.

    NOTE: protected nodes are never trimmed — if the protected set alone exceeds
    the budget, the returned ``tokens`` can exceed ``budget`` (callers keep the
    protected set small enough that this does not happen in practice).
    """
    token_budget = token_budget or settings.MAX_TOKENS
    protected = set(protected_ids or [])
    # accept a bare node list (scores unknown -> keep original order)
    ranked = ranked_nodes if ranked_nodes and isinstance(ranked_nodes[0], tuple) \
        else [(n, 0.0) for n in ranked_nodes]

    kept: list[dict] = []          # keep order: protected first, then by rank
    kept_ids: set[str] = set()
    dropped: list[str] = []
    used = 0

    for node, _score in ranked:
        if node["id"] not in protected:
            continue
        kept.append(node)
        kept_ids.add(node["id"])
        used += count_tokens(serialize_node(node))
    protected_count = len(kept)

    for node, _score in ranked:
        if node["id"] in kept_ids:
            continue
        cost = count_tokens(serialize_node(node))
        if used + cost <= token_budget or (not kept and token_budget > 0):
            kept.append(node)
            kept_ids.add(node["id"])
            used += cost
        else:
            dropped.append(node["id"])

    kept_edges = []
    if edges:
        for e in edges:
            if e["source"] in kept_ids and e["target"] in kept_ids:
                kept_edges.append(e)

    text, tokens = _render(kept, kept_edges)
    # exact enforcement: edges may have pushed us over the budget — trim the
    # lowest-ranked kept nodes until it fits (never trim protected nodes)
    while tokens > token_budget and len(kept) > protected_count:
        removed = kept.pop()  # kept is rank-ordered; pop the least relevant
        kept_ids.discard(removed["id"])
        dropped.append(removed["id"])
        kept_edges = [e for e in kept_edges
                      if e["source"] in kept_ids and e["target"] in kept_ids]
        text, tokens = _render(kept, kept_edges)

    return {
        "nodes": kept,
        "edges": kept_edges,
        "text": text,
        "tokens": tokens,
        "budget": token_budget,
        "kept": [n["id"] for n in kept],
        "dropped": dropped,
        "node_count": len(kept),
        "dropped_count": len(dropped),
    }
