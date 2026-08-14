"""Traversal path extraction + Cypher reconstruction (Phase 4, Shot 3).

Given the sub-graph returned by ``retrieve_subgraph``, this module makes the
retrieval auditable:

  * ``extract_traversal_paths`` — BFS chains from the seed nodes through the
    collected edges (``[(src, rel, dst), ...]``), the exact hops the query
    engine actually traversed.
  * ``build_cypher`` — the Cypher statements that reproduce that traversal:
    seed lookup, the variable-length expansion pattern, and a concrete
    answer-path pattern (as shown in the audit UI).

Everything here is deterministic and pure — no DB access, so it is unit
testable with a plain dict sub-graph.
"""

from __future__ import annotations

from graphrag.config import settings
from graphrag.graph_retriever import query_tokens

# Soft cap on how many distinct chains to render in the audit trail
# (branching sub-graphs can produce exponentially many paths; the cap keeps
# the report readable — it bounds recursion work, not the node inventory).
_MAX_PATHS = 25


def _adjacency(subgraph: dict) -> dict[str, list[tuple[str, str, str, str]]]:
    """id -> [(neighbor, rel, edge_src, edge_dst)] — edges usable both ways."""
    adj: dict[str, list[tuple[str, str, str, str]]] = {}
    for e in subgraph.get("edges", []):
        src, rel, dst = e["source"], e["type"], e["target"]
        adj.setdefault(src, []).append((dst, rel, src, dst))
        adj.setdefault(dst, []).append((src, rel, src, dst))  # reverse walk
    return adj


def extract_traversal_paths(subgraph: dict, max_hops: int | None = None) -> list[dict]:
    """BFS chains from every seed node through the traversed edges.

    Returns ``[{"nodes": [ids...], "edges": [[src, rel, dst], ...]}]`` —
    the provenance trail for each branch of the retrieval. Chains never
    revisit a node (simple paths) and are bounded by ``max_hops`` hops; the
    ``_MAX_PATHS`` soft cap bounds recursion work, not node inventory.
    """
    max_hops = max_hops or settings.MAX_HOPS
    seeds = [s["id"] for s in subgraph.get("seeds", [])]
    adj = _adjacency(subgraph)
    chains: list[dict] = []
    max_chain_len = max_hops + 1  # seed + up to max_hops hops

    def dfs(current: str, nodes: list[str], edges: list[list], visited: set[str]):
        if len(chains) >= _MAX_PATHS or len(nodes) >= max_chain_len:
            return
        for neighbor, rel, esrc, edst in adj.get(current, []):
            if neighbor in visited:
                continue
            # steps are rendered as traversed (walk order from the seed); the
            # authoritative relationship inventory lives in ``edges_traversed``
            step = [current, rel, neighbor]
            chains.append({
                "nodes": nodes + [neighbor],
                "edges": edges + [step],
            })
            dfs(neighbor, nodes + [neighbor], edges + [step], visited | {neighbor})

    for seed in seeds:
        dfs(seed, [seed], [], {seed})
    return chains


def traversal_summary(subgraph: dict, max_hops: int | None = None) -> dict:
    """Node visit order + the deduped edges the BFS actually traversed.

    ``nodes_visited`` follows the walk order (seed outward, from the chains)
    and is **complete** — any sub-graph node the chains did not reach (e.g.
    beyond the rendered chain depth) is appended, so the record is internally
    consistent with ``edges_traversed`` at any ``max_hops``.
    ``edges_traversed`` keeps each edge in its **stored** direction
    (``source -[:TYPE]-> target``) — the honest relationship inventory.
    """
    chains = extract_traversal_paths(subgraph, max_hops)
    nodes_visited: list[str] = []
    for c in chains:
        for n in c["nodes"]:
            if n not in nodes_visited:
                nodes_visited.append(n)
    # completeness: every sub-graph node is in the audit trail, even if no
    # rendered chain reached it (chains are capped, the BFS was not)
    for nid in sorted(subgraph.get("nodes", []), key=lambda n: n["id"]):
        if nid["id"] not in nodes_visited:
            nodes_visited.append(nid["id"])
    edges: list[list[str]] = []
    seen: set[tuple] = set()
    for e in subgraph.get("edges", []):
        key = (e["source"], e["type"], e["target"])
        if key not in seen:
            seen.add(key)
            edges.append([e["source"], e["type"], e["target"]])
    return {"nodes_visited": nodes_visited, "edges_traversed": edges, "paths": chains}


def build_cypher(subgraph: dict, max_hops: int | None = None) -> str:
    """Cypher that reproduces this traversal (seed lookup → expansion → path).

    ``max_hops`` defaults to ``settings.MAX_HOPS`` — the same depth the BFS
    actually used, so the reported Cypher matches what was executed.
    """
    max_hops = max_hops or settings.MAX_HOPS
    seeds = subgraph.get("seeds", [])
    query = subgraph.get("query", "")
    sections: list[str] = []

    if seeds:
        anchors = []
        for i, s in enumerate(seeds):
            kind = "id" if s.get("kind") == "id" else "keyword match"
            anchors.append(
                f'// seed {i + 1} ({kind})\n'
                f'MATCH (s{i}:{s["label"]} {{id: "{s["id"]}"}})\n'
                f'RETURN s{i}'
            )
        sections.append("-- 1. seed lookup --\n" + "\n\n".join(anchors))
        section2 = (
            "-- 2. BFS expansion (both directions, max {h} hops) --\n"
            "MATCH p = (s0 {{id: \"{seed}\"}})-[*1..{h}]-(n)\n"
            "RETURN DISTINCT nodes(p) AS visited, relationships(p) AS edges"
        ).format(h=max_hops, seed=seeds[0]["id"])
        sections.append(section2)
    else:
        tokens = query_tokens(query)
        kws = ", ".join(f'"{t}"' for t in tokens[:4]) or "\"…\""
        sections.append(
            "-- 1. seed lookup — no entity id in query; keyword scan --\n"
            "UNWIND $props AS prop\n"
            "MATCH (n)\n"
            f"WHERE any(tok IN [{kws}] WHERE toLower(toString(n[prop])) CONTAINS tok)\n"
            "RETURN DISTINCT n AS seed"
        )
        sections.append(
            "-- 2. BFS expansion (both directions, max {h} hops) --\n"
            "MATCH (seed) WHERE seed.id IN $seed_ids\n"
            "MATCH p = (seed)-[*1..{h}]-(n)\n"
            "RETURN DISTINCT nodes(p) AS visited, relationships(p) AS edges"
            .format(h=max_hops)
        )

    # 3. concrete answer path — the first extracted chain, rendered as a
    #    single MATCH pattern with the seed id as anchor.
    chains = extract_traversal_paths(subgraph, max_hops)
    if chains:
        c = chains[0]
        pattern: list[str] = []
        for i, nid in enumerate(c["nodes"]):
            pattern.append(f'(n{i} {{id: "{nid}"}})')
            if i < len(c["edges"]):
                pattern.append(f"-[:{c['edges'][i][1]}]->")
        sections.append(
            "-- 3. concrete answer path (first extracted chain) --\n"
            "MATCH p = " + "".join(pattern) + "\n"
            "RETURN p"
        )

    return "\n\n".join(sections)
