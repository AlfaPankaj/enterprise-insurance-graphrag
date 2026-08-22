"""Multi-hop sub-graph retrieval from Neo4j (Phase 3, Shot 2 input).

Pipeline:

  1. **Seed detection** — entity ids mentioned in the query (``POL-0005``,
     ``CLM-0003``, ...) plus keyword matches over text properties
     (names, causes, statuses, ...).
  2. **BFS expansion** — expand ``max_hops`` relationship hops from the seeds,
     collecting every touched node and edge (deduplicated).
  3. **Serialization** — turn the sub-graph into LLM-ready text lines
     (``[Policy] POL-0005 status=ACTIVE ...`` + ``(A)-[:TYPE]->(B)``), which is
     what gets token-counted and pruned.

``retrieve_subgraph(driver, query, max_hops)`` returns::

    {"query", "seeds", "nodes": [{id, label, props}], "edges": [{source, type, target}]}
"""

from __future__ import annotations

import re

from graphrag.config import settings

# ---------------------------------------------------------------------------
# v2 tenant isolation (WS-B, G4)
# ---------------------------------------------------------------------------

def tenant_predicate(node_var: str = "n", param: str = "$tenant") -> str:
    """Cypher predicate scoping ``node_var`` to one tenant.

    ``$tenant IS NULL`` keeps unscoped behaviour bit-identical (v1 tests and
    benchmarks); when a tenant id is passed, only nodes carrying that
    ``tenant_id`` property are visible.
    """
    return f"({param} IS NULL OR coalesce({node_var}.tenant_id, '') = {param})"


def tenant_active(tenant_id: str | None) -> bool:
    """True when scoping should be applied (mode on AND a tenant is known)."""
    return bool(tenant_id) and settings.TENANT_MODE == "column"

# Entity ids used across the synthetic dataset (matches docs/graph_schema.md).
ENTITY_ID_RE = re.compile(r"\b((?:POL|CLM|PH|END|FRD|INV)-\d{3,})\b")
# Schema nouns are stopwords for KEYWORD seeding: they match generic prop
# values everywhere ("claim" hits every fraud reason, "clm" hits every claim
# number). Entity ids and value tokens (names, causes, statuses) do the real
# seed discovery; the reranker's answer-type prior handles schema nouns.
# Prepositions/auxiliary verbs are excluded too — they carry no retrieval
# signal and would otherwise dilute BM25 term weighting in the lexical backend.
STOPWORDS = {
    # determiners & pronouns
    "a", "an", "the", "this", "that", "these", "those", "it", "its",
    "their", "they", "them", "his", "her", "he", "she", "we", "our",
    "all", "any", "each", "every", "some", "other", "another",
    # auxiliaries & copulas
    "am", "are", "be", "been", "being", "can", "could", "did", "do",
    "does", "had", "has", "have", "is", "may", "might", "must",
    "shall", "should", "was", "were", "will", "would",
    # interrogatives
    "what", "which", "who", "whom", "whose", "when", "where", "why",
    "how", "howmany", "howmuch",
    # prepositions
    "about", "above", "across", "after", "against", "along", "among",
    "around", "at", "before", "behind", "below", "beneath", "beside",
    "between", "beyond", "by", "down", "during", "except", "for",
    "from", "in", "into", "near", "of", "off", "on", "onto", "out",
    "over", "past", "since", "through", "to", "toward", "under",
    "until", "up", "upon", "with", "within", "without",
    # query scaffolding
    "not", "no", "yes", "please", "tell", "give", "list", "show",
    "find", "get", "me", "like", "want", "need", "related", "relating",
    # schema nouns (see comment above)
    "claim", "claims", "policy", "policies", "coverage", "coverages",
    "endorsement", "endorsements", "fraud", "flag", "flags", "insurance",
    "investigator", "investigators", "policyholder", "holder", "status",
    "type", "premium", "deductible", "amount", "annual", "risk", "score",
    "number", "numbers", "id", "ids",
}

# Text properties searched for query keywords (all labels flattened).
# occupation is included for the real datasets (insurance_dataset.csv has no
# name column — "claims from doctors" must still seed).
_KEYWORD_PROPS = sorted({
    "name", "address", "email", "policy_number", "type", "status",
    "claim_number", "cause", "reason", "severity", "endorsement_number",
    "role", "category", "occupation",
})

# Numeric props + the label that owns them, for amount/threshold keywords
# ("claims over $100,000" -> Claim.amount >= 100000).
_NUMERIC_PROPS: list[tuple[str, str]] = [
    ("amount", "Claim"), ("limit", "Coverage"), ("premium", "Policy"),
    ("deductible", "Policy"), ("risk_score", "Policyholder"),
    ("confidence", "FraudFlag"),
]

# When the query NAMES a numeric schema noun ("premium", "deductible", ...),
# the threshold scan must be restricted to that prop's label — otherwise
# "premium over $5,000" would seed any Coverage whose *limit* exceeds $5k.
# Keys are query words (checked whole-word, lowercase); the schema nouns are
# stopwords for keyword seeding, so this is the only place they signal intent.
_PROP_FOCUS: dict[str, list[tuple[str, str]]] = {
    "premium": [("premium", "Policy")],
    "deductible": [("deductible", "Policy"), ("deductible", "Coverage")],
    "amount": [("amount", "Claim")],
    "limit": [("limit", "Coverage")],
    "risk": [("risk_score", "Policyholder")],
    "confidence": [("confidence", "FraudFlag")],
}


def _numeric_prop_focus(query: str) -> list[tuple[str, str]] | None:
    """Narrow the threshold scan to the numeric prop the query names."""
    words = re.findall(r"[a-z]+", query.lower())
    for word, pairs in _PROP_FOCUS.items():
        if word in words:
            return pairs
    # no explicit prop word: fall back by answer type ("claims over $X" -> Claim)
    if "claim" in words or "claims" in words:
        return [("amount", "Claim")]
    if "coverage" in words or "coverages" in words:
        return [("limit", "Coverage")]
    return None


def _singular(tok: str) -> str:
    """Light plural stripping for keyword seeding ("doctors" -> "doctor")."""
    if tok.endswith("ies") and len(tok) > 4:       # policies -> policy
        return tok[:-3] + "y"
    if tok.endswith("es") and len(tok) > 4:        # losses -> loss
        return tok[:-2]
    if tok.endswith("s") and len(tok) > 3 \
            and not tok.endswith(("ss", "is")):    # doctors -> doctor
        return tok[:-1]
    return tok

_THRESHOLD_ABOVE = ("over", "above", "more", "exceeds", "exceeding", "greater", "higher")
_THRESHOLD_BELOW = ("under", "below", "less", "lower")

_NODE_TEXT_PROPS = {
    "Policyholder": ["name", "risk_score"],
    "Policy": ["policy_number", "type", "status", "premium", "deductible",
               "start_date", "end_date"],
    "Claim": ["claim_number", "status", "amount", "date", "cause"],
    "FraudFlag": ["severity", "confidence", "reason", "created_by"],
    "Endorsement": ["endorsement_number", "type", "effective_date",
                    "premium_adjustment"],
    "Investigator": ["name", "role", "email"],
    "Coverage": ["code", "category", "limit", "deductible"],
}

# Natural-language descriptor per label — helps neural AND lexical scorers
# connect query words ("coverage", "fraud flag") to the node text.
_NODE_KIND = {
    "Policyholder": "policyholder",
    "Policy": "insurance policy",
    "Claim": "insurance claim",
    "FraudFlag": "fraud flag",
    "Endorsement": "policy endorsement",
    "Investigator": "claims investigator",
    "Coverage": "coverage",
}


def query_tokens(query: str) -> list[str]:
    """Lowercased, stopword-free tokens of a natural-language query."""
    words = re.findall(r"[a-zA-Z]+", query.lower())
    return [w for w in words if w not in STOPWORDS]


def extract_seed_ids(query: str) -> list[str]:
    """Entity ids explicitly mentioned in the query (e.g. ``CLM-0003``)."""
    return list(dict.fromkeys(ENTITY_ID_RE.findall(query)))


# ---------------------------------------------------------------------------
# seed detection
# ---------------------------------------------------------------------------

def _keyword_seeds(session, tokens: list[str], limit: int = 5,
                   tenant_id: str | None = None) -> list[dict]:
    """Global keyword scan — used when the query names no entity id."""
    if not tokens:
        return []
    tokens = [_singular(t) for t in tokens]  # "doctors" must hit occupation "Doctor"
    tp = tenant_predicate("n")
    rows = session.run(
        f"""
        UNWIND $tokens AS tok
        UNWIND $props AS prop
        MATCH (n)
        // the (:Dataset) metadata marker must never be keyword-seeded
        WHERE NOT 'Dataset' IN labels(n)
          AND {tp}
          AND n[prop] IS NOT NULL AND toLower(toString(n[prop])) CONTAINS toLower(tok)
        WITH n, labels(n) AS labels, count(*) AS hits
        ORDER BY hits DESC
        RETURN labels, n.id AS id
        LIMIT $limit
        """,
        tokens=tokens, props=_KEYWORD_PROPS, limit=limit,
        tenant=tenant_id if tenant_active(tenant_id) else None,
    ).data()
    return [{"id": r["id"], "label": r["labels"][0], "kind": "keyword"} for r in rows]


def _numeric_seeds_global(session, numbers: list[int], direction: int, limit: int = 5,
                          pairs: list[tuple[str, str]] | None = None,
                          tenant_id: str | None = None) -> list[dict]:
    """Global scan for amount/threshold keywords ("claims over $100,000"),
    restricted to the label that owns each numeric prop (or the focused
    prop when the query names one — e.g. "premium over $5,000")."""
    if not numbers:
        return []
    op = ">=" if direction >= 0 else "<="
    threshold = float(max(numbers)) if direction >= 0 else float(min(numbers))
    pairs = pairs or [(prop, label) for prop, label in _NUMERIC_PROPS]
    tp = tenant_predicate("n")
    rows = session.run(
        f"""
        UNWIND $pairs AS pair
        MATCH (n)
        WHERE pair[1] IN labels(n) AND n[pair[0]] IS NOT NULL
          AND toFloat(n[pair[0]]) {op} $threshold
          AND {tp}
        RETURN labels(n) AS labels, n.id AS id
        LIMIT $limit
        """,
        pairs=pairs, threshold=threshold, limit=limit,
        tenant=tenant_id if tenant_active(tenant_id) else None,
    ).data()
    return [{"id": r["id"], "label": r["labels"][0], "kind": "keyword"} for r in rows]


def _neighborhood_keyword_seeds(nodes: dict[str, dict], tokens: list[str],
                                numbers: list[int] | None = None,
                                direction: int = 0, limit: int = 5,
                                pairs: list[tuple[str, str]] | None = None) -> list[dict]:
    """Keyword matches restricted to an id-anchored neighborhood — filters like
    "paid" or "fire damage" must apply within the anchor's context, not seed
    random nodes from across the graph. Numeric thresholds ("over $100,000")
    are matched against the owning label's numeric props."""
    scored: list[tuple[dict, int]] = []
    numbers = numbers or []
    for node in nodes.values():
        text = " ".join(str(v) for v in node["props"].values()).lower()
        props = {**node["props"], "_label": node["label"]}
        hits = sum(1 for t in tokens if _singular(t) in text)
        hits += _numeric_hits(props, numbers, direction, pairs)
        if hits:
            scored.append((node, hits))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [{"id": n["id"], "label": n["label"], "kind": "keyword"}
            for n, _ in scored[:limit]]


def _id_seeds(session, query: str, tenant_id: str | None = None) -> list[dict]:
    seeds: list[dict] = []
    tp = tenant_predicate("n")
    for eid in extract_seed_ids(query):
        row = session.run(
            f"MATCH (n {{id: $id}}) WHERE {tp} "
            "RETURN labels(n) AS labels, n.id AS id",
            id=eid,
            tenant=tenant_id if tenant_active(tenant_id) else None,
        ).single()
        if row:
            seeds.append({"id": eid, "label": row["labels"][0], "kind": "id"})
    return seeds


def _value_tokens(query: str) -> list[str]:
    """Keyword tokens, excluding schema nouns and id substrings (e.g. "clm")."""
    seed_ids = extract_seed_ids(query)
    return [t for t in query_tokens(query)
            if not any(t in eid.lower() for eid in seed_ids)]


def _numeric_tokens(query: str) -> list[int]:
    """Money/amount numbers in the query ("$100,000" -> [100000])."""
    out: list[int] = []
    for m in re.findall(r"\$?\d[\d,]*", query):
        digits = m.replace("$", "").replace(",", "")
        if digits.isdigit():
            out.append(int(digits))
    return out


def _threshold_direction(query: str) -> int:
    """+1 for "over/above/more than", -1 for "under/below/less", else 0."""
    q = query.lower()
    if any(w in q for w in _THRESHOLD_ABOVE):
        return 1
    if any(w in q for w in _THRESHOLD_BELOW):
        return -1
    return 0


def _numeric_hits(props: dict, numbers: list[int], direction: int,
                  pairs: list[tuple[str, str]] | None = None) -> int:
    """Count (prop, number) matches, restricted to the prop's owning label."""
    if not numbers:
        return 0
    hits = 0
    for prop, label in pairs or _NUMERIC_PROPS:
        if props.get("_label") != label:  # only the label that owns this prop
            continue
        value = props.get(prop)
        if not isinstance(value, (int, float)):
            continue
        if direction > 0 and value >= max(numbers):
            hits += 1
        elif direction < 0 and value <= min(numbers):
            hits += 1
        elif direction == 0 and int(value) in numbers:
            hits += 1
    return hits


# ---------------------------------------------------------------------------
# BFS expansion
# ---------------------------------------------------------------------------

def _expand_both(session, frontier: list[str],
                 tenant_id: str | None = None) -> tuple[list[str], list[dict]]:
    """One hop in both directions: (new neighbor ids, deduped edges)."""
    edges: list[dict] = []
    neighbors: list[str] = []
    tn = tenant_id if tenant_active(tenant_id) else None
    tpn, tpm = tenant_predicate("n"), tenant_predicate("m")
    rows = session.run(
        f"""
        MATCH (n)-[r]->(m)
        WHERE n.id IN $frontier AND {tpn} AND {tpm}
        RETURN n.id AS src, type(r) AS type, m.id AS dst
        """,
        frontier=frontier, tenant=tn,
    )
    for r in rows:
        edges.append({"source": r["src"], "type": r["type"], "target": r["dst"]})
        neighbors.append(r["dst"])
    rows = session.run(
        f"""
        MATCH (n)<-[r]-(m)
        WHERE n.id IN $frontier AND {tpn} AND {tpm}
        RETURN m.id AS src, type(r) AS type, n.id AS dst
        """,
        frontier=frontier, tenant=tn,
    )
    for r in rows:
        edges.append({"source": r["src"], "type": r["type"], "target": r["dst"]})
        neighbors.append(r["src"])
    return neighbors, edges


def _fetch_nodes(session, ids: list[str],
                 tenant_id: str | None = None) -> dict[str, dict]:
    """id -> {id, label, props} for every requested node id.

    PII-classified props stored encrypted (``PII_ENCRYPTION_KEY``) are
    decrypted here — the read path of the at-rest encryption scheme.
    """
    try:
        from graphrag.pii import decrypt_node, encryption_enabled
    except ImportError:  # pragma: no cover - graphrag package always present
        encryption_enabled = lambda: False  # noqa: E731
        decrypt_node = lambda n: n  # noqa: E731
    nodes: dict[str, dict] = {}
    tp = tenant_predicate("n")
    for row in session.run(
        f"MATCH (n) WHERE n.id IN $ids AND {tp} RETURN labels(n) AS labels, n",
        ids=ids,
        tenant=tenant_id if tenant_active(tenant_id) else None,
    ):
        node = row["n"]
        nid = node["id"]
        props = {k: v for k, v in dict(node).items()
                 if k != "id" and not isinstance(v, (dict, list))}
        fetched = {"id": nid, "label": row["labels"][0], "props": props}
        if encryption_enabled():
            fetched = decrypt_node(fetched)
        nodes[nid] = fetched
    return nodes


def _expand_graph(session, seed_ids: list[str], max_hops: int,
                  tenant_id: str | None = None) -> tuple[set[str], list[dict]]:
    """BFS from seed ids: (visited node ids, deduped edges)."""
    frontier = list(seed_ids)
    visited: set[str] = set(frontier)
    edges: list[dict] = []
    seen_edges: set[tuple] = set()
    for _ in range(max_hops):
        if not frontier:
            break
        neighbors, hop_edges = _expand_both(session, frontier, tenant_id=tenant_id)
        for e in hop_edges:
            key = (e["source"], e["type"], e["target"])
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append(e)
        new_frontier = [n for n in neighbors if n not in visited]
        visited.update(new_frontier)
        frontier = new_frontier
    return visited, edges


def retrieve_subgraph(driver, query: str, max_hops: int | None = None,
                      tenant_id: str | None = None,
                      vector_store=None) -> dict:
    """Retrieve the multi-hop sub-graph reachable from the query's seed nodes.

    If the query names an entity id (precise anchor), keyword filters are
    restricted to that anchor's neighborhood — e.g. "paid claims under policy
    POL-0084" matches only POL-0084's paid claims. Pure keyword queries scan
    the graph globally.

    ``tenant_id`` (with ``settings.TENANT_MODE="column"``) scopes every
    Cypher statement to nodes carrying that ``tenant_id`` — the v2 tenant
    isolation guard. ``None`` = unscoped (v1 behavior).

    ``vector_store`` (v2 hybrid retrieval): when id/keyword/numeric seeding
    finds NOTHING, the query is embedded and the store supplies semantic
    seeds (``kind="semantic"``) — the paraphrase fallback. Omit for pure
    v1 behavior.
    """
    max_hops = max_hops or settings.MAX_HOPS
    with driver.session() as session:
        id_seeds = _id_seeds(session, query, tenant_id=tenant_id)
        tokens = _value_tokens(query)
        numbers = _numeric_tokens(query)
        direction = _threshold_direction(query)
        prop_focus = _numeric_prop_focus(query)
        if id_seeds:
            seeds = list(id_seeds)
            seed_ids = {s["id"] for s in seeds}
            visited, edges = _expand_graph(session, [s["id"] for s in seeds],
                                           max_hops, tenant_id=tenant_id)
            nodes = _fetch_nodes(session, list(visited), tenant_id=tenant_id)
            for kw in _neighborhood_keyword_seeds(nodes, tokens, numbers, direction,
                                                  pairs=prop_focus):
                if kw["id"] not in visited or kw["id"] in seed_ids:
                    continue  # already inside the neighborhood / already an id seed
                seeds.append(kw)  # flagged as keyword seed for provenance
        else:
            seeds = _keyword_seeds(session, tokens, tenant_id=tenant_id)
            if direction == 0:
                seeds += _numeric_seeds_global(session, numbers, 0, pairs=prop_focus,
                                               tenant_id=tenant_id)
            else:
                seeds += _numeric_seeds_global(session, numbers, direction, pairs=prop_focus,
                                               tenant_id=tenant_id)
            seeds = list({s["id"]: s for s in seeds}.values())  # dedup by id
            # v2 semantic fallback: paraphrase queries with no lexical signal
            if not seeds and vector_store is not None:
                from graphrag.vector_store import semantic_seeds
                seeds = semantic_seeds(session, query, vector_store, k=3)
            visited, edges = _expand_graph(session, [s["id"] for s in seeds],
                                           max_hops, tenant_id=tenant_id)
            nodes = _fetch_nodes(session, list(visited), tenant_id=tenant_id)

        node_list = [nodes[nid] for nid in sorted(visited)]

    return {
        "query": query,
        "seeds": seeds,
        "nodes": node_list,
        "edges": edges,
        "node_count": len(node_list),
        "edge_count": len(edges),
    }


# ---------------------------------------------------------------------------
# serialization (LLM context)
# ---------------------------------------------------------------------------

def serialize_node(node: dict) -> str:
    """One node as a compact text line, e.g.::

        [Policy] POL-0005 policy_number=CP-2023-0005 status=ACTIVE premium=54036.0
    """
    label = node["label"]
    props = node.get("props", {})
    keys = [k for k in _NODE_TEXT_PROPS.get(label, []) if k in props]
    extras = [k for k in props if k not in keys]  # any remaining/join props
    parts = [f"{k}={props[k]}" for k in keys] + [f"{k}={props[k]}" for k in extras]
    kind = _NODE_KIND.get(label, label.lower())
    return f"[{label}] {node['id']} {kind}" + (("; " + "; ".join(parts)) if parts else "")


def serialize_subgraph(subgraph: dict) -> str:
    """The full sub-graph as LLM context text (nodes + edges, deduplicated)."""
    lines = [f"QUERY: {subgraph['query']}"]
    for node in subgraph["nodes"]:
        lines.append(serialize_node(node))
    seen: set[tuple] = set()
    for e in subgraph["edges"]:
        key = (e["source"], e["type"], e["target"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"({e['source']})-[:{e['type']}]->({e['target']})")
    return "\n".join(lines)
