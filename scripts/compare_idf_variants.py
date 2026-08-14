"""A/B: old exact-token IDF vs new variant-max IDF across benchmark queries.

Measures how the user's ``variant_df`` change affects ranking: order agreement,
score deltas, and whether ground-truth answer ids stay inside the kept set.
"""
import json
import math
import sys
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neo4j import GraphDatabase  # noqa: E402
from graphrag.config import settings  # noqa: E402
from graphrag.graph_retriever import (ENTITY_ID_RE, query_tokens,  # noqa: E402
                                      retrieve_subgraph, serialize_node)
from graphrag.reranker import _K1, _B, _prefix_variants  # noqa: E402

_TOKEN_RE = __import__("re").compile(r"[a-zA-Z]+")


def _bm25_idf(n_docs: int, df: int) -> float:
    return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


def _rank(nodes, query, idf_mode: str) -> list[tuple[str, float]]:
    """Replica of LexicalReranker.rank; idf_mode in {'exact', 'variant'}."""
    q_tokens = query_tokens(query)
    q_ids = set(__import__("re").findall(ENTITY_ID_RE, query.upper()))
    docs = [_TOKEN_RE.findall(serialize_node(n).lower()) for n in nodes]
    n_docs = len(docs)
    avgdl = (sum(len(d) for d in docs) / n_docs) or 1.0
    vocab = {t for d in docs for t in d}
    df = Counter(t for d in docs for t in set(d))
    variants = {qt: _prefix_variants(qt, vocab) for qt in q_tokens}

    out = []
    for node, tokens in zip(nodes, docs):
        tf = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for qt in q_tokens:
            if idf_mode == "variant":
                variant_df = max((df.get(t, 0) for t in variants[qt]), default=0)
            else:  # old behavior: exact query token's df only
                variant_df = df.get(qt, 0)
            idf = _bm25_idf(n_docs, variant_df)
            best = max((tf.get(t, 0) for t in variants[qt]), default=0)
            if not best:
                continue
            denom = best + _K1 * (1 - _B + _B * (doc_len / avgdl))
            score += idf * (best * (_K1 + 1)) / denom
        bonus = 5.0 if node["id"].upper() in q_ids else 0.0
        out.append((node["id"], round(score + bonus, 6)))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def main():
    driver = GraphDatabase.driver(settings.NEO4J_URI,
                                  auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    queries = json.loads(
        (ROOT / "data" / "benchmarks" / "benchmark_queries.json").read_text(encoding="utf-8")
    )

    total_order_changes = 0
    agree = 0
    avg_delta = 0.0
    for q in queries:
        sub = retrieve_subgraph(driver, q["query"], 2)
        old = _rank(sub["nodes"], q["query"], "exact")
        new = _rank(sub["nodes"], q["query"], "variant")
        old_ids = [i for i, _ in old]
        new_ids = [i for i, _ in new]
        # how many top-10 positions differ
        diffs = sum(1 for a, b in zip(old_ids[:10], new_ids[:10]) if a != b)
        total_order_changes += diffs
        # ground truth still inside kept set?
        expected = set(q["expected"])
        old_ok = expected.issubset(set(old_ids[:15]))
        new_ok = expected.issubset(set(new_ids[:15]))
        agree += old_ok == new_ok
        delta = sum(abs(b - a) for (_, a), (_, b) in zip(old, new))
        avg_delta += delta
        flag = "" if old_ok == new_ok else "  <-- BEHAVIOR CHANGE"
        print(f"{q['id']} top10 diffs={diffs:2d} | expected-in-old15={old_ok} "
              f"expected-in-new15={new_ok}{flag}")

    n = len(queries)
    print(f"\nqueries={n} | avg top-10 diffs/query={total_order_changes / n:.2f} "
          f"| truth-agreement unchanged={agree}/{n} | avg abs score delta={avg_delta / n:.4f}")
    driver.close()


if __name__ == "__main__":
    main()
