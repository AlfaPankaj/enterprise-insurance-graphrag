## complete picture — differences, similarities, and pros/cons
```
┌──────────────────┬───────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┐
│                  │ Lexical (BM25)                                        │ Cross-Encoder                                            │
├──────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
│ What it is       │ Pure-Python Okapi BM25 (k1=1.5, b=0.75,               │ Neural model — sentence-transformers CrossEncoder        │
│                  │ Robertson-Sparck Jones IDF with +1 smoothing) over    │ (ms-marco-MiniLM-L-6-v2) scoring (query, node_text)      │
│                  │ node texts                                            │ pairs                                                    │
├──────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
│ Scoring formula  │ score = BM25(Q, node_text) + 5.0 × (query names this  │ score = model(query, node_text) + 2.5 × (answer-type     │
│                  │ id)                                                   │ label prior)                                             │
├──────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
│ Dependencies     │ Zero — ~60 lines of stdlib math                       │ torch + sentence-transformers (~2–3 GB) + 90 MB model    │
│                  │                                                       │ download                                                 │
├──────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
│ Latency          │ ~31 ms per query                                      │ ~1,211 ms per query                                      │
│ (measured)       │                                                       │                                                          │
├──────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
│ Accuracy         │ 100%                                                  │ 100%                                                     │
│ (20-query        │                                                       │                                                          │
│ benchmark)       │                                                       │                                                          │
└──────────────────┴───────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────┘
```

## Similarities

1. Same interface — both implement  `rank(query, nodes) -> [(node, score)]` , sorted best-first. The pipeline, API, and dashboard treat them interchangeably; that's why the dashboard's dropdown can switch modes with zero downstream changes.
2. Same input & output contract — both rank the same retrieved sub-graph nodes, serialized through  `serialize_node()` , and both feed the identical token-budget pruner afterward.
3. Both get a "prior" on top — hybrid scoring. Lexical adds an entity-id bonus ( `CLM-0003`  named in the query → jumps to rank 1); cross-encoder adds an answer-type label prior (query says "coverages" → Coverage nodes boosted).
4. Both protect the same retrieval semantics — same stopwords, prefix handling, tokenization downstream of  `query_tokens()` .
5. Same purpose — both solve the same problem: decide which retrieved graph nodes are relevant enough to keep inside the token budget, so the answer doesn't get starved or diluted.
6. Same measured result on our data — both achieved 20/20 benchmark accuracy and ~18.5% average token savings. On this demo graph, they're functionally equivalent.

## Differences (the full list)

```
┌────────────────┬─────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┐
│ Dimension      │ Lexical (BM25)                                          │ Cross-Encoder                                            │
├────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
│ How it         │ Word/prefix matching only — "coverages" matches         │ Semantic — understands meaning, synonyms, paraphrase     │
│ understands    │ "coverage" via prefix expansion, but it has no semantic │ ("claim was rejected" ≈ "claim denied") with zero        │
│ language       │ understanding                                           │ lexical overlap                                          │
│ Model size /   │ ~0 (just code)                                          │ ~90 MB weights + torch runtime in RAM (several hundred   │
│ memory         │                                                         │ MB resident)                                             │
├────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
│ First-query    │ Instant                                                 │ ~20 s first run (download + load); cached per process    │
│ cold start     │                                                         │ afterward                                                │
├────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
│ Determinism    │ Fully deterministic — same input, same scores,          │ Near-deterministic but GPU/FP variability can shift      │
│                │ bit-identical                                           │ scores by tiny amounts                                   │
├────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
│ Interpretabili │ Transparent: you can trace exactly why a node scored    │ Black box: a float from a transformer — no per-term      │
│ ty             │ high (which term, what IDF, what tf)                    │ attribution                                              │
│ Language       │ Works on any text, but only token-overlap matching (no  │ Multilingual-capable; understands inflection, synonyms,  │
│ support        │ morphology beyond prefixes, no synonyms)                │ paraphrase                                               │
│ Typo           │ Weak — "clm-003" won't match "CLM-0003" unless          │ Strong — embeddings are robust to minor typos            │
│ robustness     │ prefix-covered                                          │                                                          │
├────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
│ Training data  │ None — the formula is fixed                             │ Pre-trained on 500K+ labeled (query, passage) relevance  │
│                │                                                         │ pairs (MS MARCO)                                         │
├────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
│ Query-time     │ O(nodes × query_terms) — microseconds per node          │ One forward pass through a transformer per node —        │
│ cost           │                                                         │ CPU-bound, scales linearly with node count but with a    │
│                │                                                         │ big constant                                             │
├────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
│ Failure mode   │ Can't fail — pure math on what's in front of it         │ Can fail: missing torch, offline download, OOM — and     │
│                │                                                         │ then falls back to lexical (built in)                    │
├────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
│ Where it       │ Exact ids, codes, amounts, technical insurance terms    │ Natural-language, fuzzy, paraphrased questions           │
│ shines         │                                                         │                                                          │
├────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
│ Score scale    │ Positive unbounded (typically 0–8 with the id bonus)    │ Logits roughly −5 to +5, calibrated for ranking not for  │
│                │                                                         │ absolute meaning                                         │
└────────────────┴─────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────┘
```

## Advantages / Disadvantages

### ✅ Lexical (BM25) advantages
```
- Zero dependencies, zero downloads — runs on any machine, in CI, on a Raspberry Pi, in an air-gapped environment
- ~40× faster (31 ms vs 1.2 s) — matters for interactive demos and high QPS
- Deterministic & auditable — you can show why a node ranked: "node X ranks #1 because the query names  CLM-0003  (+5.0) and it contains 'fraud' twice (idf 1.4)" — great for the audit-trail story you're pitching to EXL
- Predictable resource footprint — no GPU, no VRAM, no model cache
- Exact-match supremacy — entity ids, policy numbers, dollar thresholds are matched perfectly; a neural model can miss an exact code
- Boring = reliable — nothing to break, no version drift, no HF Hub rate limits
```
### ❌ Lexical disadvantages
```
- No semantics — "vehicle damaged in hail" won't match a node describing "car dented by hailstorm" unless words overlap
- No synonyms / paraphrase — you need the exact-ish vocabulary
- Sensitive to tokenization & stopwords — "claims over $100,000" needed special numeric-threshold handling (which we built) because plain BM25 can't compare numbers
- Repeated terms saturate (by design) — good for fairness, but a node with many distinct relevant terms can still lose to a terse exact hit
```
### ✅ Cross-encoder advantages
```
- Real relevance — understands meaning, synonymy, paraphrase, context; ranks like a human reader
- Robust to surface noise — typos, rewordings, abbreviations
- State of the art — this is the industry-standard re-ranker (reranking stage in production RAG/GraphRAG systems)
- Better headroom — as queries get more natural-language-y (executives asking "is this claim fishy?"), the cross-encoder keeps up where BM25 stalls
```
### ❌ Cross-encoder disadvantages
```
- Heavy footprint — 2–3 GB install, ~90 MB model, seconds of cold start, meaningful RAM/CPU per query
- Slow — 40× slower; at scale you'd want GPU or batching
- Black box — can't attribute a score to specific terms; weakens the "explainable AI" story unless you add post-hoc explanation (Phase 4's traversal logging helps here)
- Operational risk — download failures, offline envs, HF rate limits, model version drift; that's why it carries the lexical fallback
- Can be fooled differently — exact ids buried in long text can score lower than BM25's explicit match (mitigated by our id-anchored retrieval, which happens before ranking)
```

## The bottom line for your project
```
┌───────────────────────────────────────────────────────┬───────────────────────────────────────────────────────┐
│ Decision point                                        │ Choose                                                │
├───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Live demo, CI tests, no-download environments         │ Lexical (identical accuracy here, instant, auditable) │
├───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Production semantic quality, natural-language queries │ Cross-encoder (with lexical as the safety net)        │
├───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Executive-facing audit demo                           │ Lexical — you can print why each node ranked          │
├───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Cost-sensitive batch re-ranking                       │ Cross-encoder in large batches (amortizes model load) │
└───────────────────────────────────────────────────────┴───────────────────────────────────────────────────────┘
```
`And that's exactly why the architecture has  auto  as the default: it tries the cross-encoder (best quality), and the moment the model is unavailable or fails to load, it degrades permanently to BM25 for that process — same interface, same downstream pipeline, no downtime. Our benchmark shows you lose nothing on this dataset by falling back: 100% accuracy either way, 18.5% savings either way — you just trade 1.2 s latency for 31 ms and a 3 GB install for nothing.`
