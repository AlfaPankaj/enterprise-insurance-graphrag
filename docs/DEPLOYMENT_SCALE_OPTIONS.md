# Production Deployment Options — Scaling Past the Reference Implementation

> **Status:** Option A is **implemented and opt-in** on this branch. Option B
> is **designed, not built** (deferred by decision — see §B.4).

## 1. The working core (unchanged)

> *"Our reference implementation uses Python, Neo4j, and custom
> BM25/Cross-Encoder reranking because it provides native graph data science
> capabilities for fraud ring detection and handles our 10,200-query
> benchmark with 100% precision."*

The Neo4j stack stays the default: every benchmark, every test, every demo
number in the README was produced on it. Nothing in this document changes
that. The question this document answers is the one a CTO asks next:

> *"Your Python + Neo4j system works great on 50,000 rows. How do you scale
> it to 50 million rows without high cloud bills or query timeouts?"*

## 2. The two expansion paths

> *"For production deployment, we designed two expansion paths: a unified
> PostgreSQL + pgvector + Apache AGE layer for teams prioritizing zero
> multi-DB sync overhead, and a Rust microservice tier with in-memory graph
> engines for sub-millisecond throughput."*

```
                        Enterprise GraphRAG v2 Core Engine
                 (Python + Neo4j + Hybrid Reranker + trust controls)
                                      |
        ┌─────────────────────────────┴─────────────────────────────┐
        │                                                           │
  Option A: Unified Data Stack                    Option B: Rust Tier
  (PostgreSQL + pgvector + Apache AGE)      (microservices + in-memory graph)
  ✅ implemented, opt-in, this branch           📐 designed, deferred
```

Both paths are **backend swaps behind the same Python core** — the pipeline,
reranker, PII/tenant controls, audit chain, cache, and evals are identical
in all configurations.

---

# Option A — The Unified Data Stack (PostgreSQL + pgvector + Apache AGE)

**Analogy: the Swiss Army knife — everything in one pocket.**

Instead of running three stores (relational DB + vector DB + Neo4j), one
PostgreSQL instance serves all three:

| Layer | Neo4j reference stack | Option A unified stack |
|---|---|---|
| Property graph (Cypher) | Neo4j 5.x | **Apache AGE** — openCypher inside PostgreSQL |
| Vector index (semantic seeds) | in-memory cosine (`vector_store.py`) | **pgvector** — HNSW index in the same database |
| Relational/business data | — | plain SQL tables, same instance, one backup |
| CDC writes | Neo4j transaction | **one ACID PostgreSQL transaction** (diff + derived edges + revision bump commit or roll back together) |
| Ops story | graph DB + (vector DB) + RDBMS | **one database**: one backup, one HA story, one security model, zero cross-DB sync |

Why it fits this codebase specifically:

* The graph layer speaks a **narrow driver surface** (`session.run` →
  `.data()`/`.single()`/iteration), so a ~300-line shim
  (`src/graphrag/postgres_backend.py`) makes the backend a config flag.
* All 34 Cypher call-sites are plain openCypher — no APOC, no `CALL {}`
  subqueries — the exact subset AGE implements.
* Tenant scoping, PII masking, and the audit hash-chain live in Python and
  port unchanged.

## A.1 What changes and what doesn't

| | Default (`GRAPH_BACKEND=neo4j`) | Option A (`GRAPH_BACKEND=age`) |
|---|---|---|
| Query results | identical | identical (same Cypher, same pipeline) |
| 10,200-query benchmark | canonical | re-validatable via parity test |
| Dependencies | requirements.txt | **+ requirements-postgres.txt** (psycopg2, optional) |
| CDC semantics | Neo4j tx | single PostgreSQL ACID tx (no multi-DB sync window at all) |
| Semantic index | in-memory (RAM-bounded) | pgvector HNSW (scales past RAM) or in-memory fallback |
| Operational scripts (`backup_neo4j.py`, benchmarks) | as-is | still Neo4j-targeted (source of truth) |

## A.2 Architecture

```
┌────────────────────────── one PostgreSQL instance ──────────────────────────┐
│                                                                             │
│  SQL tables (relational)      pgvector (vectors)      Apache AGE (graph)    │
│  e.g. jobs*, review*, any     graphrag_vectors        openCypher over the   │
│  existing business tables     HNSW cosine index       'graphrag' graph      │
│                                                                             │
│  * jobs/review stay SQLite by default — colocating them is optional         │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ psycopg2 (optional dep)
                    ┌─────────────────┴──────────────────┐
                    │  AgeDriver — neo4j-driver-shaped    │
                    │  shim (postgres_backend.py)         │
                    └─────────────────┬──────────────────┘
                                      │ identical interface
              api_server · audit_ui · query_pipeline · CDC updater
                     (no code changes above this line)
```

## A.3 Setup (10 minutes)

```bash
# 1) one database: PostgreSQL + AGE + pgvector
docker compose -f docker-compose.age.yml up -d

# 2) optional Python dependency (only needed for this backend)
pip install -r requirements-postgres.txt

# 3) copy the existing Neo4j graph into AGE (Neo4j stays source of truth)
python scripts/migrate_to_age.py --reset

# 4) validate the whole configuration
python scripts/check_config.py          # shows a [Graph backend] section

# 5) switch (.env)
GRAPH_BACKEND=age
POSTGRES_DSN=postgresql://graphrag:graphrag-demo@localhost:5432/graphrag
VECTOR_BACKEND=pgvector                 # optional; in-memory works too

# 6) run the live parity test against your instance
TEST_AGE_DSN=postgresql://graphrag:graphrag-demo@localhost:5432/graphrag \
  python -m pytest tests/test_age_backend.py -k live -v
```

Then start the app exactly as before (`uvicorn …`, `streamlit run app.py`) —
the API lifespan, audit UI, retrieval, and CDC now run on PostgreSQL.

## A.4 What to validate on first run (parity checklist)

The unit suite covers translation/parsing; against a **live** AGE instance
confirm:

1. `pytest -k live` (above) — write, read, transactional CDC, cleanup.
2. A session switch + a keyword/id/numeric query set from
   `data/benchmarks/golden_questions.json` — spot-check answers match the
   Neo4j run.
3. `scripts/benchmark_real_dataset.py --queries 200` style sampling before
   trusting AGE with the full benchmark.
4. Known AGE edges to watch: `labels(n)[0]` indexing behavior, list
   parameters in `UNWIND` (inlined client-side by the shim), and property
   maps returned instead of node objects (identical to how the code
   consumes them).

## A.5 When to choose Option A

* **Team/ops simplicity is the constraint** — one database to back up,
  secure, and monitor; no graph/vector/relational sync pipelines.
* **Relational + graph + vector workloads coexist** — claims ops tables and
  the knowledge graph in one transactional store.
* **Existing PostgreSQL investment** — HA, PITR, permissions, and on-call
  knowledge already in place; Neo4j would be a new system to operate.
* Scale: tens of millions of nodes/edges on one well-sized instance, with
  the same read path (Cypher→AGE) and pgvector HNSW for the semantic tier.

When **not**: heavy use of Neo4j GDS graph-data-science algorithms, or
workloads that need Neo4j's native multi-hop traversal engine tuned to the
last millisecond — keep the reference stack (or Option B for raw QPS).

---

# Option B — Rust Microservice Tier with In-Memory Graph Engines

**Analogy: the race-car pit crew — purpose-built speed.**

**Designed, not built.** This section documents the architecture we would
build when throughput (not ops simplicity) becomes the binding constraint.

## B.1 Target architecture

```
        Python core (unchanged: pipeline, trust controls, audit, evals)
                     │  gRPC/tonic  (hot path only)
        ┌────────────┴───────────────┐
        │  graph-hot-path service    │   Rust + tokio
        │  · slotmap/petgraph        │   in-memory graph, read-optimized,
        │    adjacency in arena      │   rebuilt from CDC events (Kafka/
        │  · rank_fusion RRF         │   PG logical slot), no per-query
        │  · lexical BM25 (tantivy)  │   network hops to a database
        └────────────┬───────────────┘
                     │ batch/snapshot
             Neo4j or AGE (system of record, cold path)
```

* **Hot path** (per query): seed → expand ≤2 hops → serialize → rank — all
  in memory, sub-millisecond, lock-free reads over an immutable snapshot
  swapped atomically on CDC applies.
* **Cold path** (per document update): the existing Python CDC engine
  commits to the system of record and emits the change event; the Rust tier
  applies it to the next snapshot.
* The Python core keeps everything an auditor needs — identity, PII, tenant
  scoping, the audit hash-chain, evals — because the Rust tier is a
  **stateless accelerator**, not a second brain.

## B.2 Expected envelope

| Metric | Reference (Neo4j, measured) | Option B (design target) |
|---|---|---|
| retrieval hot path | ~45 ms cross-encoder / 0.2 ms lexical | < 1 ms p99 (in-memory BFS + RRF) |
| throughput | rate-limit bound (60 req/min default) | 10⁴–10⁵ QPS per node, horizontal |
| cost curve | linear with DB tier | flat: RAM-bound replicas |

## B.3 When to choose Option B

* Query volume where per-request database round-trips dominate latency and
  spend (interactive product surfaces, not analyst tools).
* Steady-state graphs that fit in RAM (≤ ~50M nodes with id/props) and
  change incrementally — exactly what the CDC engine already guarantees.
* Strict tail-latency SLOs (p99 < 5 ms) that no database round-trip meets.

## B.4 Why it is deferred (locked decision)

1. The measured bottlenecks today are the **LLM** (answer generation) and
   the cross-encoder — not graph retrieval (0.2 ms lexical path).
2. The answer cache already serves repeated operational questions at zero
   LLM cost, and Option A removes the multi-DB tax for the scale most
   enterprises actually deploy at.
3. A second service language doubles the security-review and on-call
   surface; that is justified only when (B.3) is actually true.
4. Building it later is additive by design — the gRPC tier hides behind the
   same retriever interface, exactly like `GRAPH_BACKEND` does today.
