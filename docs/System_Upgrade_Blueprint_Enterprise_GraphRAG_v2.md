# System Upgrade Blueprint — Enterprise GraphRAG v2

**Workstream:** GenAI + RAG · **Scope:** production-grade GraphRAG for insurance & banking digital operations
**Branch:** `arena/01a02647-enterprise-insurance-graphrag` · **Baseline:** v1.x (commit `daaa1f5`)

This document is derived **from the code itself** (not from external research):
every gap below points at the file that shows it, and every v2 workstream maps
to concrete modules to build or change.

---

## 1. Where the system is today (v1 snapshot)

The repo is a working, benchmark-validated GraphRAG for **commercial insurance
claims** — a demo-grade system with unusually honest engineering:

| Layer | What exists | Where |
|---|---|---|
| Ingestion (Shot 1) | PDF → text → entity extraction (heuristic + Ollama LLM) → **CDC diff** → surgical Neo4j update in one transaction; per-doc snapshots; reference-counted deletes | `pdf_processor.py`, `entity_extractor.py`, `change_detector.py`, `graph_updater.py`, `graph_store.py` |
| Retrieval (Shot 2) | Cypher BFS multi-hop retrieval, id/keyword/numeric-threshold seeding, edge-aware cross-encoder or **zero-dependency BM25** re-ranking, protected-seed token pruning | `graph_retriever.py`, `reranker.py`, `context_pruner.py`, `token_counter.py` |
| Answer | Ollama LLM (`llama3.2:3b`) with deterministic extractive fallback; brace-safe prompt with grounding rules | `answer_generator.py`, `prompts/answer_prompts.txt` |
| Explainability (Shot 3) | Per-query audit record: seeds, traversal paths, Cypher, ranking scores, tokens, timings; JSON/HTML/PDF exports; pyvis lineage | `traversal_logger.py`, `path_extractor.py`, `audit_reporter.py`, `lineage_visualizer.py` |
| Sessions | 4 built-in datasets + BYO PDF/CSV uploads, re-seed from the UI, auto-benchmark in background | `sessions.py`, `custom_sessions.py`, `fraud_ground_truth.py` |
| API | FastAPI: `/upload`, `/query`, `/session`, `/metrics`, `/audit`, health probes; API-key auth, CORS, security headers, validation, JSON logs, in-memory rate limit | `api_server.py`, `security.py`, `validators.py`, `rate_limiter.py`, `health_check.py`, `monitoring.py`, `logger_config.py` |
| UI | Streamlit: Home / Dashboard / Audit Trail / Datasets (+ 2 standalone apps) | `app.py`, `dashboard.py`, `audit_ui.py` |
| Proof | 10,200 ground-truth queries @ 100% retrieval/pruning accuracy (Wilson CI 99.96–100%), fraud P/R/F1 = 100% on 1,212 labels, 83/83 anti-circularity probes, ~8% token savings (2,044,311 → 1,857,621) | `data/benchmarks/*.json`, `data/real_dataset_results.md` |
| Ops | Docker compose (Neo4j + API + dashboard), CI with real Neo4j, backup script, E2E compose variant | `Dockerfile*`, `docker-compose*.yml`, `.github/workflows/ci.yml`, `scripts/backup_neo4j.py` |

**What must be preserved in v2 (non-negotiables):**
- Deterministic fallbacks at every LLM touchpoint (extractive answer, lexical BM25, heuristic extraction) — the system must never hard-fail when a model is down.
- Atomic CDC (graph update + snapshot in one transaction).
- The benchmark culture: ground truth computed from source data, anti-circularity probes, honest caveats (e.g. the 88.6% exact-id query mix is disclosed).
- Protected-seed pruning semantics (answer-context protection).

---

## 2. What "production-grade" means here

For an insurance/banking digital-operations audience, v2 must be able to answer
**"yes"** to the questions a CISO, an ops lead, and an auditor will ask:

1. **Trust & compliance** — Who ran this query? Can this user see PII? Is the
   audit trail tamper-evident? Are model answers grounded and refusal-aware?
2. **Isolation** — Can client A's data ever appear in client B's answer?
3. **Resilience & scale** — Does load degrade gracefully? What happens when the
   LLM, the DB, or a replica dies? Is cost per answer predictable?
4. **Quality control** — How do we know a change didn't silently degrade
   answer quality? Is answer quality *measured*, not just retrieval fidelity?
5. **Operability** — Deployable with secrets management, TLS, DR, and
   observability (traces, metrics), not just logs.

v1 proves the **product idea** (the 3 kill shots work on real data). v2's job
is to close the distance from *demo* to *deployable platform*.

---

## 3. Gap analysis (v1 → v2)

### 3.1 P0 — blockers for any real deployment

| # | Gap | Evidence in code | Why it blocks |
|---|---|---|---|
| G1 | **No user identity / RBAC** — one static `X-API-Key` for everyone; the Streamlit UI has **no auth at all** | `security.py` (`require_api_key` compares one key), `app.py` (no login) | Auditors require per-user identity; banking ops need role separation (analyst vs admin vs auditor). Also: the audit trail cannot say *who* asked. |
| G2 | **PII is plaintext and queryable by anyone** — policyholder `name`, `dob`, `address`, `phone`, `email` are raw properties; the retriever serializes all of them into LLM context | `entity_extractor.py` (`_build_policy`), `graph_retriever.py` (`_NODE_TEXT_PROPS`), `serialize_node` | GDPR/HIPAA/IRDAI-class data in insurance ops; field-level masking/encryption and purpose-based access are table stakes. |
| G3 | **Event loop is blocked by the sync pipeline** — `run_query` (Neo4j + rerank + LLM HTTP, up to seconds) is called directly inside `async def query_graph`; same for uploads | `api_server.py` (`query_graph`, `upload_pdf`) | Under concurrency the API serializes requests; a slow LLM call stalls everything else. |
| G4 | **Single global graph, swapped by re-seeding** — "sessions" DETACH DELETE and re-ingest the whole database; one user switching sessions changes what *every* user queries | `sessions.py` (`_run_blocking` → ingest scripts with `--reset`) | No multi-user concurrency, no multi-tenancy. Client isolation is impossible. |
| G5 | **Audit trail is not tamper-evident and not attributed** — append-only JSONL with no hash chaining, no signature, no user id; capped by periodic rewrite (`_trim`) | `traversal_logger.py` | A compliance audit trail that can be silently edited or deleted does not survive an auditor. |
| G6 | **LLM provider is hard-wired to Ollama** — every LLM call is an httpx call to `/api/generate` with `stream: False`; no provider abstraction, retries, or model routing | `answer_generator.py`, `entity_extractor.py` | Enterprises run Azure OpenAI / Vertex / Bedrock / internal gateways, with per-call cost accounting. |
| G7 | **No guardrails beyond prompt text** — no prompt-injection defense for documents, no output PII redaction, no policy checks, no citation enforcement | `prompts/*.txt`, `answer_generator.py` | OWASP LLM Top-10 exposure; a malicious policy PDF or query can steer the model; answers can echo PII. |

### 3.2 P1 — production engineering gaps

| # | Gap | Evidence |
|---|---|---|
| G8 | In-memory rate limiter & metrics — lost on restart, per-process only (multi-replica breaks) | `rate_limiter.py` (docstring admits it), `monitoring.py` |
| G9 | No **answer/retrieval caching** — identical questions (very common in ops: "status of CLM-…?") recompute retrieval + LLM each time | no cache except probe/negative cache in `answer_generator.py` |
| G10 | No **streaming** — answers return as one blob after `stream: False` | `answer_generator.py`, `api_server.py` |
| G11 | Jobs run as **daemon threads / subprocesses inside the web process** (seeding, benchmarks) — no durable job state, no retries, dies with the process | `sessions.py` (`start_switch`, `_maybe_auto_benchmark`) |
| G12 | Observability stops at JSON logs — no Prometheus endpoint, no OTel spans, request-id only on the *exception* path | `logger_config.py`, `exception_handlers.py` |
| G13 | Secrets & config — `NEO4J_PASSWORD` default `graphrag-demo` in code; **no `.env.example`**; no `.gitignore` (94 MB repo — CSVs, zips, PDFs, audit exports committed) | `config.py`, repo root |
| G14 | Benchmarks are scripts, **not CI gates** — a regression in retrieval/pruning accuracy or fraud F1 can merge silently | `.github/workflows/ci.yml` runs unit tests only |
| G15 | **Answer quality is never evaluated** — all 10,200-query metrics measure retrieval/pruning fidelity; nothing measures faithfulness, answer relevance, or groundedness of LLM answers (no LLM-judge / RAG triad) | `benchmark_real_dataset.py` measures hit/miss + tokens only |

### 3.3 P2 — capability gaps for the real use case

| # | Gap | Evidence |
|---|---|---|
| G16 | **No semantic retrieval** — pure Cypher BFS + keyword seeding + BM25/cross-encoder over serialized nodes; no vector index, no hybrid fusion (RRF). Paraphrases work only via prefix tricks | `graph_retriever.py`, `reranker.py` |
| G17 | **Extraction tuned to generated PDFs** — regex rows (`_COVERAGE_ROW`, `_ENDORSEMENT_ROW`) parse only the synthetic reportlab documents; real policy wordings, FNOL reports, and medical documents will not parse | `entity_extractor.py` |
| G18 | **Insurance-only ontology** — the pitch covers insurance **and banking**; there is no banking domain model (accounts, transactions, disputes, AML alerts) and no pluggable-ontology mechanism | `docs/graph_schema.md`, `ingest_real_dataset.py` adapters |
| G19 | Neo4j **Community edition**: no multi-DB, no RBAC, no clustering — the topology assumed by sessions (single mutable DB) is a demo topology | `docker-compose.yml` (`neo4j:5.26-community`) |
| G20 | No TLS termination example, no k8s manifests, no DR runbook, backup is a script not a schedule | `docker-compose.yml`, `scripts/backup_neo4j.py` |

---

## 4. V2 blueprint — workstreams

Each workstream lists concrete deliverables, the files involved, and acceptance
criteria. Order reflects dependency, not necessarily priority — see §6.

### WS-A — Platform core (concurrency, provider abstraction, caching, streaming, jobs)

1. **Async offload (G3)** — run `run_query`/uploads in `asyncio.to_thread` /
   `run_in_executor`, or make the pipeline async. *Files:* `api_server.py`, `query_pipeline.py`.
   *Accept:* p95 latency under concurrent load (locust, 50 VU) ≤ 2× single-user p95.
2. **LLM provider layer (G6)** — `LLMProvider` protocol: `generate(prompt, *, max_tokens, temperature, stream) -> answer + usage(tokens, cost)`; implementations: Ollama, OpenAI-compatible (covers Azure, vLLM, Together), with retry/backoff/timeouts and per-call token/cost accounting. `ANSWER_MODE` → `LLM_PROVIDER` config. *Files:* new `src/graphrag/llm/` package; rewire `answer_generator.py`, `entity_extractor.py`.
   *Accept:* all existing tests green with provider mocked; cost per answer appears in the audit record.
3. **Answer & retrieval cache (G9)** — exact-query cache + semantic-dedup cache keyed by query hash + session/dataset version + schema version; TTL; invalidation on CDC write. *Files:* new `cache.py`; `query_pipeline.py`, `graph_updater.py` (invalidate).
   *Accept:* identical repeated query returns cached in <5 ms with `cached: true` in audit.
4. **Streaming answers (G10)** — provider streams tokens; API `POST /api/v1/query/stream` (SSE); UI renders incrementally. *Files:* `llm/`, `api_server.py`, `app.py`.
   *Accept:* first token < 500 ms for LLM answers on local gateway.
5. **Job runner (G11)** — durable jobs (Postgres-backed or Redis queue) for seeding/ingest/benchmarks; API `POST /api/v1/jobs`, `GET /api/v1/jobs/{id}`; UI progress from job state instead of in-process threads. *Files:* new `jobs.py`; replace daemon threads in `sessions.py`.
   *Accept:* job survives API restart and reports final state.

### WS-B — Trust & compliance (identity, isolation, PII, audit integrity, guardrails)

1. **Identity & RBAC (G1)** — OIDC (Keycloak/Azure AD) or mTLS + JWT; roles `analyst / admin / auditor`; per-role endpoint policy; per-request identity logged into the audit trail. *Files:* `security.py`, `api_server.py`, `traversal_logger.py`, `app.py` (login gate).
   *Accept:* unauthenticated request → 401; auditor role can read audit but not seed; every audit record carries `user_id`/`subject`.
2. **Tenant isolation (G4)** — replace "one mutable graph swapped by reset" with per-tenant scoping: Neo4j multi-DB (Enterprise/Aura) or a `tenant_id` property on every node + tenant predicate injected into every Cypher statement. Sessions become a per-tenant concept. *Files:* `graph_retriever.py` (all queries), `sessions.py`, `graph_updater.py`, `seed/ingest scripts`.
   *Accept:* cross-tenant leakage probe suite (queries that must return empty) passes; two concurrent users on different tenants cannot affect each other.
3. **PII policy engine (G2)** — classification tags per property (`PII_NAME`, `PII_CONTACT`, `PII_HEALTH`, `NONE`); encrypt-at-rest (Neo4j field encryption or app-layer AES-GCM with KMS); mask-in-transit: retrieval serialization redacts fields the caller's role can't see; LLM context and answers both respect it. *Files:* new `pii.py` + schema tags; `graph_retriever.py` (`serialize_node`), `answer_generator.py`.
   *Accept:* `analyst` role's context/answers contain no raw DOB/phone/email; redaction visible in audit; tests for each field class.
4. **Tamper-evident audit (G5)** — hash-chain each record (SHA-256 of prev hash + record), periodic anchor write, signature key from KMS/HSM; stop rewriting the file (`_trim` → rotated segments with manifest); add `user_id`, `tenant_id`, model usage/cost. *Files:* `traversal_logger.py`, `audit_reporter.py`.
   *Accept:* modifying/removing a middle record is detected (verification script + test).
5. **Guardrails (G7)** — layered: (a) input — query & document injection screening; (b) output — PII redaction, forbidden-content check, groundedness check ("does the answer contain ids/values not in context?"); (c) policy — refusal templates, escalation. *Files:* new `guardrails.py`; hooks in `query_pipeline.py`.
   *Accept:* adversarial probe suite (injection PDFs, "ignore instructions" queries, PII-echoing queries) passes; refusals are logged and audit-tagged.

### WS-C — Retrieval & document intelligence v2

1. **Hybrid retrieval (G16)** — vector index for node texts/chunks (Neo4j native vector index or pgvector/Qdrant), embedding provider via the LLM layer, RRF fusion with BM25 + graph traversal; keep the pure-Cypher + lexical path as the deterministic fallback. *Files:* `graph_retriever.py`, new `vector_store.py`, `embedder.py`.
   *Accept:* paraphrase benchmark (no id mentions, synonyms) improves without regressing the existing 10,200-query suite.
2. **Real-document extraction (G17)** — layout-aware parsing (tables, clauses) + LLM structured output validated against the ontology (pydantic schemas per entity), confidence scores, and a **human-in-the-loop review queue** for low-confidence extractions. *Files:* `entity_extractor.py`, new `extraction_review.py`, UI page.
   *Accept:* extraction works on real-worded (non-synthetic) insurance PDFs; low-confidence extractions land in the review queue; CDC still applies only confirmed changes.
3. **Answer-quality evaluation (G15)** — LLM-judge + rule-based eval: faithfulness, answer relevance, groundedness, refusal correctness; a golden answer set (curated ~200 hard questions across sessions); `scripts/benchmark_answer_quality.py`; results in `data/benchmarks/`. *Files:* new `evals/`; `fraud_ground_truth.py` stays for the P/R/F1 gate.
   *Accept:* every release reports answer-quality deltas next to retrieval accuracy.

### WS-D — Observability, CI gates & release quality

1. **Metrics & tracing (G12)** — Prometheus `/metrics` (requests, latency histograms, token/cost, cache hit rate, queue depth); OTel spans per pipeline stage (retrieve → rank → prune → answer); request-id propagated end-to-end on all paths. *Files:* `monitoring.py`, `api_server.py`, `query_pipeline.py`, `logger_config.py`.
2. **CI regression gates (G14)** — a benchmark job in CI: seed Neo4j, run a bounded benchmark suite (e.g. 500 queries + fraud subset + anti-circularity probes + eval subset) and fail on accuracy/F1 regression vs. committed baselines. *Files:* `.github/workflows/ci.yml`, `scripts/ci_benchmark_gate.py`.
   *Accept:* a PR that degrades retrieval accuracy or fraud F1 cannot merge.
3. **Load & resilience tests** — locust scenarios; chaos checks (LLM down → fallback; Neo4j down → 503 + recovery; audit dir read-only → degraded but safe). *Files:* `tests/load/`, `scripts/chaos_*.py`.
4. **Dependency & secret hygiene** — `pip-audit`/Dependabot, secret scanning, pinned lockfiles. *Files:* CI.

### WS-E — Domain, deployment & data governance

1. **Banking domain extension (G18)** — pluggable ontology: a second domain package (Account / Customer / Transaction / Dispute / AMLAlert + relationship verbs) with its own extraction adapters and benchmark builders, selected per tenant. *Files:* new `src/graphrag/domains/{insurance,banking}/`, refactor `ingest_real_dataset.py` into per-domain adapters.
2. **Enterprise topology (G19/G20)** — Neo4j Enterprise/AuraDS support (multi-DB per tenant, RBAC, clustering); k8s manifests or compose with TLS-terminating proxy (nginx/caddy); backup → S3 schedule + restore runbook + RPO/RTO test. *Files:* new `deploy/` directory.
3. **Repo hygiene (G13)** — add `.gitignore` (CSVs/zips/audit exports/`.env`), move datasets to LFS or external storage, add `.env.example` with no secrets, remove default password from `config.py`.
4. **Prompt & model governance** — prompts move to versioned registry with per-version eval snapshots; model changes require eval re-run. *Files:* `prompts/` → `src/graphrag/prompts/*.yaml` + loader.

---

## 5. Proposed v2 acceptance targets (draft — to confirm)

| Metric | v1 (measured) | v2 target |
|---|---|---|
| Retrieval & pruning accuracy (ground-truth suite) | 100% (10,200 q) | ≥ 99.9% incl. new paraphrase/hybrid suite |
| Fraud P/R/F1 | 100% / 100% / 100% (1,212 labels) | ≥ 99% with CI gate |
| Answer faithfulness (eval judge) | not measured | ≥ 95% |
| Answer relevance (eval judge) | not measured | ≥ 90% |
| Refusal correctness (unanswerable queries) | 83/83 probes | ≥ 95% on expanded probe set |
| p95 query latency (50 VU load) | not measured | ≤ 1.5 s (LLM) / ≤ 150 ms (extractive, cached) |
| LLM token cost per answer | not measured per call | tracked + ≤ v1 context budget |
| Security posture | static key, no UI auth | OIDC + RBAC + PII controls + OWASP LLM Top-10 mitigations evidenced |
| Audit integrity | plain JSONL | hash-chained, verifiable, user-attributed |
| CI regression protection | unit tests only | benchmark + eval + security gates |

---

## 6. Quick wins (1–3 days each, unblock immediately)

1. `.gitignore` + `.env.example` + remove default `NEO4J_PASSWORD` (G13).
2. Async offload of `run_query` (G3) — one-line-ish change with real concurrency payoff.
3. Request-id on the success path + Prometheus `/metrics` (G12-lite).
4. Exact-query answer cache (G9) — big cost/latency win for ops-style repeated questions.
5. PII masking toggle in `serialize_node` + answer post-filter (G2-lite) — immediate risk reduction.
6. Benchmark gate in CI at reduced scale (G14).
7. Audit record gains `user_id`/`tenant_id` fields + hash chaining (G5 core).

## 7. What this blueprint deliberately does **not** change

- The dual-engine reranker (BM25 zero-dependency fallback stays first-class).
- The extractive deterministic answer fallback and the honest "Not determinable"
  refusal behavior.
- Atomic CDC semantics and the ground-truth-by-construction benchmark method.
- The existing 10,200-query suite and fraud benchmarks — they become regression
  gates, and every new capability must pass them unchanged.

---

## 8. Decisions (locked) & build status

| Decision | Locked choice |
|---|---|
| Priority order | **WS-B Trust & compliance first**, then WS-A platform core |
| LLM direction | **Multi-provider layer** (OpenAI-compatible gateways incl. Azure + Ollama fallback) |
| Domain scope | **Insurance + banking** via a pluggable-ontology layer |
| Deployment target | **Neo4j Aura (managed)** for the graph; app tier on compose, k8s-ready |
| Tenancy model | `tenant_id`-scoped predicates today; Aura multi-DB per tenant when available |

### v2 implementation status (this branch)

**Done — foundation + trust core (v2 slice 1):**

* **LLM provider layer** — `src/graphrag/llm/` (`base`, `ollama`, `openai_compat`,
  `pricing`, `factory`): one `LLMResult` contract with token usage + cost
  accounting, fallback chain `openai → ollama → extractive`, probe caching,
  actionable errors. `answer_generator` + `entity_extractor` rewired through
  it; the v1 Ollama wire contract is preserved (all 273 pre-existing tests
  stay green). (G6, G15-lite)
* **Identity & RBAC** — `src/graphrag/identity.py`: `UserIdentity` with
  subject/roles/tenant; `AUTH_MODE = none | static | jwt` (HS256 shared secret
  or RS256 via JWKS); `require_user(*roles)` dependency with per-endpoint role
  policies (upload/session/admin/audit/metrics); dev mode keeps the v1 open
  API (anonymous = all roles). Every query/upload is now attributed. (G1)
* **Tenant isolation plumbing** — `graph_retriever.tenant_predicate()` +
  `TENANT_MODE=column`: every Cypher statement carries a `$tenant` predicate
  (NULL = unscoped v1 behavior); identity tenant flows through
  `run_query(..., identity=...)`. (G4 groundwork)
* **PII policy engine** — `src/graphrag/pii.py`: (label, prop) classification
  table + name-pattern fallback, `MaskingPolicy.for_roles()`, masked retrieval
  context + answer scrub; `PII_MODE=mask` opts in (retrieval semantics
  unchanged — only PII-classed fields are redacted). (G2)
* **Guardrails** — `src/graphrag/guardrails.py`: query/document injection
  screening, answer groundedness (ids/values must exist in context),
  refusal detection, PII-echo scan; findings recorded in the result + audit
  record, injection = blocked answer when `GUARDRAILS_ENABLED=true`. (G7)
* **Tamper-evident audit** — `traversal_logger.AuditStore`: SHA-256 hash
  chain (`prev_hash`/`record_hash`), append-only segment rotation (no more
  rewrite-trim), `verify()` + `GET /api/v1/audit/verify`; v1 legacy records
  anchor the chain; audit records now carry `user`, `tenant_id`, provider
  `usage`, `cost_usd`. (G5)
* **API hardening** — sync pipeline offloaded to worker threads (no event-loop
  blocking); request-id on every response; RBAC-gated endpoints. (G3)
* **Repo hygiene** — `.gitignore`, `.env.example` (no secrets), `PyJWT` added
  to both requirements files. (G13 partial)
* **Tests** — 58 new tests (providers, identity, PII, guardrails, audit
  integrity, tenant scoping, end-to-end pipeline w/ scripted fake driver):
  **331 passed, 12 skipped** (skips = DB-dependent, as before).

**Next slices (in order):** WS-B remainder (tenant stamping on ingest + CDC
writes, PII at-rest encryption, OIDC integration test) → WS-A (streaming,
answer cache, job runner, Prometheus/OTel) → WS-C (hybrid retrieval + evals)
→ WS-E (banking domain, Aura topology).
