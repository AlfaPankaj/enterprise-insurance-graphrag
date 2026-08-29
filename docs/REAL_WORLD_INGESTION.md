# Real-world ingestion and retrieval

## Trust boundary and workflow

`upload.py` validates and atomically persists bytes. It does not infer schemas
or build graphs. Every persisted source is revalidated (path containment,
structure, configured limits, optional malware scan, and recorded SHA-256)
before profiling or ingestion.

CSV workflow:

1. `persisted`
2. `profiling` — one-pass CSV profiling with disk-backed exact cardinalities
3. `awaiting_mapping_review` — deterministic proposal, optionally improved by
   an LLM and then strictly validated
4. `approved` — a named human accepts the versioned mapping fingerprint
5. `ingesting` — bounded row batches and disk-backed ID/entity indexes
6. `succeeded` or `failed` — timings, warnings, cost, checkpoints, duplicate,
   orphan, resolution, and row/edge counts remain durable

`data/custom_sessions.json` is atomically replaced under process and OS file
locks. Its immutable manifest binds tenant, sources, content metadata, and a
manifest checksum. SQLite jobs provide execution status and crash recovery.

## Schema mapping contract

A mapping contains:

- one entry per CSV: node label, source ID column, source-to-property map, types;
- FK relationships: source/target file and column, safe relationship type, and
  direction;
- conservative, anchored ID regexes learned from source IDs;
- entity-resolution mode, exact keys, fuzzy threshold, and reviewed accepted or
  rejected pairs;
- version, review status, and SHA-256 fingerprint.

Every input column is mapped; omitted editable properties are restored to a
safe normalized name during validation. LLM output cannot invent files or
columns and cannot bypass human approval.

## Entity resolution

Email and phone values use Unicode/case/format normalization and can be merged
exactly. Name/address similarity is blocked by postal code (or a coarse name
block) and enters a review list. Fuzzy pairs are never auto-applied. Accepted
merges retain canonical and source IDs; reports retain method, score, fields,
and decision, allowing an alias to be removed and re-ingested.

## PDF and OCR

`extract_document_from_pdf` returns page text, tables, OCR pages, provider, and
warnings. Only pages below `OCR_MIN_CHARS_PER_PAGE` are rendered. Set:

```dotenv
OCR_PROVIDER=glm-ocr
GLM_OCR_BASE_URL=http://glm-ocr-service:8000/v1
GLM_OCR_MODEL=zai-org/GLM-OCR
```

Install `requirements-ocr.txt` in the application only for PDF rendering. Run
GLM-OCR separately with a compatible vLLM/SGLang endpoint; the multi-gigabyte
model is intentionally not a default dependency. Extraction prompts are built
from registered `DomainSpec` ontologies, delimit documents as untrusted data,
and reject unregistered labels or non-flat properties.

## Retrieval and vectors

- Exact seeds combine registered domain IDs, learned per-dataset patterns,
  conservative generic IDs, and explicitly quoted arbitrary values.
- Neo4j full-text indexes are the normal keyword path. The old property scan is
  only a compatibility fallback for AGE or an unapplied schema.
- `VECTOR_BACKEND=neo4j` stores embeddings on `:Searchable` nodes and queries a
  native HNSW vector index.
- `VECTOR_BACKEND=pgvector` preserves PostgreSQL/AGE deployments and supports
  entity upserts/deletes.
- `VECTOR_BACKEND=memory` is demo-only and logs when its configured cap causes
  incomplete semantic recall.
- CDC and relational batches update durable vectors incrementally. A vector
  failure is reported as lag without rolling back an already committed graph
  transaction.

Production (`APP_ENV=production`) fails startup/config checks when embedding
resolution selects `HashEmbedder`, unless the explicit unsafe demo override is
set.

## Tenant isolation

Custom-session names are unique per tenant and stored below a hashed tenant
directory. Registry CRUD, jobs, review items, snapshots, graph nodes/edges,
markers, cache revisions, full-text results, vector results, and reset commands
are tenant-qualified. In `TENANT_MODE=column`, apply the schema on a fresh
database with:

```bash
python scripts/seed_graph.py --reset --apply-schema --tenant "$TENANT_ID"
```

This creates composite `(tenant_id, id)` uniqueness. Before converting an
existing globally constrained database, remove the old `*_id_unique`
constraints in a planned migration; global ID uniqueness and duplicate IDs
across tenants cannot coexist. When upgrading an existing Neo4j deployment,
recreate the expanded keyword index once so its label/property definition is
updated (an `IF NOT EXISTS` statement cannot alter an old definition):

```cypher
DROP INDEX graphrag_fulltext IF EXISTS;
```

Then rerun `scripts/seed_graph.py --apply-schema`.

## Independent evaluation

`data/benchmarks/independent_golden_questions.json` is a frozen 100-question
holdout, separate from `scripts/build_golden_set.py`, with 20 questions each
for ID lookup, paraphrase, multi-hop, aggregation, and negative/refusal.

```bash
python scripts/benchmark_answer_quality.py \
  --golden data/benchmarks/independent_golden_questions.json \
  --require-independent
```

Reports include aggregate and per-category quality means.
