# API Reference (Phase 5)

Base URL: `http://localhost:8000` — OpenAPI docs at `/docs` (Swagger UI).

## Authentication

When `API_KEY` is set, every `/api/v1/*` endpoint requires:

```
X-API-Key: <your-key>
```

Requests without a valid key → `403`. With `API_KEY` empty (local dev) auth is
disabled. Keys are compared with a constant-time digest.

## Rate limiting

All `/api/v1/*` endpoints share a per-client sliding window
(`RATE_LIMIT_PER_MINUTE`, default 60/min). Exceeding it → `429`. The limiter
is in-memory (single process) — see DEPLOYMENT.md before scaling out.

## Health

| Method | Path | Response |
|---|---|---|
| GET | `/health/live` | `{"status": "alive"}` — process up |
| GET | `/health/ready` | `{"status": "ready", "neo4j": true, "ollama": true}` or `503` with per-dependency breakdown |

## Endpoints

### `POST /api/v1/upload` — CDC ingest (multipart/form-data)

Upload an insurance PDF; the server extracts entities, diffs against the
existing snapshot, and surgically updates the graph.

```bash
curl -X POST http://localhost:8000/api/v1/upload \
     -H "X-API-Key: $API_KEY" \
     -F "file=@data/pdfs/policy_POL-0009.pdf"
```

Response: `{status, file, doc_id, extraction_mode, changes: {added, modified,
deleted}, update_stats: {entities_added, entities_updated, entities_deleted,
edges_added, update_time_ms, ...}}`

Validation (400/422): non-PDF extension, bad filename, empty file, >25 MB,
missing `%PDF` header, or no extractable entities.

### `GET /api/v1/session` — current session + available sessions

```bash
curl http://localhost:8000/api/v1/session -H "X-API-Key: $API_KEY"
```

```json
{"status": "success", "current_session": "pdf_demo",
 "sessions": [{"id": "fraud_oracle", "kind": "excel", "label": "...", "desc": "..."},
               {"id": "insurance_claims", "kind": "excel", ...},
               {"id": "insurance_dataset", "kind": "excel", ...},
               {"id": "pdf_demo", "kind": "pdf", ...}]}
```

### `POST /api/v1/session` — switch datasets without the web UI

Seeds the graph for the requested session (3 Excel datasets + the PDF demo
graph) by running the matching ingest/seed script in a worker thread — the
same backend the web UI's sidebar uses. Idempotent: requesting the already-
loaded session is a no-op (`seeded: false`); pass `"force": true` to re-seed
regardless. Takes ~2s for the small sessions, up to ~1 min for the large CSVs.

```bash
curl -X POST http://localhost:8000/api/v1/session \
     -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
     -d '{"session_id": "fraud_oracle"}'
```

```json
{"status": "success", "session": "fraud_oracle",
 "seeded": true, "current_session": "fraud_oracle",
 "elapsed_ms": 10234}
```

Validation/errors: unknown `session_id` → `400`; seeding failure → `500` with
the script's output tail. Valid ids: `fraud_oracle`, `insurance_claims`,
`insurance_dataset`, `pdf_demo`.

### `POST /api/v1/query` — token-optimized retrieval

```json
{"query": "Does claim CLM-0003 have a fraud flag?",
 "max_hops": 2, "token_budget": 1280,
 "reranker_mode": "auto", "answer_mode": "auto"}
```

Response: `{status, query, answer, answer_mode, answer_model,
answer_fallback, reranker, tokens: {before, after, savings_percent},
retrieval: {seeds, node_count, edge_count}, pruned: {kept, dropped, ...},
traversal: {audit_id, nodes_visited, edges_traversed, cypher, timings_ms},
execution_time_ms}`

Validation (422 via Pydantic): `query` 1–500 chars, `max_hops` 1–4,
`token_budget` 128–16384, mode enums.

### `GET /api/v1/metrics`

Rolling stats: `{summary: {total_requests, queries, uploads,
avg_query_latency_ms, avg_token_savings_pct}, metrics: [...last 100]}`

### `GET /api/v1/audit?limit=50`

Newest audit records (traversal + Cypher + tokens for every query, max 500).
These are the explainability artifacts the audit UI renders and the
JSON/HTML/PDF exporters consume.
