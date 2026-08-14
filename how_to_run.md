# How to run

```bash
# Single app — Home + Dashboard + Audit Trail (recommended):
.venv/Scripts/python.exe -m streamlit run app.py

# Shot 2 — token optimization dashboard (standalone)
.venv/Scripts/python.exe -m streamlit run dashboard.py

# Shot 3 — audit trail & lineage (browse recorded queries, their traversal
# paths, Cypher, and export JSON/HTML/PDF compliance reports)
.venv/Scripts/python.exe -m streamlit run src/graphrag/audit_ui.py

# REST API (upload = CDC; query = token-optimized retrieval + traversal)
.venv/Scripts/python.exe -m uvicorn graphrag.api_server:app --app-dir src

# Tests + benchmarks
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe scripts/run_benchmark.py --reranker-mode lexical

# Benchmark proof (single consolidated JSON — the run: accuracy, tokens, savings,
# fraud P/R/F1, reranker latency, projected annual cost)
.venv/Scripts/python.exe scripts/export_benchmark_proof.py   # -> data/benchmarks/benchmark_results.json

# Live execution telemetry showcase (one real query, full audit trail to stdout)
.venv/Scripts/python.exe scripts/showcase_execution_telemetry.py
```

## Sessions — no terminal needed for dataset switching

Inside `app.py`, the sidebar's **Active Session** selector exposes 4 sessions:

| Session | Pipeline | Seeded by |
|---|---|---|
| `fraud_oracle` | Excel / real CSV | `scripts/ingest_real_dataset.py fraud_oracle --reset` |
| `insurance_claims` | Excel / real CSV | `scripts/ingest_real_dataset.py insurance_claims --reset` |
| `insurance_dataset` | Excel / real CSV | `scripts/ingest_real_dataset.py insurance_dataset --reset` |
| `pdf_demo` | PDF / synthetic demo graph | `scripts/seed_graph.py --reset --apply-schema` |

Picking a different session re-seeds the graph **inside the app's backend
process** (idempotent — it is a no-op if that session is already loaded; the
**Re-seed this session** button forces a fresh ingest). Every page then
queries the selected session, and ground-truth fraud comparisons follow the
loaded dataset. Equivalent one-liners if you prefer the terminal:

### Custom sessions — upload your own PDF/CSV

The **Datasets** page in `app.py` accepts your own files: upload a PDF or CSV,
give the session a **unique name** (no collision with the built-ins above),
and it is re-seeded into the graph so you can query your own data. Sessions
can be **renamed** / **removed** there, are persisted in
`data/custom_sessions.json`, and appear in the sidebar + API session list like
any other session. **CSV uploads are benchmarked automatically**: after the
session seeds, `scripts/benchmark_real_dataset.py --custom-session <name>`
runs in the background and the Dashboard's Pipeline Validation row fills in by
itself. Terminal equivalent:

```bash
.venv/Scripts/python.exe scripts/ingest_custom_dataset.py <name> --reset
.venv/Scripts/python.exe scripts/seed_graph.py --reset --apply-schema   # back to PDF demo
```

Built-in sessions from the terminal:

```bash
.venv/Scripts/python.exe scripts/ingest_real_dataset.py fraud_oracle --reset
```

Every query run through the dashboard, the API (`POST /api/v1/query`), or the
benchmark is logged to `data/audit_trail/audit_trail.jsonl` (trimmed to the
newest 2000 records) — the audit UI reads that store. Exports land in
`data/audit_trail/exports/`.