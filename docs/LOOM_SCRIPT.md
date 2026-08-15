# Phase 6 — 90-Second Loom Demo Script

**Target:** 90 seconds · **Audience:** EXL Scale & Ops team · **Tone:** confident, precise, honest
**Stack shown:** Streamlit dashboard + Audit UI + FastAPI + Neo4j + Ollama (`llama3.2:3b`)

> All numbers below are **recorded measurements** from the real-data validation runs
> (`data/real_dataset_results.md`, `data/benchmarks/*.json`). Nothing is invented.

---

## The one-line pitch (what the whole video proves)

> "I built a real-time GraphRAG for insurance claims that updates surgically instead
> of re-indexing, prunes context to cut token spend, and produces an audit trail
> executives can actually inspect — validated end-to-end on **3 real insurance
> datasets (4 CSV files), ~209,000 entities, 10,200 ground-truth queries**."

---

## Timed script

### 🎬 0:00 – 0:15 — The problem (15s)

**On screen:** Opening card → the dashboard homepage.

**Narration:**
> "EXL processes over 100,000 insurance claims a year. Three things break in
> GraphRAG at that scale: re-indexing the whole graph every time a policy changes
> costs real money; verbose retrieval wastes tokens; and when an executive asks
> *why* a claim was flagged, most systems can't show their work. I built a system
> that fixes all three — and validated it against three real insurance datasets
> (insurance_dataset and its synthetic variant are one source), including 1,212
> real fraud labels, on **10,200 ground-truth queries**."

---

### 🎯 Shot 1 — 0:15 – 0:35 — Real-time CDC, surgical updates (20s)

**On screen:** FastAPI terminal → `POST /api/v1/upload` with an endorsement PDF →
JSON response with `update_stats` (entities added/updated, edges, `update_time_ms`).

**Narration:**
> "Shot one: real-time change data capture. When a new endorsement lands, the
> system diffs it against what's already in the graph and updates **only the
> affected nodes and edges** — no full re-index. On the demo graph that surgical
> update completes in under **500 milliseconds**. The API returns exactly what
> changed, with timings, so operations knows the graph is always current."

**On screen (optional second beat — the auto-pipeline):**
> "And it's fully automatic: the pipeline watches the dataset folder — when a file
> is added or modified it ingests, re-benchmarks, and refreshes the dashboard by
> itself. All 53,503 rows of the largest dataset ingest in **16.6 seconds**."

---

### 🎯 Shot 2 — 0:35 – 0:58 — Token optimization (23s)

**On screen:** Dashboard "Real dataset validation" section → side-by-side token
comparison → the savings table.

**Narration:**
> "Shot two: token optimization. A cross-encoder re-ranker scores every retrieved
> node, and the pruner keeps only what's actually relevant to the question —
> measured per-query, per-dataset. On **10,200 ground-truth queries** across all
> four graphs, retrieval and pruning accuracy are both **100%** — reported with a
> Wilson 95% confidence interval (99.96–100%) so it's a statistical claim, not a
> screenshot."

**Anti-overfitting beat (say this — it answers the question auditors ask):**
> "And because 88% of a benchmark like this is exact id-lookups, I added an
> anti-circularity probe suite: 83 paraphrased queries without id quotes,
> hallucination checks on non-existent ids, and cross-schema phrasing — all 83
> passed. Those probes actually caught and fixed a real bug: a non-existent id
> used to return other real claims instead of 'not found'."

**Honesty beat (say this — it's your credibility):**
> "Token savings depend on retrieval density. On precise real queries — an ID, a
> policy, a threshold — the subgraph is already small, so we save **6 to 8%**.
> On dense subgraphs with many sibling nodes — coverage lookups, fraud lists —
> savings run **15%+ and up to 80%**. That's why we publish the number per query
> instead of a single marketing figure."

**On screen:** flash the savings table:
`fraud_oracle 7.5% · insurance_claims 6.4% · insurance_dataset 8.4% · pdf demo 15.5%`

---

### 🎯 Shot 3 — 0:58 – 1:20 — Lineage & audit trail (22s)

**On screen:** Audit UI → a fraud query → answer box → traversal path graph →
Cypher query → fraud ground-truth comparison panel.

**Narration:**
> "Shot three: explainability. Every query writes an audit record — question,
> answer, every node and edge traversed, the exact Cypher, and the model that
> answered. For fraud checks it's compared against the dataset's ground-truth
> labels live. And here's my favorite moment: when the model has **no evidence**
> that a claim is fraudulent, it says so instead of inventing a verdict."

**On screen:** the honest-refusal row from the fraud proof:
> *"Is claim CLM-00117 fraudulent?"* → **"Not determinable from the retrieved
> context… no information indicating fraudulent"** — CSV ground truth: **not fraud** ✅

**Narration (close of shot):**
> "Across all labeled graphs, **all 1,212 fraud claims were caught — 100%
> precision, 100% recall, zero false positives, zero false negatives.** The
> benchmark proves end-to-end fidelity: every flag survives retrieval,
> re-ranking, pruning, and answer generation."

---

### 🏁 1:20 – 1:30 — Production readiness & close (10s)

**On screen:** CI badge / pytest output / docker compose ps / README.

**Narration:**
> "It's built for production, not just the demo: **296 passing tests**, API
> authentication, rate limiting, JSON logging, health probes, Dockerized —
> Neo4j, API and dashboard in one `docker compose`, backed by CI. This is why I
> think I'm the right fit for your Scale & Ops team. Thank you."

---

## Number card (memorize these)

| Metric | Value |
|---|---|
| Real datasets validated | 3 (fraud_oracle, insurance_claims, insurance_dataset — its data_synthetic.csv is the synthetic form of the same source) |
| Entities in graph | ~209,000 |
| Largest ingest (all 53,503 rows) | 16.6s · 40,259 unique customers |
| Ground-truth retrieval accuracy | 10,200/10,200 = **100%** (Wilson 95% CI 99.96–100%) |
| Pruning accuracy | 10,200/10,200 = **100%** |
| Anti-circularity probes | 83/83 (paraphrase · hallucination · cross-schema · answer-level) |
| Fraud labels evaluated (full pipeline) | 1,212 · **P/R/F1 = 100% / 100% / 100%** |
| False positives / false negatives | 0 / 0 |
| CDC surgical update | **< 500 ms** |
| Retrieval+prune latency (15k+ entities) | 20–500 ms |
| Token savings (precise real queries) | 6.4–8.4% real · 15.5% pdf demo (published per query) |
| Token savings (dense subgraphs) | 15%+ avg · up to 80% |
| Tests | 296 passing |
| Demo LLM | llama3.2:3b via Ollama (local, no API cost) |

---

## Pre-recording checklist

1. **Restore demo graph:** `.venv/Scripts/python.exe scripts/seed_graph.py --reset --apply-schema`
   (or load a real dataset: `.venv/Scripts/python.exe scripts/ingest_real_dataset.py fraud_oracle --reset`).
2. **Start dashboard:** `streamlit run dashboard.py` → hard-refresh (Ctrl+Shift+R).
3. **Start API:** `.venv/Scripts/python.exe -m uvicorn graphrag.api_server:app --app-dir src --port 8000`.
4. **Ollama running** with `llama3.2:3b` pulled; answer mode = `auto`.
5. **Have the terminal ready** with the `/upload` call + JSON response pre-staged (paste it, don't type live).
6. **Audit UI** tab open with a fraud query recorded (so the lineage graph isn't empty on camera).
7. **Practice the honesty beat** — it's the most memorable 10 seconds of the video.

## Recording tips

- Record in **1080p**, keep the cursor slow and deliberate; no dead air — cut between beats.
- Put the **number card** up as a lower-third or a closing frame.
- If it runs long, cut the auto-pipeline beat (0:15) — keep the honesty beat (0:45) no matter what.
- Test the upload demo twice offline first: the LLM answer is the only multi-second stage (~5s on a 3B model); everything else is sub-second.

---

## Alt: 60-second cut (if asked for shorter)

- 0:00–0:10 problem → 0:10–0:22 Shot 1 (CDC numbers only) → 0:22–0:38 Shot 2 (savings + 100% accuracy) → 0:38–0:52 Shot 3 (audit trail + honest refusal) → 0:52–1:00 close (231 tests, dockerized).
