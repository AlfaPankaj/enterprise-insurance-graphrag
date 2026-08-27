# Aura Topology — Running GraphRAG v2 on Neo4j Aura (WS-E)

The graph tier runs on **Neo4j AuraDB** (managed); the app tier (FastAPI +
Streamlit) runs in Docker. Everything below works with the AuraDB **free
tier** for trials.

---

## 1. Provision Aura (free tier)

1. Create a free AuraDB instance at https://neo4j.com/cloud/aura/.
2. Copy the **connection URI** (`neo4j+s://xxxx.databases.neo4j.io`) and the
   generated password — the password is shown only once.
3. Optional but recommended: restrict the **IP allowlist** to your IPs.

## 2. Wire the app tier

```bash
cp .env.example .env
# .env:
#   NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
#   NEO4J_USER=neo4j
#   NEO4J_PASSWORD=<the generated password>

docker compose -f docker-compose.aura.yml up -d --build
python scripts/check_config.py      # probes the Aura instance + providers
```

`neo4j+s://` is native TLS — the driver needs no extra configuration. Seed
any session from the app sidebar or:

```bash
python scripts/seed_graph.py --reset --apply-schema \
  --uri "$NEO4J_URI" --user neo4j --password "$NEO4J_PASSWORD"
```

## 3. Tenancy on Aura

| Model | Isolation | Cost | When |
|---|---|---|---|
| **One AuraDB instance per tenant** | Strongest — separate storage, separate credentials | N instances | Production (client data) |
| **Single instance + `TENANT_MODE=column`** | tenant_id predicate on every node + query | 1 instance | Trials / internal multi-team |

With `TENANT_MODE=column`, every write path stamps `tenant_id` (first owner
wins) and every Cypher statement is scoped — see the blueprint §WS-B.

## 4. Backups & restore (RPO/RTO)

Aura runs **continuous backups** with point-in-time recovery (instance
settings → snapshots). For defense-in-depth, also export periodically:

```bash
# daily cron: full cypher export + S3 upload (uses the existing backup script)
python scripts/backup_neo4j.py --out /tmp/backups \
  --uri "$NEO4J_URI" --user neo4j --password "$NEO4J_PASSWORD"
aws s3 sync /tmp/backups s3://your-bucket/graphrag-backups/   # or any object store
```

| Scenario | Recovery path | Target RPO | Target RTO |
|---|---|---|---|
| Accidental DELETE/session reseed | Aura point-in-time snapshot | ≤ 5 min (continuous) | ~5–15 min |
| Region/instance failure | Restore snapshot to a new Aura instance, repoint `NEO4J_URI` | ≤ 24 h (snapshot cadence) | ~30 min |
| Full app-tier loss | Rebuild containers (`docker compose up -d`) + repoint URI | 0 (stateless tier) | ~10 min |

Restore drill (quarterly):
1. Create a scratch Aura instance from the snapshot.
2. Point a staging app tier at it (`NEO4J_URI` override).
3. Run `python scripts/check_config.py` + the CI benchmark subset:
   `python scripts/benchmark_answer_quality.py --min-faithfulness 0.8`.

## 5. Production hardening checklist (beyond the trial)

* **Auth**: `AUTH_MODE=jwt` + your IdP's `JWKS_URL` (Keycloak/Azure AD/Okta)
  — see `tests/test_oidc_rs256.py` for the verification flow.
* **PII**: `PII_MODE=mask` + `PII_ENCRYPTION_KEY` (at-rest Fernet; production
  path = envelope encryption with a KMS-wrapped data key).
* **Guardrails**: `GUARDRAILS_ENABLED=true` (injection + groundedness).
* **TLS**: Aura terminates TLS; put the API behind your edge proxy for
  end-to-end TLS and WAF.
* **Observability**: `TRACING_ENABLED=true` + OTLP collector; `/metrics` for
  Prometheus; `X-Request-ID` / `X-Trace-ID` on every response.
* **Scale**: the app tier is stateless (audit/jobs/vector-index volumes are
  per-instance) — run N replicas behind a load balancer; move the audit
  store to shared storage or a DB when multi-replica.

## 6. Topology diagram (trial → production)

```
                 trial                          production
  ┌──────────────────────────┐      ┌──────────────────────────────────┐
  │  browser / curl          │      │  browser / B2B clients           │
  └───────────┬──────────────┘      └────────┬─────────────────────────┘
              ▼                              ▼
  ┌──────────────────────────┐      ┌──────────────────────────────────┐
  │ Streamlit + FastAPI      │      │ LB / WAF (TLS)                    │
  │ (compose.aura.yml)       │      │  ├─ FastAPI × N (stateless)       │
  └───────────┬──────────────┘      │  ├─ Streamlit                     │
              ▼                      │  └─ OTLP → Jaeger/Tempo           │
  ┌──────────────────────────┐      └────────┬─────────────────────────┘
  │ Neo4j AuraDB Free        │               ▼
  │ (neo4j+s:// …)           │      ┌──────────────────────────────────┐
  └──────────────────────────┘      │ Neo4j AuraDB (per tenant)         │
                                    │ Ollama/OpenAI gateway            │
                                    └──────────────────────────────────┘
```
