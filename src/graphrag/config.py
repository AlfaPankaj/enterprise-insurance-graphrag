"""Typed application settings (pydantic-settings, .env aware)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "graphrag-demo"

    # ------------------------------------------------------------------
    # v2 — Option A: unified PostgreSQL data stack (Apache AGE + pgvector)
    # ------------------------------------------------------------------
    # neo4j (default) = reference implementation, zero behavior change;
    # age = route all graph reads/writes through Apache AGE inside
    # PostgreSQL (docs/DEPLOYMENT_SCALE_OPTIONS.md). Optional dep:
    #   pip install -r requirements-postgres.txt
    GRAPH_BACKEND: str = "neo4j"
    POSTGRES_DSN: str = ""    # postgresql://graphrag:graphrag@localhost:5432/graphrag
    AGE_GRAPH_NAME: str = "graphrag"
    AGE_AUTO_CREATE: bool = True          # create the AGE graph on first use
    # memory = bounded demo index; neo4j = native HNSW for Neo4j;
    # pgvector = HNSW in PostgreSQL/AGE.
    VECTOR_BACKEND: str = "memory"
    NEO4J_VECTOR_INDEX: str = "graphrag_embeddings"
    NEO4J_FULLTEXT_INDEX: str = "graphrag_fulltext"
    VECTOR_INCREMENTAL_ENABLED: bool = True
    KEYWORD_SEED_LIMIT_MIN: int = 5
    KEYWORD_SEED_LIMIT_MAX: int = 50

    # Llama (Ollama) — optional; the extractor falls back to deterministic parsing
    LLAMA_API_URL: str = "http://localhost:11434"
    LLAMA_MODEL: str = "llama3.2:3b"     # installed on the demo machine
    # heuristic = deterministic parser (default, fast, reproducible);
    # llm = require Ollama; auto = try LLM, fall back to heuristic
    EXTRACTION_MODE: str = "heuristic"

    # Answer generation (Shot 2b): auto = try Ollama LLM, fall back to
    # extractive on any failure (resilient default); extractive = deterministic
    # rule-based answers only; llm = require Ollama (raise if unavailable)
    ANSWER_MODE: str = "auto"
    ANSWER_MODEL: str = "llama3.2:3b"    # defaults to LLAMA_MODEL if empty
    ANSWER_MAX_TOKENS: int = 512          # cap on generated answer length

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    # leave empty to run WITHOUT auth (local dev); set for production
    API_KEY: str = ""
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8501,http://localhost:8502"

    # ------------------------------------------------------------------
    # v2 — multi-provider LLM layer
    # ------------------------------------------------------------------
    # auto = OpenAI-compatible when configured, else Ollama, else extractive;
    # ollama / openai = require that provider (raise/fall back per caller)
    LLM_PROVIDER: str = "auto"
    OPENAI_BASE_URL: str = ""            # Azure: https://<res>.openai.azure.com/openai/deployments/<deployment>
    OPENAI_API_KEY: str = ""             # vLLM/Together/Ollama-compat may not need a key
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_API_VERSION: str = ""         # Azure only: e.g. 2024-06-01
    LLM_TIMEOUT_S: float = 90.0
    LLM_MAX_RETRIES: int = 2             # retries on transient 5xx/timeout
    LLM_PROBE_TTL_S: float = 15.0        # cache both successful/failed health probes
    OLLAMA_KEEP_ALIVE: str = "10m"       # retain the selected model in memory
    QUERY_MODEL_WARMUP_ENABLED: bool = False
    STREAM_ANSWERS_DEFAULT: bool = True
    HTTP_POOL_MAX_CONNECTIONS: int = 100
    HTTP_POOL_MAX_KEEPALIVE_CONNECTIONS: int = 20
    HTTP_POOL_KEEPALIVE_EXPIRY_S: float = 30.0
    # per-1k-token USD used to price provider usage blocks (set contracted rates)
    LLM_PRICE_PER_1K_INPUT: float = 0.0
    LLM_PRICE_PER_1K_OUTPUT: float = 0.0

    # Schema induction stays deterministic unless explicitly enabled. Sample
    # values are withheld from LLM providers by default to protect uploaded PII.
    SCHEMA_INDUCTION_MODE: str = "deterministic"  # deterministic | auto | llm
    SCHEMA_LLM_INCLUDE_SAMPLES: bool = False

    # OCR provider interface. glm-ocr targets an OpenAI-compatible GLM-OCR
    # service (vLLM/SGLang); dependencies and the model remain optional.
    OCR_PROVIDER: str = "none"                    # none | auto | glm-ocr
    OCR_MIN_CHARS_PER_PAGE: int = 80
    GLM_OCR_BASE_URL: str = ""
    GLM_OCR_API_KEY: str = ""
    GLM_OCR_MODEL: str = "zai-org/GLM-OCR"
    OCR_TIMEOUT_S: float = 120.0

    # ------------------------------------------------------------------
    # v2 — identity & RBAC (trust & compliance workstream)
    # ------------------------------------------------------------------
    # none = dev (auth off, anonymous identity); static = X-API-Key;
    # jwt = OIDC bearer token (HS256 shared secret or RS256 via JWKS_URL)
    AUTH_MODE: str = "none"
    API_KEY_ROLES: str = "admin,analyst,auditor"   # roles granted to the static key
    JWT_SECRET: str = ""                 # HS256 shared secret (dev/simple OIDC)
    JWT_ISSUER: str = ""                 # required when AUTH_MODE=jwt
    JWT_AUDIENCE: str = ""               # optional
    JWKS_URL: str = ""                   # RS256 key discovery (Keycloak/Azure AD)
    # default tenant for unauthenticated requests (dev); JWT carries tenant_id claim
    DEFAULT_TENANT: str = "demo"
    # roles allowed to read PII-classified fields (PII_MODE=mask)
    PII_READER_ROLES: str = "admin,auditor"

    # ------------------------------------------------------------------
    # v2 — trust controls (PII, guardrails, tenant isolation)
    # ------------------------------------------------------------------
    # off = raw fields everywhere (v1 behavior); mask = redact PII-classified
    # fields in retrieval context + answers unless the caller's role can read them
    PII_MODE: str = "off"
    # app-layer field encryption for PII-classified properties (Fernet/AES).
    # Empty = disabled (v1 behavior). Set to a 32-byte urlsafe-base64 key:
    #   python -c "import base64,os;print(base64.urlsafe_b64encode(os.urandom(32)).decode())"
    # (a plain passphrase is also accepted — it is SHA-256-derived into a key)
    PII_ENCRYPTION_KEY: str = ""
    # false = v1 behavior; true = input/output guardrail checks on every query
    GUARDRAILS_ENABLED: bool = False
    # off = v1 un-scoped graph; column = every Cypher query is prefixed with a
    # tenant predicate ({tenant_id: $tenant}) so tenants can never see each other
    TENANT_MODE: str = "off"

    # Rate limiting (in-memory sliding window; single-process)
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_WINDOW_S: int = 60
    # trust X-Forwarded-For for client identity ONLY when behind a proxy that
    # sets it (never trust client-supplied headers in direct deployments)
    TRUST_PROXY_FORWARDED_FOR: bool = False

    # CDC
    DOC_SNAPSHOT_LABEL: str = "DocSnapshot"
    BATCH_SIZE: int = 500

    # Shared upload pipeline (root upload.py). Byte values are explicit so the
    # same limits apply to FastAPI, Streamlit, and custom-session re-ingestion.
    UPLOAD_MAX_PDF_BYTES: int = 25 * 1024 * 1024
    UPLOAD_MAX_CSV_BYTES: int = 50 * 1024 * 1024
    UPLOAD_MAX_BATCH_BYTES: int = 100 * 1024 * 1024
    UPLOAD_MAX_FILES: int = 20
    UPLOAD_MAX_PDF_PAGES: int = 400
    UPLOAD_MAX_CSV_ROWS: int = 100_000
    UPLOAD_MAX_CSV_COLUMNS: int = 256
    UPLOAD_MAX_CSV_FIELD_CHARS: int = 100_000
    UPLOAD_MAX_FILENAME_CHARS: int = 128
    # Optional scanner command, e.g. "clamscan --no-summary". The temporary
    # file path is appended unless an argument contains the {path} placeholder.
    UPLOAD_MALWARE_SCAN_COMMAND: str = ""
    UPLOAD_MALWARE_SCAN_TIMEOUT_S: float = 30.0
    UPLOAD_MALWARE_SCAN_FAIL_CLOSED: bool = True
    UPLOAD_DEDUPLICATE_CUSTOM: bool = True
    UPLOAD_AUDIT_PATH: str = "data/audit_trail/uploads.jsonl"

    # Retrieval & token optimization (Phase 3)
    MAX_HOPS: int = 2             # BFS expansion depth from seed nodes
    MAX_TOKENS: int = 1280        # pruned-context token budget (answer context
                                  # is protected; the budget trims the rest)
    TOP_K: int = 50               # max nodes kept after re-ranking
    # auto = cross-encoder if available, else deterministic lexical fallback
    RERANKER_MODE: str = "auto"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    TOKENIZER_MODEL: str = "cl100k_base"  # tiktoken encoding (proxy for llama)

    # Phase 4 — lineage & audit trail (Shot 3)
    AUDIT_ENABLED: bool = True          # log every query's traversal to disk
    AUDIT_DIR: str = "data/audit_trail"  # JSONL store + exported reports
    AUDIT_MAX_RECORDS: int = 2000      # trim the JSONL store to this many records

    # v2 — answer cache (WS-A). Off by default = v1 behavior. Cache keys bind
    # query + pipeline params + tenant + PII scope + dataset revision; any
    # graph write bumps the revision, so cached answers never survive a write.
    CACHE_ENABLED: bool = False
    CACHE_TTL_S: int = 300
    CACHE_MAX_ENTRIES: int = 1000

    # v2 — OpenTelemetry tracing (WS-D). Optional deps: requirements-otel.txt.
    # When enabled + endpoint set, spans flow to any OTLP collector (Jaeger,
    # Tempo, Datadog, New Relic). Response headers carry X-Trace-ID.
    TRACING_ENABLED: bool = False
    TRACING_OTLP_ENDPOINT: str = ""     # e.g. http://localhost:4318/v1/traces

    # v2 — durable job runner (WS-A, G11). Seeding / benchmark jobs are
    # tracked in SQLite (data/jobs.db) instead of only in-process threads:
    # they survive restarts and expose status via POST/GET /api/v1/jobs.
    JOB_DB_PATH: str = "data/jobs.db"

    # ------------------------------------------------------------------
    # v2 — hybrid retrieval (WS-C, G16)
    # ------------------------------------------------------------------
    # Embedding backend for semantic seed fallback + hybrid re-ranking:
    # auto = OpenAI-compatible when configured, else Ollama, else a
    # zero-dependency deterministic hash embedder (always available).
    EMBEDDING_PROVIDER: str = "auto"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_OLLAMA_MODEL: str = "nomic-embed-text"
    # Hash vectors are a demo fallback only. Production startup/config checks
    # fail if provider resolution would select them.
    APP_ENV: str = "development"
    ALLOW_HASH_EMBEDDINGS_IN_PRODUCTION: bool = False
    # cap on nodes indexed into the in-memory vector store (large CSV
    # sessions stay bounded; the store is cached per dataset revision)
    VECTOR_INDEX_MAX_NODES: int = 25000

    # ------------------------------------------------------------------
    # v2 — extraction review queue (WS-C, G17)
    # ------------------------------------------------------------------
    # When enabled, extracted entities scoring below the confidence threshold
    # are HELD for human review instead of written to the graph — CDC only
    # ever applies confirmed changes. Off = v1 behavior (apply everything).
    EXTRACTION_REVIEW_ENABLED: bool = False
    EXTRACTION_CONFIDENCE_THRESHOLD: float = 0.7
    REVIEW_DB_PATH: str = "data/extraction_review.db"


settings = Settings()
