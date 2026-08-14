"""Typed application settings (pydantic-settings, .env aware)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "graphrag-demo"

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

    # Rate limiting (in-memory sliding window; single-process)
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_WINDOW_S: int = 60
    # trust X-Forwarded-For for client identity ONLY when behind a proxy that
    # sets it (never trust client-supplied headers in direct deployments)
    TRUST_PROXY_FORWARDED_FOR: bool = False

    # CDC
    DOC_SNAPSHOT_LABEL: str = "DocSnapshot"
    BATCH_SIZE: int = 500

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


settings = Settings()
