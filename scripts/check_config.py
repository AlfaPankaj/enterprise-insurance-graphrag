#!/usr/bin/env python3
"""v2 config checker — validate a trial/production setup in one command.

Reads the same settings as the app (``.env`` → pydantic-settings), probes what
it can without spending money or leaking secrets, and prints a clear report:

    python scripts/check_config.py
    python scripts/check_config.py --no-probe    # skip network probes

Exit code: 0 = OK, 1 = a critical misconfiguration was found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graphrag.config import settings  # noqa: E402


def _ok(label: str, value: str, warn: str | None = None) -> None:
    flag = "OK " if not warn else "WARN"
    print(f"  [{flag}] {label:<28} {value}")
    if warn:
        print(f"       └─ {warn}")


def importlib_available(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


def probe_neo4j() -> str:
    """Reachability probe with a short acquisition timeout."""
    from neo4j import GraphDatabase

    try:
        kwargs = {"connection_acquisition_timeout": 5.0}
    except TypeError:  # pragma: no cover - very old driver
        kwargs = {}
    try:
        driver = GraphDatabase.driver(
            settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            **kwargs,
        )
        with driver.session() as session:
            session.run("RETURN 1").consume()
        driver.close()
        return "reachable"
    except Exception as exc:  # noqa: BLE001 - probe must never raise
        return f"NOT reachable ({type(exc).__name__})"


def probe_ollama() -> str:
    import httpx

    try:
        resp = httpx.get(f"{settings.LLAMA_API_URL}/api/tags", timeout=3)
        if resp.status_code != 200:
            return f"unreachable (HTTP {resp.status_code})"
        models = [m.get("name") for m in resp.json().get("models", [])]
        return f"reachable ({len(models)} model(s))"
    except Exception as exc:  # noqa: BLE001
        return f"unreachable ({type(exc).__name__})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-probe", action="store_true",
                        help="skip network probes (Neo4j / Ollama)")
    args = parser.parse_args(argv)

    critical = 0
    print("=" * 72)
    print("GraphRAG v2 — configuration report")
    print("=" * 72)

    print("\n[Neo4j]")
    _ok("URI", settings.NEO4J_URI)
    _ok("User", settings.NEO4J_USER)
    pw = settings.NEO4J_PASSWORD
    weak = pw in ("graphrag-demo", "password", "neo4j") or len(pw) < 8
    _ok("Password", f"***** (len={len(pw)})",
        "default/weak password — change it before any shared deployment" if weak else None)
    if not args.no_probe:
        reach = probe_neo4j()
        _ok("Probe", reach, None if reach == "reachable" else
            "start Neo4j (docker compose up -d neo4j) or fix the Aura URI")

    print("\n[LLM providers]  (LLM_PROVIDER=auto → first configured+reachable wins)")
    _ok("Mode", settings.LLM_PROVIDER)
    if settings.OPENAI_BASE_URL:
        _ok("OpenAI-compat base", settings.OPENAI_BASE_URL)
        _ok("OpenAI-compat model", settings.OPENAI_MODEL)
        key_state = "set" if settings.OPENAI_API_KEY else "NOT set"
        _ok("OpenAI-compat key", key_state,
            None if settings.OPENAI_API_KEY or "azure" not in settings.OPENAI_BASE_URL.lower()
            else "Azure requires OPENAI_API_KEY")
        if settings.OPENAI_API_VERSION:
            _ok("Azure api-version", settings.OPENAI_API_VERSION)
    else:
        _ok("OpenAI-compat", "not configured (set OPENAI_BASE_URL to enable)")
    _ok("Ollama URL", settings.LLAMA_API_URL)
    _ok("Ollama model", settings.LLAMA_MODEL)
    if not args.no_probe:
        ollama = probe_ollama()
        _ok("Ollama probe", ollama,
            "start Ollama, then: ollama pull " + settings.LLAMA_MODEL
            if "unreachable" in ollama else None)
    if settings.LLM_PROVIDER == "openai" and not settings.OPENAI_BASE_URL:
        print("  [FAIL] LLM_PROVIDER=openai but OPENAI_BASE_URL is empty")
        critical += 1
    if settings.ANSWER_MODE not in ("extractive", "auto", "llm"):
        print(f"  [FAIL] unknown ANSWER_MODE={settings.ANSWER_MODE}")
        critical += 1

    print("\n[Auth]  (AUTH_MODE: none=dev, static=API key, jwt=OIDC)")
    mode = settings.AUTH_MODE.lower()
    effective = mode if mode in ("static", "jwt") else (
        "static (API_KEY set)" if settings.API_KEY else "none (dev — DO NOT use in prod)")
    _ok("Effective mode", effective)
    if effective.startswith("static"):
        _ok("Static key", "set" if settings.API_KEY else "NOT set",
            None if settings.API_KEY else "set API_KEY in .env for production")
        _ok("Key roles", settings.API_KEY_ROLES)
    if effective == "jwt":
        has_secret = bool(settings.JWT_SECRET or settings.JWKS_URL)
        _ok("JWT secret/JWKS", "set" if has_secret else "NOT set",
            None if has_secret else "AUTH_MODE=jwt needs JWT_SECRET or JWKS_URL")
        if settings.JWT_ISSUER:
            _ok("Issuer", settings.JWT_ISSUER)
        if settings.JWT_AUDIENCE:
            _ok("Audience", settings.JWT_AUDIENCE)
        if not has_secret:
            critical += 1

    print("\n[Trust controls]")
    _ok("PII_MODE", settings.PII_MODE + (
        f" (readers: {settings.PII_READER_ROLES})" if settings.PII_MODE == "mask" else ""))
    enc = "set (len=%d)" % len(settings.PII_ENCRYPTION_KEY) if settings.PII_ENCRYPTION_KEY else "not set"
    _ok("PII encryption at rest", enc,
        None if settings.PII_ENCRYPTION_KEY else
        ("PII stored plaintext in Neo4j — generate a key:\n"
         '         python -c "import base64,os;print(base64.urlsafe_b64encode(os.urandom(32)).decode())"')
        if settings.PII_MODE == "mask" else
        "encryption off (v1 behavior); set PII_ENCRYPTION_KEY to encrypt PII fields")
    _ok("GUARDRAILS_ENABLED", str(settings.GUARDRAILS_ENABLED))
    _ok("TENANT_MODE", settings.TENANT_MODE + (
        f" (default: {settings.DEFAULT_TENANT})" if settings.TENANT_MODE == "column" else ""))
    if settings.TENANT_MODE == "column" and not settings.DEFAULT_TENANT:
        print("  [FAIL] TENANT_MODE=column but DEFAULT_TENANT is empty")
        critical += 1

    print("\n[API & retrieval]")
    _ok("API host/port", f"{settings.API_HOST}:{settings.API_PORT}")
    _ok("CORS origins", settings.CORS_ORIGINS)
    _ok("Rate limit", f"{settings.RATE_LIMIT_PER_MINUTE}/min")
    _ok("Max hops / tokens", f"{settings.MAX_HOPS} / {settings.MAX_TOKENS}")
    _ok("Reranker", settings.RERANKER_MODE)
    _ok("Embeddings", settings.EMBEDDING_PROVIDER + (
        f" ({settings.EMBEDDING_MODEL})" if settings.EMBEDDING_PROVIDER != "hash" and settings.OPENAI_BASE_URL else
        f" ({settings.EMBEDDING_OLLAMA_MODEL})" if settings.EMBEDDING_PROVIDER == "ollama" else ""))
    _ok("Audit dir", str(PROJECT_ROOT / settings.AUDIT_DIR))

    print("\n[Observability & jobs (v2)]")
    _ok("Tracing", "enabled" if settings.TRACING_ENABLED else "off",
        None if settings.TRACING_ENABLED else
        "set TRACING_ENABLED=true (+ pip install -r requirements-otel.txt) for OTel")
    if settings.TRACING_ENABLED:
        _ok("OTLP endpoint", settings.TRACING_OTLP_ENDPOINT or
            "(env OTEL_EXPORTER_OTLP_ENDPOINT)",
            None if settings.TRACING_OTLP_ENDPOINT else
            "no exporter configured — spans are created but not exported")
        if not importlib_available("opentelemetry"):
            print("  [WARN] opentelemetry packages not installed — "
                  "run: pip install -r requirements-otel.txt")
    _ok("Job DB", str(PROJECT_ROOT / settings.JOB_DB_PATH))

    env_file = PROJECT_ROOT / ".env"
    print("\n[Files]")
    _ok(".env", "present" if env_file.exists() else "missing (copy .env.example)",
        None if env_file.exists() else "cp .env.example .env and fill it in")
    if not env_file.exists():
        critical += 1

    print("\n" + "=" * 72)
    verdict = "ALL CHECKS PASSED" if critical == 0 else f"{critical} CRITICAL ISSUE(S)"
    print(f"Result: {verdict}")
    print("=" * 72)
    return 1 if critical else 0


if __name__ == "__main__":
    sys.exit(main())
