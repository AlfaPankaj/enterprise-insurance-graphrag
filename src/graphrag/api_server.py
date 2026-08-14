"""GraphRAG Insurance Claims System — production FastAPI server.

Endpoints
---------
* ``POST /api/v1/upload`` — CDC: PDF -> entities -> diff -> surgical Neo4j update
* ``POST /api/v1/query``  — retrieve -> re-rank -> prune -> token-optimized answer
* ``GET  /api/v1/session`` / ``POST /api/v1/session`` — switch datasets (Phase 6)
* ``GET  /api/v1/metrics`` — rolling latency / token-savings stats
* ``GET  /api/v1/audit``  — recent explainability records (Shot 3)
* ``GET  /health/live`` / ``GET /health/ready`` — orchestrator probes

Phase 5 hardening: API-key auth (when ``API_KEY`` is set), in-memory rate
limiting, structured JSON logs, global exception handlers with request ids,
CORS + security headers, and a readiness probe covering Neo4j + Ollama.

Run:  uvicorn graphrag.api_server:app --app-dir src
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from graphrag.change_detector import detect_changes
from graphrag.config import settings
from graphrag.entity_extractor import extract_entities
from graphrag.exception_handlers import register_exception_handlers
from graphrag.graph_store import get_existing_entities
from graphrag.graph_updater import update_graph_surgically
from graphrag.health_check import register_health_endpoints
from graphrag.logger_config import setup_logging
from graphrag.monitoring import monitor
from graphrag.pdf_processor import extract_text_from_pdf
from graphrag.query_pipeline import run_query
from graphrag.rate_limiter import rate_limit
from graphrag.security import require_api_key, setup_security
from graphrag.sessions import (all_sessions, current_session_id,
                               session_exists, switch_session)
from graphrag.traversal_logger import audit_store
from graphrag.validators import UploadValidationError, validate_pdf_upload

setup_logging()
logger = logging.getLogger("graphrag.api")


class QueryRequest(BaseModel):
    """Validated query payload for the retrieval endpoint."""

    query: str = Field(..., min_length=1, max_length=500)
    max_hops: int = Field(settings.MAX_HOPS, ge=1, le=4)
    token_budget: int = Field(settings.MAX_TOKENS, ge=128, le=16384)
    reranker_mode: str | None = Field(None, pattern=r"^(auto|cross-encoder|lexical)$")
    answer_mode: str | None = Field(None, pattern=r"^(extractive|auto|llm)$")


class SessionRequest(BaseModel):
    """Session-switch payload: which dataset/pipeline to load in Neo4j."""

    session_id: str = Field(..., min_length=1, max_length=64)
    force: bool = Field(False, description="re-ingest even if already loaded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.driver = GraphDatabase.driver(
        settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )
    logger.info("Neo4j driver connected: %s", settings.NEO4J_URI)
    yield
    app.state.driver.close()
    logger.info("Neo4j driver closed")


app = FastAPI(
    title="GraphRAG Insurance Claims System",
    version="2.0.0",
    description="Real-time CDC: PDF upload -> entity diff -> surgical Neo4j update. "
                "Token-optimized multi-hop retrieval with full audit trail.",
    lifespan=lifespan,
)

setup_security(app)
register_exception_handlers(app)
# health routes are registered at module scope; the readiness probe reads the
# driver from app.state at request time (no lifespan coupling)
register_health_endpoints(app)


@app.post("/api/v1/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    _auth=Depends(require_api_key),
    _rl=Depends(rate_limit()),
):
    """Ingest an insurance PDF. Returns the CDC diff + surgical update stats."""
    contents = await file.read()
    try:
        validate_pdf_upload(file.filename, contents)
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    start = time.perf_counter()
    try:
        text = extract_text_from_pdf(contents)
        extracted = extract_entities(text, doc_id_hint=file.filename)
        entities = extracted["entities"]
        if not entities:
            raise HTTPException(
                status_code=422,
                detail="Could not extract any entities from this PDF "
                       "(is it one of the generated insurance documents?)",
            )
        doc_id = extracted["doc_id"] or file.filename

        with app.state.driver.session() as session:
            old = get_existing_entities(session, doc_id)
            changes = detect_changes(old, entities)
            # graph update + snapshot save run in one transaction (atomic CDC)
            stats = update_graph_surgically(
                app.state.driver, doc_id, changes, new_entities=entities
            )

        stats["execution_time_ms"] = round((time.perf_counter() - start) * 1000, 2)
        monitor.record(kind="upload", file=file.filename, doc_id=doc_id,
                       latency_ms=stats["execution_time_ms"])
        logger.info("upload ok file=%s doc=%s", file.filename, doc_id,
                    extra={"doc_id": doc_id, "latency_ms": stats["execution_time_ms"]})

        return {
            "status": "success",
            "file": file.filename,
            "doc_id": doc_id,
            "extraction_mode": extracted["mode"],
            "changes": {
                "added": len(changes["added"]),
                "modified": len(changes["modified"]),
                "deleted": len(changes["deleted"]),
            },
            "update_stats": stats,
        }
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 - report and wrap; never leak internals
        logger.exception("upload failed for %s", file.filename)
        raise HTTPException(status_code=500, detail="Internal Server Error") from None


@app.post("/api/v1/query")
async def query_graph(
    request: QueryRequest,
    _auth=Depends(require_api_key),
    _rl=Depends(rate_limit()),
):
    """Token-optimized retrieval: returns context, savings, and an answer."""
    try:
        result = run_query(
            app.state.driver,
            request.query,
            max_hops=request.max_hops,
            token_budget=request.token_budget,
            reranker_mode=request.reranker_mode,
            answer_mode=request.answer_mode,
        )
        monitor.record(kind="query", query=request.query,
                       latency_ms=result["execution_time_ms"],
                       savings_pct=result["tokens"]["savings_percent"])
        return {"status": "success", **result}
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 - report and wrap; never leak internals
        logger.exception("query failed: %s", request.query)
        raise HTTPException(status_code=500, detail="Internal Server Error") from None


@app.get("/api/v1/session")
async def get_session(_auth=Depends(require_api_key)):
    """Current session + the sessions available for switching."""
    return {
        "status": "success",
        "current_session": current_session_id(app.state.driver),
        "sessions": [
            {"id": s["id"], "kind": s["kind"], "label": s["label"],
             "desc": s["desc"]}
            for s in all_sessions()
        ],
    }


@app.post("/api/v1/session")
async def set_session(
    request: SessionRequest,
    _auth=Depends(require_api_key),
    _rl=Depends(rate_limit()),
):
    """Switch the loaded graph to another session (re-seeds if needed).

    Runs the ingest/seed as a blocking subprocess (2-60s for the larger CSVs)
    in a worker thread, so the event loop stays responsive. Idempotent:
    requesting the already-loaded session is a no-op unless ``force`` is set.
    """
    if not session_exists(request.session_id):
        raise HTTPException(
            status_code=400,
            detail=f"unknown session '{request.session_id}' (expected one of "
                   f"{sorted(s['id'] for s in all_sessions())})",
        )
    start = time.perf_counter()
    try:
        info = await asyncio.to_thread(
            switch_session, app.state.driver, request.session_id, request.force
        )
    except (ValueError, RuntimeError) as exc:  # validation / seeding failure
        logger.warning("session switch failed session=%s: %s",
                       request.session_id, exc)
        code = 400 if isinstance(exc, ValueError) else 500
        raise HTTPException(status_code=code, detail=str(exc)) from exc

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info("session switched session=%s status=%s elapsed_ms=%.0f",
                request.session_id, info["status"], elapsed_ms)
    return {
        "status": "success",
        "session": info["session"],
        "seeded": info["status"] == "seeded",
        "current_session": current_session_id(app.state.driver),
        "elapsed_ms": elapsed_ms,
    }


@app.get("/api/v1/metrics")
async def get_metrics(_auth=Depends(require_api_key)):
    """Rolling request stats: counts, avg latency, avg token savings."""
    return {"summary": monitor.summary(), "metrics": monitor.recent(100)}


@app.get("/api/v1/audit")
async def get_audit(limit: int = 50, _auth=Depends(require_api_key)):
    """Recent query audit records (Shot 3 explainability trail)."""
    return {"status": "success", "records": audit_store.recent(min(limit, 500))}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT, log_level="info")
