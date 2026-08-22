"""GraphRAG Insurance Claims System — production FastAPI server.

Endpoints
---------
* ``POST /api/v1/upload`` — CDC: PDF -> entities -> diff -> surgical Neo4j update
* ``POST /api/v1/query``  — retrieve -> re-rank -> prune -> token-optimized answer
* ``POST /api/v1/query/stream`` — same pipeline as SSE (live token streaming)
* ``GET  /api/v1/session`` / ``POST /api/v1/session`` — switch datasets (Phase 6)
* ``GET  /api/v1/metrics`` — rolling latency / token-savings stats
* ``GET  /api/v1/audit`` / ``GET /api/v1/audit/verify`` — explainability records
* ``GET  /api/v1/review`` / ``POST /api/v1/review/{id}/approve|reject`` —
  extraction review queue (v2, WS-C): low-confidence entities are held for
  human decision before they ever touch the graph
* ``GET  /metrics`` — Prometheus exposition (v2)
* ``GET  /health/live`` / ``GET /health/ready`` — orchestrator probes

Phase 5 hardening: API-key auth (when ``API_KEY`` is set), in-memory rate
limiting, structured JSON logs, global exception handlers with request ids,
CORS + security headers, and a readiness probe covering Neo4j + Ollama.

Run:  uvicorn graphrag.api_server:app --app-dir src
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from graphrag.change_detector import detect_changes
from graphrag.config import settings
from graphrag.entity_extractor import extract_entities_with_confidence
from graphrag.exception_handlers import register_exception_handlers
from graphrag.extraction_review import (STATUS_PENDING, apply_review_item,
                                        get_review_store, partition_entities)
from graphrag.graph_store import get_existing_entities
from graphrag.graph_updater import update_graph_surgically
from graphrag.health_check import register_health_endpoints
from graphrag.identity import (AUDIT_ROLES, METRICS_ROLES, QUERY_ROLES,
                               SESSION_ROLES, UPLOAD_ROLES, UserIdentity,
                               require_user)
from graphrag.jobs import JobError, get_store, register_default_handlers
from graphrag.logger_config import setup_logging
from graphrag.monitoring import monitor
from graphrag.pdf_processor import extract_text_from_pdf
from graphrag import prometheus as prom
from graphrag.query_pipeline import run_query, stream_query
from graphrag.rate_limiter import rate_limit
from graphrag.security import setup_security
from graphrag.sessions import (all_sessions, current_session_id,
                               session_exists, switch_session)
from graphrag.tracing import configure as configure_tracing
from graphrag.tracing import current_trace_id, start_span
from graphrag.traversal_logger import audit_store
from graphrag.validators import UploadValidationError, validate_pdf_upload

setup_logging()
logger = logging.getLogger("graphrag.api")


class QueryRequest(BaseModel):
    """Validated query payload for the retrieval endpoint."""

    query: str = Field(..., min_length=1, max_length=500)
    max_hops: int = Field(settings.MAX_HOPS, ge=1, le=4)
    token_budget: int = Field(settings.MAX_TOKENS, ge=128, le=16384)
    reranker_mode: str | None = Field(None, pattern=r"^(auto|cross-encoder|lexical|hybrid)$")
    answer_mode: str | None = Field(None, pattern=r"^(extractive|auto|llm)$")


class SessionRequest(BaseModel):
    """Session-switch payload: which dataset/pipeline to load in Neo4j."""

    session_id: str = Field(..., min_length=1, max_length=64)
    force: bool = Field(False, description="re-ingest even if already loaded")


class JobSubmitRequest(BaseModel):
    """Enqueue a background job (v2 durable job runner).

    Supported kinds (registered at startup): ``session_switch``
    (``{session_id, force?}``), ``benchmark`` (``{dataset, queries?, workers?}``),
    ``fraud_benchmark`` (``{dataset, negatives?}``).
    """

    kind: str = Field(..., min_length=1, max_length=64)
    params: dict = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.driver = GraphDatabase.driver(
        settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )
    logger.info("Neo4j driver connected: %s", settings.NEO4J_URI)
    configure_tracing()
    register_default_handlers(lambda: app.state.driver)
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


@app.middleware("http")
async def trace_requests(request, call_next):
    """v2 OTel: one span per HTTP request; X-Trace-ID for correlation."""
    with start_span("http.request", {
        "http.method": request.method,
        "http.url.path": request.url.path,
    }) as span:
        response = await call_next(request)
        if span is not None:
            span.set_attribute("http.status_code", response.status_code)
            # the security middleware stamps request.state.request_id inside
            # the call — attach it (and the caller identity) at response time
            span.set_attribute("request_id",
                               getattr(request.state, "request_id", None))
            user = getattr(request.state, "user", None)
            if user is not None:
                span.set_attribute("enduser.id", user.subject)
                span.set_attribute("user.tenant", user.tenant_id)
                span.set_attribute("user.auth", user.auth_method)
        trace_id = current_trace_id()
        if trace_id:
            response.headers["X-Trace-ID"] = trace_id
        return response


@app.post("/api/v1/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    user: UserIdentity = Depends(require_user(*UPLOAD_ROLES)),
    _rl=Depends(rate_limit()),
):
    """Ingest an insurance PDF. Returns the CDC diff + surgical update stats."""
    contents = await file.read()
    try:
        validate_pdf_upload(file.filename, contents)
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    start = time.perf_counter()

    def _process_upload() -> dict:
        """Sync ingestion work — run in a worker thread (G3: the event loop
        must never block on extraction/LLM/Neo4j).

        v2 review queue: low-confidence entities are HELD for human review
        instead of written to the graph — CDC only ever applies confirmed
        changes (EXTRACTION_REVIEW_ENABLED)."""
        text = extract_text_from_pdf(contents)
        extracted = extract_entities_with_confidence(text, doc_id_hint=file.filename)
        entities = extracted["entities"]
        if not entities:
            raise HTTPException(
                status_code=422,
                detail="Could not extract any entities from this PDF "
                       "(is it one of the generated insurance documents?)",
            )
        doc_id = extracted["doc_id"] or file.filename

        held_ids: list[str] = []
        if settings.EXTRACTION_REVIEW_ENABLED:
            to_apply, held = partition_entities(
                entities, extracted.get("confidence", {}),
                settings.EXTRACTION_CONFIDENCE_THRESHOLD)
            store = get_review_store()
            for item in held:
                rid, _created = store.submit(
                    doc_id, file.filename, item["label"], item["entity_id"],
                    item["props"], item["confidence"],
                    extracted.get("reasons", {}).get(item["label"], {}).get(item["entity_id"]))
                held_ids.append(rid)
        else:
            to_apply = entities
        applied_count = sum(len(ents) for ents in to_apply.values())

        if not to_apply:
            # everything held for review — nothing applied, no snapshot, no CDC
            return {
                "doc_id": doc_id,
                "mode": extracted["mode"],
                "stats": None,
                "changes": {"added": 0, "modified": 0, "deleted": 0},
                "held_ids": held_ids,
                "applied_count": 0,
            }

        with app.state.driver.session() as session:
            old = get_existing_entities(session, doc_id)
            changes = detect_changes(old, to_apply)
            # graph update + snapshot save run in one transaction (atomic CDC);
            # v2: nodes are stamped with the caller's tenant (TENANT_MODE=column)
            stats = update_graph_surgically(
                app.state.driver, doc_id, changes, new_entities=to_apply,
                tenant_id=user.tenant_id,
            )
        return {
            "doc_id": doc_id,
            "mode": extracted["mode"],
            "stats": stats,
            "changes": {
                "added": len(changes["added"]),
                "modified": len(changes["modified"]),
                "deleted": len(changes["deleted"]),
            },
            "held_ids": held_ids,
            "applied_count": applied_count,
        }

    try:
        processed = await asyncio.to_thread(_process_upload)
        doc_id, extracted_mode, stats, changes = (
            processed["doc_id"], processed["mode"], processed["stats"],
            processed["changes"],
        )
        held_ids = processed["held_ids"]
        applied_count = processed["applied_count"]

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        prom.requests_total.inc(kind="upload")
        if stats is not None:
            stats["execution_time_ms"] = elapsed_ms
            prom.upload_latency.observe(elapsed_ms / 1000.0)
        monitor.record(kind="upload", file=file.filename, doc_id=doc_id,
                       user=user.subject, tenant_id=user.tenant_id,
                       latency_ms=elapsed_ms)
        logger.info("upload ok file=%s doc=%s user=%s tenant=%s applied=%d held=%d",
                    file.filename, doc_id, user.subject, user.tenant_id,
                    applied_count, len(held_ids),
                    extra={"doc_id": doc_id, "user": user.subject,
                           "tenant_id": user.tenant_id,
                           "latency_ms": elapsed_ms,
                           "held_for_review": len(held_ids)})

        return {
            "status": "success",
            "file": file.filename,
            "doc_id": doc_id,
            "extraction_mode": extracted_mode,
            "changes": changes,
            "update_stats": stats,
            "review": {
                "held": len(held_ids),
                "review_ids": held_ids,
                "applied_entities": applied_count,
            },
        }
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 - report and wrap; never leak internals
        prom.errors_total.inc(kind="upload")
        logger.exception("upload failed for %s", file.filename)
        raise HTTPException(status_code=500, detail="Internal Server Error") from None


@app.post("/api/v1/query")
async def query_graph(
    request: QueryRequest,
    user: UserIdentity = Depends(require_user(*QUERY_ROLES)),
    _rl=Depends(rate_limit()),
):
    """Token-optimized retrieval: returns context, savings, and an answer."""
    try:
        # G3: the sync pipeline (Neo4j + rerank + LLM, up to seconds) runs in
        # a worker thread so the event loop keeps serving concurrent requests
        result = await asyncio.to_thread(
            run_query,
            app.state.driver,
            request.query,
            max_hops=request.max_hops,
            token_budget=request.token_budget,
            reranker_mode=request.reranker_mode,
            answer_mode=request.answer_mode,
            identity=user,
        )
        _record_query_metrics(result, kind="query")
        monitor.record(kind="query", query=request.query, user=user.subject,
                       tenant_id=user.tenant_id,
                       latency_ms=result["execution_time_ms"],
                       savings_pct=result["tokens"]["savings_percent"])
        return {"status": "success", **result}
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 - report and wrap; never leak internals
        prom.errors_total.inc(kind="query")
        logger.exception("query failed: %s", request.query)
        raise HTTPException(status_code=500, detail="Internal Server Error") from None


def _record_query_metrics(result: dict, kind: str) -> None:
    """Prometheus observations shared by the one-shot and streamed paths."""
    prom.requests_total.inc(kind=kind)
    if result.get("execution_time_ms") is not None:
        prom.query_latency.observe(result["execution_time_ms"] / 1000.0)
    tokens = result.get("tokens") or {}
    if tokens.get("savings_percent") is not None:
        prom.token_savings.observe(tokens["savings_percent"] / 100.0)
    if result.get("cost_usd") is not None:
        prom.llm_cost_total.inc(result["cost_usd"])
    if result.get("answer_fallback"):
        prom.llm_fallbacks_total.inc()


@app.post("/api/v1/query/stream")
async def query_stream(
    request: QueryRequest,
    user: UserIdentity = Depends(require_user(*QUERY_ROLES)),
    _rl=Depends(rate_limit()),
):
    """Streaming variant of /api/v1/query — server-sent events.

    Event sequence: ``meta`` → ``delta``* → ``done`` (or ``blocked`` on
    guardrail refusal / ``error`` on failure). The ``done``/``blocked`` events
    carry the full result (same shape as /api/v1/query) after the audit
    record is written. When PII masking is active for the caller, live token
    streaming is disabled and the answer arrives as a single buffered delta.
    """

    def sse_events():
        try:
            for ev in stream_query(
                app.state.driver,
                request.query,
                max_hops=request.max_hops,
                token_budget=request.token_budget,
                reranker_mode=request.reranker_mode,
                answer_mode=request.answer_mode,
                identity=user,
            ):
                if ev["type"] in ("done", "blocked"):
                    _record_query_metrics(ev.get("result") or {}, kind="stream")
                    monitor.record(kind="stream", query=request.query,
                                   user=user.subject, tenant_id=user.tenant_id,
                                   latency_ms=(ev.get("result") or {}).get("execution_time_ms"),
                                   savings_pct=((ev.get("result") or {}).get("tokens") or {}).get("savings_percent"))
                yield f"event: {ev['type']}\ndata: {json.dumps(ev, default=str)}\n\n"
        except Exception as exc:  # noqa: BLE001 - streamed to the client
            prom.errors_total.inc(kind="stream")
            logger.exception("streamed query failed: %s", request.query)
            yield ("event: error\ndata: "
                   + json.dumps({"detail": "Internal Server Error", "error": type(exc).__name__})
                   + "\n\n")

    return StreamingResponse(
        sse_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/v1/session")
async def get_session(user: UserIdentity = Depends(require_user(*QUERY_ROLES))):
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
    user: UserIdentity = Depends(require_user(*SESSION_ROLES)),
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
async def get_metrics(user: UserIdentity = Depends(require_user(*METRICS_ROLES))):
    """Rolling request stats: counts, avg latency, avg token savings."""
    return {"summary": monitor.summary(), "metrics": monitor.recent(100)}


@app.get("/api/v1/audit")
async def get_audit(limit: int = 50,
                    user: UserIdentity = Depends(require_user(*AUDIT_ROLES))):
    """Recent query audit records (Shot 3 explainability trail)."""
    return {"status": "success", "records": audit_store.recent(min(limit, 500))}


@app.get("/api/v1/audit/verify")
async def verify_audit(user: UserIdentity = Depends(require_user(*AUDIT_ROLES))):
    """Verify the hash-chain integrity of the audit store (v2)."""
    return {"status": "success", **audit_store.verify()}


# ---------------------------------------------------------------------------
# v2 durable jobs (WS-A, G11)
# ---------------------------------------------------------------------------

@app.post("/api/v1/jobs")
async def submit_job(
    request: JobSubmitRequest,
    user: UserIdentity = Depends(require_user(*SESSION_ROLES)),
    _rl=Depends(rate_limit()),
):
    """Enqueue a background job (session switch / benchmark)."""
    try:
        job_id = await asyncio.to_thread(
            get_store().submit, request.kind, request.params
        )
    except JobError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    prom.requests_total.inc(kind="job")
    return {"status": "success", "job_id": job_id,
            "kind": request.kind, "job": get_store().get(job_id)}


@app.get("/api/v1/jobs")
async def list_jobs(limit: int = 20,
                    user: UserIdentity = Depends(require_user(*AUDIT_ROLES))):
    """Recent background jobs (newest first)."""
    return {"status": "success",
            "jobs": await asyncio.to_thread(get_store().list, min(limit, 100))}


@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str,
                  user: UserIdentity = Depends(require_user(*AUDIT_ROLES))):
    """One job's status, progress lines, and result."""
    job = await asyncio.to_thread(get_store().get, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job '{job_id}'")
    return {"status": "success", "job": job}


@app.post("/api/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str,
                     user: UserIdentity = Depends(require_user(*SESSION_ROLES)),
                     _rl=Depends(rate_limit())):
    """Cancel a pending job (or flag a running one for cooperative cancel)."""
    ok = await asyncio.to_thread(get_store().cancel, job_id)
    if not ok:
        raise HTTPException(status_code=409,
                            detail="job not found or already finished")
    return {"status": "success", "job_id": job_id}


# ---------------------------------------------------------------------------
# v2 extraction review queue (WS-C, G17)
# ---------------------------------------------------------------------------

@app.get("/api/v1/review")
async def list_review(status_filter: str | None = None, limit: int = 100,
                      user: UserIdentity = Depends(require_user(*QUERY_ROLES))):
    """Extraction review items (default: all statuses, newest first).

    ``status_filter`` in pending | approved | rejected. All roles may read;
    decisions are analyst/admin (approve/reject endpoints).
    """
    status_filter = status_filter if status_filter in \
        ("pending", "approved", "rejected") else None
    store = get_review_store()
    return {
        "status": "success",
        "summary": store.summary(),
        "items": await asyncio.to_thread(store.list, status_filter,
                                         min(limit, 500)),
    }


@app.post("/api/v1/review/{review_id}/approve")
async def approve_review(review_id: str,
                         user: UserIdentity = Depends(require_user(*UPLOAD_ROLES)),
                         _rl=Depends(rate_limit())):
    """Approve a pending extraction — applies it to the graph via CDC
    (entity add + snapshot merge + cache-revision bump, one transaction)."""
    store = get_review_store()
    item = await asyncio.to_thread(store.get, review_id)
    if item is None:
        raise HTTPException(status_code=404,
                            detail=f"unknown review item '{review_id}'")
    if item["status"] != STATUS_PENDING:
        raise HTTPException(status_code=409,
                            detail=f"review item already {item['status']}")
    try:
        stats = await asyncio.to_thread(apply_review_item, app.state.driver, item)
    except Exception:  # noqa: BLE001 - report and wrap; never leak internals
        logger.exception("review approve failed %s", review_id)
        raise HTTPException(status_code=500,
                            detail="Failed to apply review item (see logs)") from None
    store.decide(review_id, "approved", user.subject)
    prom.requests_total.inc(kind="review")
    return {"status": "success", "review_id": review_id,
            "update_stats": stats, "item": store.get(review_id)}


@app.post("/api/v1/review/{review_id}/reject")
async def reject_review(review_id: str,
                        user: UserIdentity = Depends(require_user(*UPLOAD_ROLES)),
                        _rl=Depends(rate_limit())):
    """Reject a pending extraction — discarded, no graph write."""
    store = get_review_store()
    item = await asyncio.to_thread(store.get, review_id)
    if item is None:
        raise HTTPException(status_code=404,
                            detail=f"unknown review item '{review_id}'")
    if item["status"] != STATUS_PENDING:
        raise HTTPException(status_code=409,
                            detail=f"review item already {item['status']}")
    store.decide(review_id, "rejected", user.subject)
    prom.requests_total.inc(kind="review")
    return {"status": "success", "review_id": review_id,
            "item": store.get(review_id)}


@app.get("/metrics")
async def metrics(user: UserIdentity = Depends(require_user(*METRICS_ROLES))):
    """Prometheus exposition (v2) — scrape target for Grafana/Datadog agents."""
    return PlainTextResponse(
        prom.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT, log_level="info")
