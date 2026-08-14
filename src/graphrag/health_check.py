"""Health checks (Phase 5): liveness + readiness probes.

* ``/health/live`` — process is up (always 200).
* ``/health/ready`` — dependencies healthy: Neo4j reachable AND (when
  configured) Ollama reachable. Returns 503 with a per-dependency breakdown
  so orchestrators (docker-compose healthcheck, k8s) know what is down.

The driver is passed in from the app so the same connection pool is probed.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import FastAPI, HTTPException, Request

from graphrag.config import settings

logger = logging.getLogger("graphrag.api")


def _neo4j_ok(driver) -> bool:
    try:
        with driver.session() as session:
            session.run("RETURN 1").consume()
        return True
    except Exception as exc:  # noqa: BLE001 - probe must never raise
        logger.warning("neo4j health probe failed: %s", exc)
        return False


def _ollama_ok() -> bool:
    try:
        return httpx.get(f"{settings.LLAMA_API_URL}/api/tags", timeout=2).status_code == 200
    except Exception as exc:  # noqa: BLE001 - probe must never raise
        logger.warning("ollama health probe failed: %s", exc)
        return False


def _driver_from_request(request) -> "object | None":
    """The app's Neo4j driver, when present on ``app.state``."""
    return getattr(request.app.state, "driver", None)


def register_health_endpoints(app: FastAPI, driver=None) -> None:
    """Register liveness/readiness routes.

    ``driver`` is optional: when omitted (the common case) the handlers read
    the driver from ``request.app.state.driver`` at request time, so the
    routes can be registered at module scope — no lifespan coupling.
    """
    @app.get("/health/live")
    async def liveness():
        """Liveness: the process is up and serving."""
        return {"status": "alive"}

    @app.get("/health/ready")
    async def readiness(request: Request):
        """Readiness: all required dependencies are reachable."""
        resolved = driver or _driver_from_request(request)
        deps = {"neo4j": _neo4j_ok(resolved)} if resolved is not None else {"neo4j": False}
        if settings.LLAMA_API_URL:
            deps["ollama"] = _ollama_ok()
        if all(deps.values()):
            return {"status": "ready", **deps}
        raise HTTPException(status_code=503, detail={"status": "not_ready", **deps})
