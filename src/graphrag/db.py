"""Single place where the graph driver is constructed.

The reference implementation is Neo4j; Option A
(``docs/DEPLOYMENT_SCALE_OPTIONS.md``) swaps in the PostgreSQL/Apache AGE
backend via ``GRAPH_BACKEND=age``. Every entry point (API lifespan, audit
UI, scripts) opens its driver here so the backend is a configuration
choice, not a code change.
"""

from __future__ import annotations

import logging

from graphrag.config import settings

logger = logging.getLogger("graphrag.db")


def open_driver():
    """Return a graph driver per ``GRAPH_BACKEND`` (Neo4j by default)."""
    backend = (settings.GRAPH_BACKEND or "neo4j").strip().lower()
    if backend == "age":
        from graphrag.postgres_backend import AgeDriver  # noqa: PLC0415
        if not settings.POSTGRES_DSN:
            raise RuntimeError(
                "GRAPH_BACKEND=age requires POSTGRES_DSN "
                "(e.g. postgresql://graphrag:graphrag@localhost:5432/graphrag)")
        logger.info("graph backend: PostgreSQL/Apache AGE (graph=%r)",
                    settings.AGE_GRAPH_NAME)
        return AgeDriver()
    if backend != "neo4j":
        raise ValueError(f"unknown GRAPH_BACKEND: {settings.GRAPH_BACKEND!r} "
                         "(expected 'neo4j' or 'age')")

    from neo4j import GraphDatabase  # noqa: PLC0415
    driver = GraphDatabase.driver(
        settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )
    logger.info("graph backend: Neo4j (%s)", settings.NEO4J_URI)
    return driver


def backend_name() -> str:
    return (settings.GRAPH_BACKEND or "neo4j").strip().lower()
