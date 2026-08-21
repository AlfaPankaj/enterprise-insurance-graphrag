"""Global exception handling (Phase 5).

Uncaught exceptions become structured 500s with a correlation id instead of
stack traces leaking to clients; validation errors stay 422s with a clean
body. ``register_exception_handlers`` wires both onto the app.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("graphrag.api")


def _request_id(request: Request) -> str:
    # the security middleware stamps request.state.request_id on every path;
    # fall back to the header (or a fresh id) for bare TestClient usage
    return (getattr(request.state, "request_id", None)
            or request.headers.get("X-Request-ID")
            or uuid.uuid4().hex[:12])


def _register_general(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):  # noqa: ANN001
        rid = _request_id(request)
        logger.exception("unhandled error request_id=%s: %s", rid, exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "request_id": rid,
                # never leak exception internals to clients — the detail is
                # in the server logs under the request_id
                "detail": "See server logs (request_id " + rid + ")",
            },
        )


def _register_validation(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):  # noqa: ANN001
        logger.warning("validation error request_id=%s: %s",
                       _request_id(request), exc.errors()[:3])
        return JSONResponse(
            status_code=422,
            content={"error": "Validation Error", "detail": exc.errors()[:5]},
        )


def register_exception_handlers(app: FastAPI) -> None:
    _register_general(app)
    _register_validation(app)
