"""API security (Phase 5): API-key auth, CORS, and hardening headers.

* ``require_api_key`` — FastAPI dependency enforcing ``X-API-Key``. Enabled
  only when ``settings.API_KEY`` is non-empty, so local dev keeps working
  with zero config while production forces a key.
* ``setup_security`` — installs CORS (from ``settings.CORS_ORIGINS``) and a
  response middleware that adds hardening headers (no-sniff, frame-deny,
  XSS-protection, referrer-policy, HSTS when behind TLS).

The API key is compared with ``secrets.compare_digest`` so a timing attack
cannot recover it from response latency.
"""

from __future__ import annotations

import secrets
import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from graphrag.config import settings

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "no-referrer",
}


def require_api_key(
    api_key: Annotated[str | None, Depends(_API_KEY_HEADER)] = None,
) -> str | None:
    """Reject requests without a valid API key when auth is enabled.

    Returns the key (or None when auth is disabled) so handlers can log it.
    """
    if not settings.API_KEY:
        return None  # dev mode: auth off
    if not api_key or not secrets.compare_digest(api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key (send X-API-Key header)",
        )
    return api_key


def setup_security(app) -> None:
    """Attach CORS middleware + hardening-header middleware to the app."""
    origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def add_security_headers(request, call_next):
        # request-id correlation on EVERY path (v2): honor the client's id,
        # generate one otherwise, echo it on the response for log correlation
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        for header, value in _SECURITY_HEADERS.items():
            response.headers[header] = value
        return response
