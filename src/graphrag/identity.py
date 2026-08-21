"""Identity & role-based access control (v2 — WS-B, G1).

Every request resolves to a ``UserIdentity``:

* ``AUTH_MODE=none``     — dev only: anonymous identity with full roles
  (preserves the v1 open API for local development)
* ``AUTH_MODE=static``   — ``X-API-Key`` (or any time ``API_KEY`` is set),
  identity ``api-key`` with the roles from ``API_KEY_ROLES``
* ``AUTH_MODE=jwt``      — OIDC bearer token: HS256 shared secret
  (``JWT_SECRET``) or RS256 key discovery (``JWKS_URL``); roles come from the
  ``roles`` claim (or Keycloak's ``realm_access.roles``), tenant from
  ``tenant_id`` (falls back to ``DEFAULT_TENANT``)

``require_user(*roles)`` is the FastAPI dependency used by every protected
endpoint; it also exposes ``request.state.user`` so handlers, the audit
trail, and PII masking can act on *who* is asking.
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Iterable

import jwt
from fastapi import HTTPException, Request, status

from graphrag.config import settings

logger = logging.getLogger("graphrag.identity")

ROLE_ADMIN = "admin"
ROLE_ANALYST = "analyst"
ROLE_AUDITOR = "auditor"

# Endpoint role policies (v2). Dev (AUTH_MODE=none) grants every role to the
# anonymous identity, so the open local API keeps working exactly like v1.
QUERY_ROLES = (ROLE_ADMIN, ROLE_ANALYST, ROLE_AUDITOR)
UPLOAD_ROLES = (ROLE_ADMIN, ROLE_ANALYST)
SESSION_ROLES = (ROLE_ADMIN,)
AUDIT_ROLES = (ROLE_ADMIN, ROLE_AUDITOR)
METRICS_ROLES = (ROLE_ADMIN, ROLE_AUDITOR)


@dataclass(frozen=True)
class UserIdentity:
    """Who is making this request — subject, roles, tenant."""

    subject: str
    roles: frozenset[str]
    tenant_id: str
    auth_method: str = "none"

    def has_role(self, *roles: str) -> bool:
        return any(r in self.roles for r in roles)

    def has_any(self, roles: Iterable[str]) -> bool:
        return any(r in self.roles for r in roles)

    def as_dict(self) -> dict:
        return {
            "subject": self.subject,
            "roles": sorted(self.roles),
            "tenant_id": self.tenant_id,
            "auth_method": self.auth_method,
        }


@dataclass
class _JwksCache:
    client: object | None = None
    fetched_at: float = 0.0
    ttl_s: float = 300.0

    def get(self):
        now = time.monotonic()
        if self.client is None or now - self.fetched_at > self.ttl_s:
            from jwt import PyJWKClient

            self.client = PyJWKClient(settings.JWKS_URL)
            self.fetched_at = now
        return self.client


_jwks = _JwksCache()


def _anonymous() -> UserIdentity:
    """Dev identity: everything allowed, default tenant."""
    return UserIdentity(
        subject="anonymous",
        roles=frozenset({ROLE_ADMIN, ROLE_ANALYST, ROLE_AUDITOR}),
        tenant_id=settings.DEFAULT_TENANT,
        auth_method="none",
    )


def _static_identity() -> UserIdentity:
    return UserIdentity(
        subject="api-key",
        roles=frozenset(r.strip() for r in settings.API_KEY_ROLES.split(",") if r.strip()),
        tenant_id=settings.DEFAULT_TENANT,
        auth_method="static",
    )


def _claims_identity(claims: dict) -> UserIdentity:
    roles: set[str] = set(claims.get("roles") or [])
    # Keycloak-style realm roles
    realm = claims.get("realm_access") or {}
    roles.update(realm.get("roles") or [])
    if not roles:
        roles.add(ROLE_ANALYST)  # authenticated but unprivileged
    subject = str(claims.get("sub") or claims.get("preferred_username") or "unknown")
    return UserIdentity(
        subject=subject,
        roles=frozenset(roles),
        tenant_id=str(claims.get("tenant_id") or settings.DEFAULT_TENANT),
        auth_method="jwt",
    )


def _verify_token(token: str) -> dict:
    """Validate a JWT and return its claims; raise jwt.PyJWTError."""
    if settings.JWKS_URL:
        key = _jwks.get().get_signing_key_from_jwt(token).key
        return jwt.decode(
            token, key, algorithms=["RS256"],
            issuer=settings.JWT_ISSUER or None,
            audience=settings.JWT_AUDIENCE or None,
        )
    if not settings.JWT_SECRET:
        raise jwt.PyJWTError("AUTH_MODE=jwt but no JWT_SECRET / JWKS_URL configured")
    return jwt.decode(
        token, settings.JWT_SECRET, algorithms=["HS256"],
        issuer=settings.JWT_ISSUER or None,
        audience=settings.JWT_AUDIENCE or None,
    )


def effective_auth_mode() -> str:
    """The auth mode actually in force (API_KEY set = static even in 'none')."""
    mode = settings.AUTH_MODE.lower()
    if mode in ("jwt", "static"):
        return mode
    return "static" if settings.API_KEY else "none"


def api_key_valid(key: str | None) -> bool:
    """Constant-time comparison against the configured static key."""
    return bool(settings.API_KEY and key and
                secrets.compare_digest(key, settings.API_KEY))


def identity_from_request(request: Request) -> UserIdentity:
    """Resolve the caller identity from the request (never raises)."""
    mode = effective_auth_mode()
    if mode == "jwt":
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            try:
                return _claims_identity(_verify_token(auth[7:].strip()))
            except jwt.PyJWTError as exc:
                logger.warning("JWT rejected: %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token",
                    headers={"WWW-Authenticate": "Bearer"},
                ) from exc
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if mode == "static":
        if api_key_valid(request.headers.get("x-api-key")):
            return _static_identity()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key (send X-API-Key header)",
        )
    return _anonymous()


def require_user(*allowed_roles: str):
    """FastAPI dependency: authenticate, authorize, and expose the identity.

    Usage: ``async def route(request: Request, user=Depends(require_user("admin")))``.
    ``request.state.user`` is set for downstream PII/audit logic. In dev
    (``AUTH_MODE=none`` with no API key) the anonymous identity carries every
    role, so the open local API behaves exactly like v1.
    """

    def dependency(request: Request) -> UserIdentity:
        user = identity_from_request(request)
        if allowed_roles and not user.has_any(allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {' or '.join(allowed_roles)}",
            )
        request.state.user = user
        return user

    return dependency
