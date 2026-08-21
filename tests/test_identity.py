"""v2 identity & RBAC tests (JWT/static/none modes + role gates)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jwt as pyjwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from graphrag.config import settings
from graphrag.identity import (ROLE_ADMIN, ROLE_ANALYST, UserIdentity,
                               _claims_identity, api_key_valid,
                               effective_auth_mode, require_user)


def _make_token(claims: dict, secret: str = "test-secret") -> str:
    return pyjwt.encode(claims, secret, algorithm="HS256")


# ---------------------------------------------------------------------------
# unit pieces
# ---------------------------------------------------------------------------

def test_claims_identity_maps_roles_and_tenant():
    claims = {"sub": "u-42", "roles": ["admin", "analyst"], "tenant_id": "bank-a"}
    user = _claims_identity(claims)
    assert user.subject == "u-42"
    assert user.has_role(ROLE_ADMIN) and user.has_role(ROLE_ANALYST)
    assert user.tenant_id == "bank-a"
    assert user.auth_method == "jwt"


def test_claims_identity_keycloak_realm_roles():
    user = _claims_identity({"sub": "k", "realm_access": {"roles": ["auditor"]}})
    assert user.has_role("auditor")


def test_claims_identity_without_roles_defaults_to_analyst():
    user = _claims_identity({"sub": "bare"})
    assert user.has_role(ROLE_ANALYST)


def test_api_key_valid_constant_time():
    assert api_key_valid(None) is False
    assert api_key_valid("") is False
    monkey = pytest.MonkeyPatch()
    monkey.setattr(settings, "API_KEY", "s3cret")
    try:
        assert api_key_valid("s3cret") is True
        assert api_key_valid("wrong") is False
    finally:
        monkey.undo()


def test_effective_auth_mode_resolution(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_MODE", "none")
    monkeypatch.setattr(settings, "API_KEY", "")
    assert effective_auth_mode() == "none"
    # a set API_KEY upgrades none -> static (v1 enforcement preserved)
    monkeypatch.setattr(settings, "API_KEY", "k")
    assert effective_auth_mode() == "static"
    monkeypatch.setattr(settings, "AUTH_MODE", "jwt")
    assert effective_auth_mode() == "jwt"
    monkeypatch.setattr(settings, "AUTH_MODE", "static")
    assert effective_auth_mode() == "static"


# ---------------------------------------------------------------------------
# require_user dependency on a tiny app
# ---------------------------------------------------------------------------

def _app(role=None):
    app = FastAPI()

    @app.get("/admin")
    def admin_only(user: UserIdentity = Depends(require_user(ROLE_ADMIN))):
        return {"subject": user.subject, "roles": sorted(user.roles)}

    @app.get("/query")
    def query(user: UserIdentity = Depends(require_user(ROLE_ADMIN, ROLE_ANALYST))):
        return {"subject": user.subject}

    return app


def test_dev_mode_anonymous_full_access(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_MODE", "none")
    monkeypatch.setattr(settings, "API_KEY", "")
    client = TestClient(_app())
    assert client.get("/admin").status_code == 200
    body = client.get("/admin").json()
    assert body["subject"] == "anonymous"  # dev identity attributed


def test_jwt_mode_grants_by_roles(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_MODE", "jwt")
    monkeypatch.setattr(settings, "JWT_SECRET", "test-secret")
    monkeypatch.setattr(settings, "JWT_ISSUER", "")
    monkeypatch.setattr(settings, "JWT_AUDIENCE", "")
    client = TestClient(_app())
    analyst = _make_token({"sub": "a-1", "roles": ["analyst"]})
    admin = _make_token({"sub": "a-2", "roles": ["admin"]})
    # analyst can query but not admin-only
    assert client.get("/query", headers={"Authorization": f"Bearer {analyst}"}).status_code == 200
    assert client.get("/admin", headers={"Authorization": f"Bearer {analyst}"}).status_code == 403
    # admin can do both
    assert client.get("/admin", headers={"Authorization": f"Bearer {admin}"}).status_code == 200


def test_jwt_mode_rejects_missing_and_bad_tokens(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_MODE", "jwt")
    monkeypatch.setattr(settings, "JWT_SECRET", "test-secret")
    monkeypatch.setattr(settings, "JWT_ISSUER", "")
    client = TestClient(_app())
    assert client.get("/query").status_code == 401
    assert client.get("/query", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_jwt_issuer_audience_enforced(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_MODE", "jwt")
    monkeypatch.setattr(settings, "JWT_SECRET", "test-secret")
    monkeypatch.setattr(settings, "JWT_ISSUER", "https://idp.example")
    monkeypatch.setattr(settings, "JWT_AUDIENCE", "graphrag")
    client = TestClient(_app())
    good = _make_token({"sub": "ok", "roles": ["admin"], "iss": "https://idp.example",
                        "aud": "graphrag"})
    bad = _make_token({"sub": "bad", "roles": ["admin"], "iss": "https://evil.example"})
    assert client.get("/admin", headers={"Authorization": f"Bearer {good}"}).status_code == 200
    assert client.get("/admin", headers={"Authorization": f"Bearer {bad}"}).status_code == 401


def test_static_mode_identity_and_roles(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_MODE", "static")
    monkeypatch.setattr(settings, "API_KEY", "k-123")
    monkeypatch.setattr(settings, "API_KEY_ROLES", "admin,auditor")
    client = TestClient(_app())
    assert client.get("/admin").status_code == 403  # no key
    resp = client.get("/admin", headers={"X-API-Key": "k-123"})
    assert resp.status_code == 200
    assert resp.json()["subject"] == "api-key"


# ---------------------------------------------------------------------------
# UI/scripts helpers (v2 login gate)
# ---------------------------------------------------------------------------

def test_identity_from_api_key_helper(monkeypatch):
    from graphrag.identity import identity_from_api_key

    monkeypatch.setattr(settings, "API_KEY", "ui-key")
    monkeypatch.setattr(settings, "API_KEY_ROLES", "analyst,auditor")
    user = identity_from_api_key("ui-key")
    assert user is not None
    assert user.subject == "api-key"
    assert user.has_role("analyst") and user.has_role("auditor")
    assert user.tenant_id == settings.DEFAULT_TENANT
    assert identity_from_api_key("wrong") is None


def test_identity_from_token_helper(monkeypatch):
    from graphrag.identity import identity_from_token

    monkeypatch.setattr(settings, "AUTH_MODE", "jwt")
    monkeypatch.setattr(settings, "JWT_SECRET", "test-secret")
    monkeypatch.setattr(settings, "JWT_ISSUER", "")
    monkeypatch.setattr(settings, "JWT_AUDIENCE", "")
    user = identity_from_token(_make_token({"sub": "u-9", "roles": ["admin"],
                                            "tenant_id": "bank-b"}))
    assert user.subject == "u-9" and user.has_role("admin")
    assert user.tenant_id == "bank-b"

    import jwt as _jwt
    with pytest.raises(_jwt.PyJWTError):
        identity_from_token("garbage-token")
