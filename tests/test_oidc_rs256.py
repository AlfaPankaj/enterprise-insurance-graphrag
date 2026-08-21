"""OIDC integration test — RS256 + JWKS discovery against a local IdP stub.

Simulates the full enterprise flow without an external provider: an RSA
keypair acts as the IdP's signing key, a threaded HTTP server serves its JWKS
document (what Keycloak/Azure AD/Okta expose), and the app verifies RS256
tokens end to end through the ``require_user`` dependency.

* valid token (matching kid)      -> 200, roles/tenant resolved
* token signed by an unknown key  -> 401
* token with an unknown kid       -> 401
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from graphrag.config import settings
from graphrag.identity import require_user

KID = "oidc-test-kid"
ISSUER = "https://idp.example"


# ---------------------------------------------------------------------------
# local JWKS stub
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def idp():
    """(jwks_url, private_pem, second_private_pem) — a mini OIDC provider."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    jwk = json.loads(pyjwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update({"kid": KID, "alg": "RS256", "use": "sig"})
    body = json.dumps({"keys": [jwk]}).encode()

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # quiet
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield (f"http://127.0.0.1:{server.server_address[1]}/jwks",
           key.private_bytes(serialization.Encoding.PEM,
                             serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption()),
           other.private_bytes(serialization.Encoding.PEM,
                               serialization.PrivateFormat.PKCS8,
                               serialization.NoEncryption()))
    server.shutdown()
    server.server_close()


def _token(private_pem: bytes, claims: dict | None = None, kid: str = KID) -> str:
    claims = claims or {"sub": "analyst@corp", "roles": ["analyst"],
                        "tenant_id": "bank-a", "iss": ISSUER}
    return pyjwt.encode(claims, private_pem, algorithm="RS256", headers={"kid": kid})


def _app():
    app = FastAPI()

    @app.get("/whoami")
    def whoami(user=Depends(require_user())):
        return {"subject": user.subject, "roles": sorted(user.roles),
                "tenant": user.tenant_id}

    @app.get("/admin")
    def admin(user=Depends(require_user("admin"))):
        return {"ok": True}

    return app


@pytest.fixture()
def jwt_mode(monkeypatch, idp):
    url, *_ = idp
    monkeypatch.setattr(settings, "AUTH_MODE", "jwt")
    monkeypatch.setattr(settings, "JWKS_URL", url)
    monkeypatch.setattr(settings, "JWT_SECRET", "")
    monkeypatch.setattr(settings, "JWT_ISSUER", ISSUER)
    monkeypatch.setattr(settings, "JWT_AUDIENCE", "")
    # fresh JWKS client per test (module-level cache has its own TTL)
    from graphrag import identity as identity_mod
    identity_mod._jwks.client = None
    return url


# ---------------------------------------------------------------------------
# the flow
# ---------------------------------------------------------------------------

def test_valid_rs256_token_resolves_identity(jwt_mode, idp):
    _url, private_pem, _ = idp
    client = TestClient(_app())
    resp = client.get("/whoami",
                      headers={"Authorization": f"Bearer {_token(private_pem)}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["subject"] == "analyst@corp"
    assert body["roles"] == ["analyst"]
    assert body["tenant"] == "bank-a"


def test_roles_enforced_after_oidc_login(jwt_mode, idp):
    _url, private_pem, _ = idp
    client = TestClient(_app())
    analyst = _token(private_pem)
    admin = _token(private_pem, claims={"sub": "admin@corp", "roles": ["admin"],
                                        "iss": ISSUER})
    assert client.get("/admin",
                      headers={"Authorization": f"Bearer {analyst}"}).status_code == 403
    assert client.get("/admin",
                      headers={"Authorization": f"Bearer {admin}"}).status_code == 200


def test_token_signed_by_unknown_key_rejected(jwt_mode, idp):
    _url, _, other_pem = idp
    client = TestClient(_app())
    forged = _token(other_pem)  # valid-looking claims, wrong signing key
    assert client.get("/whoami",
                      headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_token_with_unknown_kid_rejected(jwt_mode, idp):
    _url, private_pem, _ = idp
    client = TestClient(_app())
    unknown = _token(private_pem, kid="unknown-kid")
    assert client.get("/whoami",
                      headers={"Authorization": f"Bearer {unknown}"}).status_code == 401


def test_missing_token_401(jwt_mode):
    client = TestClient(_app())
    assert client.get("/whoami").status_code == 401


# ---------------------------------------------------------------------------
# identity_from_token (the UI/scripts entry point)
# ---------------------------------------------------------------------------

def test_identity_from_token_helper(jwt_mode, idp):
    from graphrag.identity import identity_from_token

    _url, private_pem, _ = idp
    user = identity_from_token(_token(private_pem))
    assert user.subject == "analyst@corp"
    assert user.has_role("analyst")
    assert user.tenant_id == "bank-a"

    with pytest.raises(Exception):
        identity_from_token("not-a-jwt")
