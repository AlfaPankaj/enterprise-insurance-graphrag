"""Process-wide HTTP connection pools for latency-sensitive model calls.

The query path is intentionally synchronous inside a worker thread so the same
provider implementation serves Streamlit, FastAPI, jobs, and CLI callers.
``httpx.Client`` is thread-safe and reuses TCP/TLS connections across those
workers. Request payloads, timeouts, and response parsing remain unchanged.

The small compatibility check preserves existing integrations/tests that
monkeypatch ``httpx.get/post/stream`` at module level: a patched function is
called directly instead of bypassed through the pooled client.
"""

from __future__ import annotations

import atexit
import threading

import httpx

from graphrag.config import settings

_ORIGINAL_GET = httpx.get
_ORIGINAL_POST = httpx.post
_ORIGINAL_STREAM = httpx.stream
_CLIENT: httpx.Client | None = None
_CLIENT_LOCK = threading.Lock()


def get_http_client() -> httpx.Client:
    """Return the shared sync client, creating it lazily and thread-safely."""
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                max_connections = max(1, settings.HTTP_POOL_MAX_CONNECTIONS)
                max_keepalive = min(
                    max_connections,
                    max(1, settings.HTTP_POOL_MAX_KEEPALIVE_CONNECTIONS),
                )
                _CLIENT = httpx.Client(
                    limits=httpx.Limits(
                        max_connections=max_connections,
                        max_keepalive_connections=max_keepalive,
                        keepalive_expiry=max(
                            1.0, settings.HTTP_POOL_KEEPALIVE_EXPIRY_S
                        ),
                    )
                )
    return _CLIENT


def request_get(url: str, **kwargs):
    """Pooled GET, while honoring a module-level httpx test/integration seam."""
    if httpx.get is not _ORIGINAL_GET:
        return httpx.get(url, **kwargs)
    return get_http_client().get(url, **kwargs)


def request_post(url: str, **kwargs):
    """Pooled POST, while honoring a module-level httpx test/integration seam."""
    if httpx.post is not _ORIGINAL_POST:
        return httpx.post(url, **kwargs)
    return get_http_client().post(url, **kwargs)


def request_stream(method: str, url: str, **kwargs):
    """Pooled streaming request context manager."""
    if httpx.stream is not _ORIGINAL_STREAM:
        return httpx.stream(method, url, **kwargs)
    return get_http_client().stream(method, url, **kwargs)


def close_http_client() -> None:
    """Close and clear the pool (application shutdown and tests)."""
    global _CLIENT
    with _CLIENT_LOCK:
        client, _CLIENT = _CLIENT, None
    if client is not None:
        client.close()


atexit.register(close_http_client)
