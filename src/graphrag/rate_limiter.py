"""In-memory sliding-window rate limiter (Phase 5).

Production note: this limiter is per-process (a threaded sliding window keyed
by client identity). That is the right trade-off for the demo — zero
infrastructure, works in tests and single-replica deployments. At multi-node
scale it must be swapped for a shared store (Redis) via the same interface:
``check(identity) -> bool``.

The window is a simple timestamp queue per identity; expired timestamps are
pruned lazily on each check, so the structure cannot grow without bound.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from graphrag.config import settings

_WINDOWS: dict[str, deque[float]] = defaultdict(deque)
_LOCK = threading.Lock()
# hard cap on distinct identities; beyond it, empty windows are swept so a
# long-running server with many clients cannot grow memory without bound
_MAX_IDENTITIES = 10_000


def client_identity(request: Request) -> str:
    """Client identity for rate limiting.

    ``X-Forwarded-For`` is only trusted when the deployment sits behind a
    proxy that is configured to set it (``TRUST_PROXY_FORWARDED_FOR=true``) —
    otherwise any client could spoof the header to reset its own window.
    """
    if settings.TRUST_PROXY_FORWARDED_FOR:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def check(identity: str, limit: int | None = None,
          window_s: int | None = None) -> bool:
    """True if ``identity`` is within its window's allowance.

    ``limit`` requests per ``window_s`` seconds (defaults from settings).
    Expired entries are pruned while we are in here, keeping memory bounded.
    """
    limit = limit or settings.RATE_LIMIT_PER_MINUTE
    window_s = window_s or settings.RATE_LIMIT_WINDOW_S
    now = time.monotonic()
    cutoff = now - window_s
    with _LOCK:
        q = _WINDOWS[identity]
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= limit:
            return False
        if len(_WINDOWS) > _MAX_IDENTITIES and not q:
            _sweep_empty()
        q.append(now)
        return True


def _sweep_empty() -> None:
    """Drop identities with fully-expired windows (caller holds the lock)."""
    for ident in [k for k, v in _WINDOWS.items() if not v]:
        del _WINDOWS[ident]


def rate_limit(limit: int | None = None, window_s: int | None = None):
    """FastAPI dependency enforcing the sliding window for the client.

    Usage: ``async def route(_, _rl: None = Depends(rate_limit()))``
    """
    def dependency(request: Request) -> None:
        if not check(client_identity(request), limit=limit, window_s=window_s):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded — slow down",
            )
    return dependency
