"""Tests for Phase 5 rate limiting (src/graphrag/rate_limiter.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from graphrag.rate_limiter import _WINDOWS, check, rate_limit


def _reset_windows() -> None:
    _WINDOWS.clear()


def test_check_allows_up_to_limit():
    _reset_windows()
    for _ in range(5):
        assert check("u1", limit=5, window_s=60) is True
    assert check("u1", limit=5, window_s=60) is False  # 6th request blocked
    assert check("u2", limit=5, window_s=60) is True   # other identity unaffected


def test_window_expires_and_allows_again(monkeypatch):
    _reset_windows()
    import graphrag.rate_limiter as rl

    now = 1000.0
    monkeypatch.setattr(rl.time, "monotonic", lambda: now)
    for _ in range(3):
        assert check("u1", limit=3, window_s=60) is True
    assert check("u1", limit=3, window_s=60) is False

    monkeypatch.setattr(rl.time, "monotonic", lambda: now + 61)  # window elapsed
    assert check("u1", limit=3, window_s=60) is True


def test_rate_limit_dependency_returns_429():
    _reset_windows()
    app = FastAPI()

    @app.get("/limited")
    def limited(_rl=Depends(rate_limit(limit=2, window_s=60))):
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/limited").status_code == 200
    assert client.get("/limited").status_code == 200
    assert client.get("/limited").status_code == 429
