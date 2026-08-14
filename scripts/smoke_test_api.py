"""Live-server smoke test for the Phase 2 CDC API.

Starts the real uvicorn server in a subprocess, waits for readiness, then:
  * GET  /health/ready   -> liveness/readiness probes
  * POST /api/v1/upload  -> a real PDF through the live HTTP stack
  * GET  /api/v1/metrics -> the in-memory metrics log

Usage:  .venv/Scripts/python.exe scripts/smoke_test_api.py
Requires: Neo4j running (docker start graphrag-neo4j) + seeded baseline.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PORT = 8011
BASE = f"http://127.0.0.1:{PORT}"


def main() -> int:
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "graphrag.api_server:app",
         "--app-dir", str(PROJECT_ROOT / "src"), "--host", "127.0.0.1", "--port", str(PORT)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        # wait for readiness (max ~30s)
        for _ in range(60):
            if server.poll() is not None:
                raise SystemExit(f"uvicorn exited early:\n{server.stdout.read()}")
            try:
                if httpx.get(f"{BASE}/health/live", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            raise SystemExit("server did not become ready in time")

        ready = httpx.get(f"{BASE}/health/ready", timeout=5)
        print(f"GET /health/ready            -> {ready.status_code} {ready.json()}")

        pdf = (PROJECT_ROOT / "data" / "pdfs" / "policy_POL-0009.pdf").read_bytes()
        resp = httpx.post(
            f"{BASE}/api/v1/upload",
            files={"file": ("policy_POL-0009.pdf", pdf, "application/pdf")},
            timeout=60,
        )
        body = resp.json()
        print(f"POST /api/v1/upload          -> {resp.status_code}")
        print("    update_stats:", body.get("update_stats"))
        assert resp.status_code == 200, body
        assert body["update_stats"]["entities_added"] == 0, body  # idempotent

        metrics = httpx.get(f"{BASE}/api/v1/metrics", timeout=5).json()
        print(f"GET /api/v1/metrics          -> {len(metrics['metrics'])} entries")

        print("\nLive-server smoke test: ALL CHECKS PASSED")
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    sys.exit(main())
