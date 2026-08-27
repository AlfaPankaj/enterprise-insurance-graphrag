"""Durable job runner (v2 — WS-A, G11).

Seeding and benchmark runs become **tracked jobs** instead of in-process
daemon threads: they live in a small SQLite store (``settings.JOB_DB_PATH``),
survive process restarts, stream progress lines, and expose status through
``POST/GET /api/v1/jobs``.

Lifecycle: ``pending → running → succeeded | failed | cancelled``.
Crash recovery: on startup, jobs left ``running`` by a dead process are
marked ``interrupted`` (they can be resubmitted idempotently — every handler
must be safe to re-run).

Handlers are registered per ``kind``; the default set (``session_switch``,
``benchmark``, ``fraud_benchmark``) is wired by the API lifespan. Handlers
receive ``(job, params, progress_cb, cancel_cb)`` and return a JSON-able
result. Workers are daemon threads — the store, not the thread, is the
source of truth.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from graphrag.config import settings
from graphrag.prometheus import jobs_completed_total, jobs_running

logger = logging.getLogger("graphrag.jobs")

ROOT = Path(__file__).resolve().parents[2]

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_INTERRUPTED = "interrupted"
TERMINAL = {STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELLED, STATUS_INTERRUPTED}

_MAX_PROGRESS_LINES = 40
_MAX_LINE_LEN = 200


class JobCancelled(Exception):
    """Raised inside a handler when its cancel flag is set."""


class JobError(Exception):
    """A handler failed with a user-facing message."""


class JobStore:
    """SQLite-backed job registry. Thread-safe; one store per process."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cancel_flags: set[str] = set()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS jobs ("
            " id TEXT PRIMARY KEY,"
            " kind TEXT NOT NULL,"
            " params TEXT NOT NULL,"
            " status TEXT NOT NULL,"
            " progress TEXT NOT NULL DEFAULT '[]',"
            " error TEXT,"
            " result TEXT,"
            " created_at TEXT NOT NULL,"
            " started_at TEXT,"
            " finished_at TEXT)"
        )
        self._conn.commit()
        # crash recovery: anything still running was abandoned by a dead process
        with self._conn:
            self._conn.execute(
                "UPDATE jobs SET status=?, finished_at=? WHERE status=?",
                (STATUS_INTERRUPTED, _now(), STATUS_RUNNING),
            )

    # ------------------------------------------------------------- queries
    def _row_to_job(self, row: sqlite3.Row) -> dict:
        job = dict(zip(row.keys(), row))
        try:
            job["params"] = json.loads(job["params"] or "{}")
        except ValueError:
            job["params"] = {}
        try:
            job["progress"] = json.loads(job["progress"] or "[]")
        except ValueError:
            job["progress"] = []
        try:
            job["result"] = json.loads(job["result"]) if job["result"] else None
        except ValueError:
            job["result"] = job["result"]
        return job

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (int(limit),)).fetchall()
        return [self._row_to_job(r) for r in rows]

    def clear(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM jobs")

    # ------------------------------------------------------------- lifecycle
    def submit(self, kind: str, params: dict | None = None) -> str:
        """Enqueue a job (status pending) and start its worker thread."""
        handler = get_handler(kind)
        if handler is None:
            raise JobError(f"unknown job kind: {kind!r}")
        job_id = uuid.uuid4().hex[:12]
        now = _now()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO jobs (id, kind, params, status, created_at) "
                "VALUES (?,?,?,?,?)",
                (job_id, kind, json.dumps(params or {}), STATUS_PENDING, now),
            )
        threading.Thread(target=self._worker, args=(job_id,),
                         name=f"job-{job_id}", daemon=True).start()
        return job_id

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job (never started) or flag a running one."""
        with self._lock:
            job = self.get(job_id)
            if job is None:
                return False
            if job["status"] == STATUS_PENDING:
                with self._conn:
                    self._conn.execute(
                        "UPDATE jobs SET status=?, finished_at=? WHERE id=?",
                        (STATUS_CANCELLED, _now(), job_id))
                jobs_completed_total.inc(status=STATUS_CANCELLED)
                return True
            if job["status"] == STATUS_RUNNING:
                self._cancel_flags.add(job_id)  # cooperative: handler checks
                return True
            return False  # already terminal

    def _set_progress(self, job_id: str, lines: list[str]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE jobs SET progress=? WHERE id=?",
                (json.dumps(lines[-_MAX_PROGRESS_LINES:]), job_id))

    def _worker(self, job_id: str) -> None:
        handler = None
        with self._lock:
            job = self.get(job_id)
            if not job or job["status"] != STATUS_PENDING:
                return  # cancelled before start
            handler = get_handler(job["kind"])
            with self._conn:
                self._conn.execute(
                    "UPDATE jobs SET status=?, started_at=? WHERE id=?",
                    (STATUS_RUNNING, _now(), job_id))
        if handler is None:  # pragma: no cover - submit validates
            return
        jobs_running.inc()
        lines: list[str] = []

        def progress(line: str) -> None:
            lines.append(str(line)[:_MAX_LINE_LEN])
            self._set_progress(job_id, lines)

        def cancelled() -> bool:
            return job_id in self._cancel_flags

        try:
            result = handler(job, job["params"], progress, cancelled)
            status, payload = STATUS_SUCCEEDED, (json.dumps(result), None)
        except JobCancelled:
            status, payload = STATUS_CANCELLED, (None, None)
        except Exception as exc:  # noqa: BLE001 - recorded on the job row
            logger.warning("job %s failed: %s", job_id, exc)
            status, payload = STATUS_FAILED, (None, str(exc)[:1000])
        finally:
            with self._lock, self._conn:
                self._conn.execute(
                    "UPDATE jobs SET status=?, result=?, error=?, finished_at=? "
                    "WHERE id=?",
                    (status, payload[0], payload[1], _now(), job_id))
            jobs_running.dec()
            self._cancel_flags.discard(job_id)
        jobs_completed_total.inc(status=status)


# ---------------------------------------------------------------------------
# handler registry
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, callable] = {}
_HANDLERS_LOCK = threading.Lock()


def register_handler(kind: str, fn) -> None:
    with _HANDLERS_LOCK:
        _HANDLERS[kind] = fn


def get_handler(kind: str):
    with _HANDLERS_LOCK:
        return _HANDLERS.get(kind)


def register_default_handlers(driver_getter) -> None:
    """Wire the built-in job kinds (idempotent). ``driver_getter()`` must
    return the app's Neo4j driver (lazy — called inside the worker)."""

    def session_switch(job, params, progress, cancelled):
        from graphrag import sessions

        session_id = params.get("session_id")
        if not session_id:
            raise JobError("session_switch requires 'session_id'")
        info = sessions.switch_session(
            driver_getter(), session_id, force=bool(params.get("force")),
            line_cb=progress,
        )
        return {"status": info["status"], "session": session_id}

    def benchmark(job, params, progress, cancelled):
        dataset = params.get("dataset")
        if not dataset:
            raise JobError("benchmark requires 'dataset'")
        return _run_subprocess(
            job, ["python", "scripts/benchmark_real_dataset.py", dataset,
                  "--queries", str(params.get("queries", 100)),
                  "--workers", str(params.get("workers", 4))],
            progress, cancelled)

    def fraud_benchmark(job, params, progress, cancelled):
        dataset = params.get("dataset")
        if not dataset:
            raise JobError("fraud_benchmark requires 'dataset'")
        return _run_subprocess(
            job, ["python", "scripts/benchmark_fraud_detection.py",
                  "--dataset", dataset,
                  "--negatives", str(params.get("negatives", 500))],
            progress, cancelled)

    register_handler("session_switch", session_switch)
    register_handler("benchmark", benchmark)
    register_handler("fraud_benchmark", fraud_benchmark)


def _run_subprocess(job, cmd: list[str], progress, cancelled) -> dict:
    """Run a benchmark script with line streaming + cooperative cancel."""
    import subprocess
    import sys

    proc = subprocess.Popen(
        [sys.executable, "-u", *cmd[1:]], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        if cancelled():
            proc.terminate()
            raise JobCancelled()
        line = line.rstrip()
        lines.append(line)
        progress(line)
    proc.wait()
    if proc.returncode != 0:
        raise JobError(f"subprocess failed (rc={proc.returncode}): "
                       f"{' | '.join(lines[-5:])}")
    return {"returncode": proc.returncode, "output": lines[-10:]}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# process-wide store (lazy: no db file created unless jobs are used)
# ---------------------------------------------------------------------------

_store: JobStore | None = None
_store_lock = threading.Lock()


def get_store() -> JobStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = JobStore(ROOT / settings.JOB_DB_PATH)
        return _store
