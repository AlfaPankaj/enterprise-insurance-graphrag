"""Session management — which dataset/pipeline is loaded in Neo4j (Phase 6).

The app can switch between **5 sessions** from the web UI — no terminal needed:

  * **Excel sessions** — real CSVs from ``data/Real_datasets/``, ingested by
    ``scripts/ingest_real_dataset.py``:

      - ``fraud_oracle``       — 15,420 auto claims, 923 fraud labels
      - ``insurance_claims``   — 1,000 claims, 247 fraud labels
      - ``insurance_dataset``  — 13,000 customers + claim amounts (its sibling
        ``data_synthetic.csv`` is the same Kaggle source's synthetic variant,
        so it maps onto the same session)

  * **PDF session** — the synthetic demo graph (``scripts/seed_graph.py``),
    the graph the PDF upload → extraction → CDC pipeline produces.

Two execution modes:

  * ``switch_session`` — **blocking** (used by the REST API): runs the seed
    script synchronously and returns when the graph is ready.
  * ``start_switch`` — **background streaming** (used by the web UI): runs the
    seed script in a daemon thread, appending each log line to a progress
    handle the sidebar can render live instead of a frozen spinner.

Both are idempotent: requesting the already-loaded session is a no-op unless
``force=True``. ``-u`` (unbuffered) is added to the child so log lines stream
in real time. A process-wide lock serializes the actual (re)seed so two
concurrent switches cannot interleave ingests against the same database.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

from graphrag.config import settings
from src.graphrag.fraud_ground_truth import detect_dataset

# Serialize the actual (re)seed: two concurrent switches (e.g. API + web UI)
# must not interleave two ingests against the same database.
_switch_lock = threading.Lock()

# Live progress handles for in-flight background switches: session_id -> handle
_PROGRESS: dict[str, dict] = {}

# Session ids whose auto-benchmark is currently running (custom uploads get a
# ground-truth benchmark run in the background right after they seed, so the
# dashboard's Pipeline Validation fills in without any manual step).
_BENCH_RUNNING: set[str] = set()

ROOT = Path(__file__).resolve().parents[2]

# ``kind`` drives which script seeds the session:
#   excel -> ingest_real_dataset.py (real CSVs)
#   pdf   -> seed_graph.py (synthetic demo graph the PDF pipeline works on)
SESSIONS: list[dict] = [
    {
        "id": "fraud_oracle",
        "label": "fraud_oracle — Excel (real)",
        "kind": "excel",
        "dataset": "fraud_oracle",
        "desc": "15,420 claims · 923 fraud labels",
    },
    {
        "id": "insurance_claims",
        "label": "insurance_claims — Excel (real)",
        "kind": "excel",
        "dataset": "insurance_claims",
        "desc": "1,000 claims · 247 fraud labels",
    },
    {
        "id": "insurance_dataset",
        "label": "insurance_dataset — Excel (real)",
        "kind": "excel",
        "dataset": "insurance_dataset",
        "desc": "13,000 claims (+ data_synthetic variant, same source)",
    },
    {
        "id": "pdf_demo",
        "label": "PDF demo graph — synthetic",
        "kind": "pdf",
        "dataset": "synthetic",
        "desc": "Demo graph from the PDF pipeline (policies, claims, endorsements)",
    },
    {
        "id": "banking_demo",
        "label": "Banking demo — transactions, disputes, AML",
        "kind": "banking",
        "dataset": "banking",
        "desc": "60 customers · 80 accounts · 400 transactions · 30 disputes · 18 AML alerts",
    },
]

SESSION_BY_ID = {s["id"]: s for s in SESSIONS}

# (:Dataset) marker stamped on the loaded graph -> session id
_MARKER_TO_SESSION = {
    "fraud_oracle": "fraud_oracle",
    "insurance_claims": "insurance_claims",
    "insurance_dataset": "insurance_dataset",
    "data_synthetic": "insurance_dataset",  # same Kaggle source, synthetic variant
    "synthetic": "pdf_demo",
    "banking": "banking_demo",
}


def custom_sessions_meta() -> list[dict]:
    """Built-in-style session dicts for the registered custom uploads."""
    from graphrag.custom_sessions import list_custom_sessions

    out: list[dict] = []
    for r in list_custom_sessions():
        out.append({
            "id": r["name"],
            "label": f"{r['name']} — custom ({r['kind'].upper()})",
            "kind": "custom",
            "dataset": r["name"],
            "desc": r.get("note") or f"{r['kind'].upper()} upload — "
                                      f"{len(r['sources'])} file(s)",
        })
    return out


def all_sessions() -> list[dict]:
    """Built-in sessions + any user-uploaded custom sessions."""
    return SESSIONS + custom_sessions_meta()


def get_session_meta(session_id: str) -> dict | None:
    """Session metadata (built-in or custom) for a session id."""
    if session_id in SESSION_BY_ID:
        return SESSION_BY_ID[session_id]
    return next((s for s in custom_sessions_meta() if s["id"] == session_id), None)


def session_exists(session_id: str) -> bool:
    """True if the id is a built-in or registered custom session."""
    return get_session_meta(session_id) is not None


def session_for_marker(marker: str | None) -> str:
    """Session id for the dataset marker stamped on the loaded graph."""
    if not marker:
        return "pdf_demo"  # graph predates the marker -> demo graph
    if marker in _MARKER_TO_SESSION:
        return _MARKER_TO_SESSION[marker]
    # a custom session stamps its own name as the marker
    if get_session_meta(marker) is not None:
        return marker
    return "pdf_demo"


def current_session_id(driver) -> str:
    """Session id currently loaded in Neo4j (best-effort)."""
    try:
        return session_for_marker(detect_dataset(driver))
    except Exception:
        return "pdf_demo"


def _seed_command(session: dict) -> list[str]:
    """The CLI command that (re)seeds the graph for this session.

    ``-u`` (unbuffered) keeps the child's stdout line-buffered so the
    streaming runner can relay progress lines in real time.
    """
    if session["kind"] == "excel":
        return [
            sys.executable, "-u",
            str(ROOT / "scripts" / "ingest_real_dataset.py"),
            session["dataset"],
            "--reset",
        ]
    if session["kind"] == "custom":
        # user-uploaded dataset — reprocess its stored sources
        return [
            sys.executable, "-u",
            str(ROOT / "scripts" / "ingest_custom_dataset.py"),
            session["dataset"],
            "--reset",
        ]
    if session["kind"] == "banking":
        # v2 banking domain demo (WS-E)
        return [
            sys.executable, "-u",
            str(ROOT / "scripts" / "ingest_banking_dataset.py"),
            "--reset",
        ]
    return [
        sys.executable, "-u",
        str(ROOT / "scripts" / "seed_graph.py"),
        "--reset",
        "--apply-schema",
        *(["--tenant", settings.DEFAULT_TENANT]
          if settings.TENANT_MODE == "column" else []),
    ]


def benchmark_running(session_id: str) -> bool:
    """True while an auto-benchmark for ``session_id`` is in flight."""
    return session_id in _BENCH_RUNNING


def _maybe_auto_benchmark(session_id: str) -> None:
    """Kick off ground-truth benchmarks for a custom CSV session, in the
    background, when its results are missing.

    Runs, in order: ``scripts/benchmark_real_dataset.py --custom-session
    <name>`` (retrieval/pruning accuracy, token savings) and, when the CSV has
    a fraud column, ``scripts/benchmark_fraud_detection.py --custom-session
    <name>`` (precision/recall over every fraud label). Both write
    ``data/benchmarks/real_<name>.json`` / ``fraud_detection_<name>.json``,
    which the dashboard's Pipeline Validation table picks up on its next
    refresh. Only triggered once per session (results file absent); re-seeding
    a custom session that already has results does NOT re-benchmark.
    """
    meta = get_session_meta(session_id)
    if not meta or meta.get("kind") != "custom":
        return
    if benchmark_running(session_id):
        return
    bfile = ROOT / "data" / "benchmarks" / f"real_{session_id}.json"
    if bfile.exists():
        return

    def work() -> None:
        _BENCH_RUNNING.add(session_id)
        try:
            steps = [
                [sys.executable,
                 str(ROOT / "scripts" / "benchmark_real_dataset.py"),
                 "--custom-session", session_id],
            ]
            ffile = ROOT / "data" / "benchmarks" / \
                f"fraud_detection_{session_id}.json"
            if not ffile.exists():
                steps.append(
                    [sys.executable,
                     str(ROOT / "scripts" / "benchmark_fraud_detection.py"),
                     "--custom-session", session_id, "--negatives", "500"]
                )
            for cmd in steps:
                subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                               timeout=1800)
        except Exception:  # noqa: BLE001 - background job; dashboard shows "—"
            pass
        finally:
            _BENCH_RUNNING.discard(session_id)

    threading.Thread(target=work, daemon=True,
                     name=f"auto-bench-{session_id}").start()


def ensure_pdf_demo_fraud_benchmark() -> None:
    """Generate ``fraud_detection_synthetic.json`` for the PDF demo graph once
    (42 fraud labels from ``data/samples/claims.json``), so the dashboard's
    PDF row shows real Fraud P/R/F1 instead of "—". No-op when the file
    already exists or the graph isn't the demo graph."""
    ffile = ROOT / "data" / "benchmarks" / "fraud_detection_synthetic.json"
    if ffile.exists() or benchmark_running("pdf_demo"):
        return

    def work() -> None:
        _BENCH_RUNNING.add("pdf_demo")
        try:
            subprocess.run(
                [sys.executable,
                 str(ROOT / "scripts" / "benchmark_fraud_detection.py"),
                 "--dataset", "synthetic", "--negatives", "0"],
                cwd=ROOT, capture_output=True, text=True, timeout=1800,
            )
        except Exception:  # noqa: BLE001
            pass
        finally:
            _BENCH_RUNNING.discard("pdf_demo")

    threading.Thread(target=work, daemon=True,
                     name="auto-bench-pdf-demo").start()


def _run_blocking(driver, session_id: str, force: bool, timeout: int,
                  line_cb=None) -> dict:
    """Execute the seed for ``session_id`` (serialized under the lock).

    ``line_cb(line)`` is invoked for every stdout line when streaming is
    wanted; otherwise the subprocess output is captured and returned whole.
    Raises ``ValueError`` for unknown sessions and ``RuntimeError`` on seed
    failure/timeout.
    """
    if not session_exists(session_id):
        raise ValueError(f"unknown session: {session_id!r} (expected one of "
                         f"{sorted(s['id'] for s in all_sessions())})")
    session = get_session_meta(session_id)

    if not force and current_session_id(driver) == session_id:
        _maybe_auto_benchmark(session_id)
        return {"status": "already_loaded", "session": session_id, "output": ""}

    with _switch_lock:
        # re-check under the lock: another caller may have landed this session
        # while we were waiting for the lock
        if not force and current_session_id(driver) == session_id:
            _maybe_auto_benchmark(session_id)
            return {"status": "already_loaded", "session": session_id, "output": ""}
        cmd = _seed_command(session)
        try:
            if line_cb is None:
                proc = subprocess.run(cmd, cwd=ROOT, capture_output=True,
                                      text=True, timeout=timeout)
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"Session '{session_id}' seeding failed "
                        f"(rc={proc.returncode}):\n{proc.stdout[-1500:]}"
                        f"\n{proc.stderr[-1500:]}"
                    )
                _maybe_auto_benchmark(session_id)
                return {"status": "seeded", "session": session_id,
                        "output": proc.stdout[-2000:]}

            # streaming: relay every stdout line to the caller as it arrives
            proc = subprocess.Popen(
                cmd, cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            lines: list[str] = []
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                lines.append(line)
                line_cb(line)
            proc.wait(timeout=timeout)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Session '{session_id}' seeding failed (rc={proc.returncode}):\n"
                    + "\n".join(lines[-30:])
                )
            _maybe_auto_benchmark(session_id)
            return {"status": "seeded", "session": session_id,
                    "output": "\n".join(lines[-2000:])}
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Session '{session_id}' seeding timed out after {timeout}s."
            ) from None


def switch_session(driver, session_id: str, force: bool = False,
                   timeout: int = 900, line_cb=None) -> dict:
    """Seed the graph for the requested session; **blocking** (REST API path).

    Returns ``{"status": "already_loaded"|"seeded", "session": id,
    "output": tail-of-log}``. Raises ``ValueError`` for unknown sessions and
    ``RuntimeError`` when the seeding command fails.

    ``line_cb(line)`` (v2, jobs) streams every subprocess stdout line as it
    arrives — the job runner uses it to record live progress.
    """
    return _run_blocking(driver, session_id, force, timeout, line_cb=line_cb)


# ---------------------------------------------------------------------------
# background streaming runner (web UI path — live progress in the sidebar)
# ---------------------------------------------------------------------------

def start_switch(driver, session_id: str, force: bool = False,
                 timeout: int = 900) -> dict:
    """Start a session switch in a background daemon thread.

    Returns a **progress handle**::

        {"session": id, "lines": [log lines...], "done": bool,
         "ok": bool, "error": str | None, "result": dict | None}

    ``lines`` grows in real time as the seed script prints; ``done`` flips to
    True when the thread finishes (``ok`` + ``result`` on success,
    ``error`` on failure). The handle is also registered in the module-level
    ``_PROGRESS`` store (keyed by session id) for ``active_switches`` /
    ``clear_progress``.
    """
    handle: dict = {
        "session": session_id,
        "lines": [],
        "done": False,
        "ok": False,
        "error": None,
        "result": None,
    }
    _PROGRESS[session_id] = handle

    def work() -> None:
        try:
            info = _run_blocking(driver, session_id, force, timeout,
                                 line_cb=handle["lines"].append)
            handle["result"] = info
            handle["ok"] = True
        except Exception as exc:  # noqa: BLE001 - surfaced via the handle
            handle["error"] = str(exc)
        finally:
            handle["done"] = True

    threading.Thread(target=work, daemon=True,
                     name=f"session-switch-{session_id}").start()
    return handle


def switch_progress(session_id: str) -> dict | None:
    """The progress handle for a session, if a switch was started for it."""
    return _PROGRESS.get(session_id)


def active_switches() -> list[dict]:
    """Handles of switches still running (oldest first)."""
    return [h for h in _PROGRESS.values() if not h["done"]]


def switch_in_progress() -> bool:
    """True while any session switch is running in the background."""
    return bool(active_switches())


def clear_progress(session_id: str) -> None:
    """Drop a finished switch's handle (safe no-op if absent)."""
    _PROGRESS.pop(session_id, None)
