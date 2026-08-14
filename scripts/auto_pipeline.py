"""Auto-pipeline: keep benchmark results fresh whenever real data changes.

Watches ``data/Real_datasets/*.csv`` (3 distinct real datasets —
``insurance_dataset.csv`` and ``data_synthetic.csv`` are the same Kaggle source
in two forms). When a file is
added, modified or re-saved, it automatically runs the full validation chain
for that dataset and refreshes the JSON results the dashboard reads:

    session switch (POST /api/v1/session — so each benchmark runs inside its
                     own session; falls back to a direct ingest when the API
                     is down; ``data_synthetic`` is always direct-ingested at
                     full scale because it has no dedicated API session)
      -> real benchmark (scripts/benchmark_real_dataset.py)
      -> fraud benchmark (scripts/benchmark_fraud_detection.py, labeled sets)
      -> data/benchmarks/*.json updated  (dashboard picks them up on reload)

Usage:
    # one-shot: process every dataset whose file changed since the last run
    .venv/Scripts/python.exe scripts/auto_pipeline.py --once

    # force re-run of everything regardless of fingerprints
    .venv/Scripts/python.exe scripts/auto_pipeline.py --once --force

    # watch the folder forever (debounced; Ctrl-C to stop)
    .venv/Scripts/python.exe scripts/auto_pipeline.py --watch

Fingerprints (size + mtime + sha256) are stored in
``data/benchmarks/.auto_pipeline_state.json`` so unchanged files are skipped.
Change detection is content-based: a file whose content is identical to the
last run (even if it was deleted and re-added) is not reprocessed. A dataset
is only marked done when its whole chain succeeded — partial failures are
retried on the next run. ``--data-dir`` points the watcher at a different
folder (useful for testing).

The fraud benchmark is the slow stage (all 923 fraud_oracle labels, ~20-70s);
use ``--fraud-limit`` to shrink it (e.g. ``--fraud-limit 20 --fraud-negatives 20``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # for graphrag.* imports

from graphrag.config import settings  # noqa: E402

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "Real_datasets"
DEFAULT_STATE = PROJECT_ROOT / "data" / "benchmarks" / ".auto_pipeline_state.json"
REPORT = PROJECT_ROOT / "data" / "real_dataset_results.md"

# datasets that carry fraud ground truth -> run the fraud benchmark too
LABELED = {"fraud_oracle", "insurance_claims"}

# CSV stem -> API session id. data_synthetic has NO dedicated API session (it
# is the synthetic variant of insurance_dataset), so it is always
# direct-ingested at full scale to keep its benchmark anchored.
API_SESSION: dict[str, str] = {
    "fraud_oracle": "fraud_oracle",
    "insurance_claims": "insurance_claims",
    "insurance_dataset": "insurance_dataset",
}

PY = Path(sys.executable)
INGEST = [str(PY), str(PROJECT_ROOT / "scripts" / "ingest_real_dataset.py")]
REAL_BENCH = [str(PY), str(PROJECT_ROOT / "scripts" / "benchmark_real_dataset.py")]
FRAUD_BENCH = [str(PY), str(PROJECT_ROOT / "scripts" / "benchmark_fraud_detection.py")]


def fingerprint(path: Path) -> dict:
    """(size, mtime_ns, sha256) — content-based so a re-save triggers a run."""
    data = path.read_bytes()
    return {
        "size": len(data),
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"files": {}, "last_run": None}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def changed_datasets(data_dir: Path, state: dict, force: bool) -> list[str]:
    """Datasets whose CSV is new or differs from the last fingerprint."""
    changed = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        name = csv_path.stem
        if force or name not in state.get("files", {}) \
                or fingerprint(csv_path) != state["files"][name]:
            changed.append(name)
    return changed


def run(cmd: list[str]) -> int:
    print(f"  $ {' '.join(cmd[-2:])}")
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def switch_session_via_api(session_id: str, api_url: str,
                           api_key: str | None, timeout: int = 900) -> str:
    """Switch the loaded graph via ``POST /api/v1/session``.

    Returns ``"ok"`` when the switch succeeded (or the session was already
    loaded) and ``"unreachable"`` when the API is down (caller should fall
    back to a direct ingest). Raises ``RuntimeError`` on an API error
    response (400/500) so the pipeline retries next run instead of
    benchmarking the wrong graph.
    """
    url = f"{api_url.rstrip('/')}/api/v1/session"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        resp = httpx.post(url, json={"session_id": session_id},
                          headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        print(f"  !! API {url} unreachable ({exc.__class__.__name__}) — "
              f"falling back to direct ingest")
        return "unreachable"
    if resp.status_code == 200:
        seeded = resp.json().get("seeded")
        print(f"  API session '{session_id}' "
              f"-> {'seeded' if seeded else 'already loaded'}")
        return "ok"
    raise RuntimeError(
        f"API session switch failed ({resp.status_code}): {resp.text[:300]}"
    )


def _load_session(name: str, args: argparse.Namespace) -> bool:
    """Load a dataset into the graph for benchmarking.

    Prefers the API session endpoint (``POST /api/v1/session``) so each
    benchmark runs inside its own session; falls back to a direct
    ``ingest_real_dataset.py <name> --reset`` when the API is down or the
    dataset has no dedicated session (``data_synthetic``). Returns False on
    failure so the caller skips benchmarking and retries next run.
    """
    session_id = None if args.no_api else API_SESSION.get(name)
    if session_id:
        print(f"\n--- [{name}] session switch via API ({session_id}) ---")
        try:
            outcome = switch_session_via_api(session_id, args.api_url,
                                             args.api_key)
        except RuntimeError as exc:
            print(f"  !! {exc} — skipping benchmarks (will retry next run)")
            return False
        if outcome == "ok":
            return True
        print("  API unreachable — falling back to direct ingest")
    print(f"\n--- [{name}] ingest (direct) ---")
    rc = run(INGEST + [name, "--reset"])
    if rc != 0:
        print(f"  !! ingest failed for {name} (rc={rc}) — skipping benchmarks "
              f"(will retry next run)")
        return False
    return True


def process_datasets(datasets: list[str], args: argparse.Namespace,
                     state: dict, state_path: Path) -> int:
    """Run ingest -> real benchmark -> fraud benchmark for each dataset.

    The dataset's fingerprint is recorded only when the ENTIRE chain
    succeeded, so a partial failure (e.g. a crashed benchmark) is retried on
    the next run instead of being skipped as "unchanged".
    """
    if not datasets:
        print("no changes detected — nothing to run")
        return 0
    print(f"=== auto-pipeline: {len(datasets)} dataset(s) changed: "
          f"{', '.join(datasets)} ===")
    files = state.setdefault("files", {})
    t0 = time.time()
    exit_code = 0
    for name in datasets:
        if not _load_session(name, args):
            exit_code = exit_code or 1
            continue
        print(f"--- [{name}] real benchmark ---")
        rc = run(REAL_BENCH + [name])
        chain_ok = rc == 0
        if name in LABELED:
            print(f"--- [{name}] fraud benchmark ---")
            fraud_cmd = FRAUD_BENCH + ["--dataset", name]
            if args.fraud_limit:
                fraud_cmd += ["--limit", str(args.fraud_limit)]
            if args.fraud_negatives is not None:
                fraud_cmd += ["--negatives", str(args.fraud_negatives)]
            rc = run(fraud_cmd)
            chain_ok = chain_ok and rc == 0
        if not chain_ok:
            exit_code = exit_code or 1
            print(f"  !! benchmark failed for {name} — fingerprint NOT recorded, "
                  f"will retry next run")
            continue
        # record success only after the full chain for this dataset
        csv_path = Path(args.data_dir) / f"{name}.csv"
        files[name] = fingerprint(csv_path)

    state["last_run"] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": datasets,
        "elapsed_s": round(time.time() - t0, 1),
    }
    save_state(state_path, state)
    stamp_report(datasets, state["last_run"])
    print(f"\n=== done in {state['last_run']['elapsed_s']}s — JSONs refreshed, "
          f"dashboard picks them up on reload ===")
    return exit_code


def stamp_report(datasets: list[str], last_run: dict) -> None:
    """Append a one-line trace of the latest auto-run to the report."""
    line = (f"\n*Auto-pipeline run {last_run['timestamp']} ({last_run['elapsed_s']}s): "
            f"{', '.join(datasets)} — results refreshed in `data/benchmarks/`.*\n")
    if not REPORT.exists():
        REPORT.write_text(line, encoding="utf-8")
        return
    text = REPORT.read_text(encoding="utf-8")
    # keep only the most recent stamp to avoid unbounded growth; rstrip so
    # repeated runs can't accumulate stray blank lines in the report
    if "*Auto-pipeline run " in text:
        head, _, tail = text.partition("\n*Auto-pipeline run ")
        tail = tail.split("\n", 1)[1] if "\n" in tail else ""
        REPORT.write_text(head.rstrip("\n") + "\n\n" + line + "\n" + tail.lstrip("\n"),
                          encoding="utf-8")
    else:
        REPORT.write_text(text.rstrip("\n") + "\n\n" + line + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# watch mode (watchdog with debounce)
# ---------------------------------------------------------------------------

def watch(data_dir: Path, state_path: Path, args: argparse.Namespace) -> int:
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        print("watchdog not installed — falling back to polling "
              "(--poll 10, every 10s)")
        return poll_loop(data_dir, state_path, args)

    state = load_state(state_path)
    event = threading.Event()  # set from the observer thread, consumed in main

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event_):  # noqa: N802 (watchdog API)
            name = Path(getattr(event_, "src_path", "")).name
            if name.endswith(".csv"):
                event.set()

    observer = Observer()
    observer.schedule(Handler(), str(data_dir), recursive=False)
    observer.start()
    print(f"watching {data_dir} for *.csv changes (Ctrl-C to stop)…")
    try:
        while True:
            time.sleep(args.poll)
            if event.is_set():
                event.clear()
                # debounce: wait for the file to stop changing (writes happen
                # in chunks) before starting the chain
                datasets = changed_datasets(data_dir, state, force=False)
                if not datasets:
                    continue
                for _ in range(args.settle // args.poll):
                    time.sleep(args.poll)
                    current = changed_datasets(data_dir, state, force=False)
                    if current == datasets:
                        break
                    datasets = current
                process_datasets(datasets, args, state, state_path)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        observer.stop()
        observer.join()
    return 0


def poll_loop(data_dir: Path, state_path: Path, args: argparse.Namespace) -> int:
    state = load_state(state_path)
    print(f"polling {data_dir} every {args.poll}s (Ctrl-C to stop)…")
    try:
        while True:
            datasets = changed_datasets(data_dir, state, force=False)
            if datasets:
                time.sleep(args.settle)  # let the file finish writing
                datasets = changed_datasets(data_dir, state, force=False)
                if datasets:
                    process_datasets(datasets, args, state, state_path)
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--once", action="store_true",
                   help="run once on every dataset whose CSV changed, then exit")
    p.add_argument("--watch", action="store_true",
                   help="watch the folder and auto-run on changes (default)")
    p.add_argument("--force", action="store_true",
                   help="ignore fingerprints and re-run everything")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                   help=f"folder to watch (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--state", default=str(DEFAULT_STATE))
    p.add_argument("--poll", type=int, default=5,
                   help="watch: re-check interval in seconds (default 5)")
    p.add_argument("--settle", type=int, default=10,
                   help="watch: seconds to wait after a change so the file "
                        "finishes writing (default 10)")
    p.add_argument("--fraud-limit", type=int, default=None,
                   help="fraud benchmark: max fraud claims (shrink for fast runs)")
    p.add_argument("--fraud-negatives", type=int, default=None,
                   help="fraud benchmark: sample of clean claims")
    p.add_argument("--api-url", default="http://localhost:8000",
                   help="GraphRAG API base URL for session switching "
                        "(default: http://localhost:8000)")
    p.add_argument("--api-key", default=settings.API_KEY or None,
                   help="X-API-Key header for the API (default: settings.API_KEY)")
    p.add_argument("--no-api", action="store_true",
                   help="never call the API session endpoint — always ingest "
                        "directly")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state_path = Path(args.state)
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        return 2
    if args.once:
        state = load_state(state_path)
        datasets = changed_datasets(data_dir, state, force=args.force)
        return process_datasets(datasets, args, state, state_path)
    return watch(data_dir, state_path, args)


if __name__ == "__main__":
    sys.exit(main())
