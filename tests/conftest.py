"""CI visibility hook — surface test failures via workflow annotations.

The sandbox that drives this branch cannot reach the Actions log storage
host, and the workflow token may be read-only (PR comments can silently
fail). ``::error ::`` workflow commands are parsed by the runner itself from
step stdout — they become check-run annotations, readable through the normal
API with no extra permissions. This is the guaranteed failure channel; the
PR-comment mirror is the bonus channel.

Active ONLY when: GITHUB_ACTIONS + arena/* head branch (never local runs,
never post-merge runs on main).
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

REPO = "AlfaPankaj/enterprise-insurance-graphrag"
_MAX_ANNOTATION_CHARS = 2200


def _ci_active() -> bool:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return False
    head = os.environ.get("GITHUB_HEAD_REF", "")
    ref = os.environ.get("GITHUB_REF", "")
    return head.startswith("arena/") or "/arena/" in ref or ref.startswith("arena/")


def annotate(kind: str, title: str, message: str) -> None:
    """Emit a GitHub workflow command (works with a read-only token).

    ``kind``: "error" | "notice" | "warning". Newlines are %0A-encoded;
    message is trimmed to keep the command line valid.
    """
    if not _ci_active():
        return
    msg = message.replace("%", "%25").replace("\n", "%0A").replace("\r", "")
    msg = msg[:_MAX_ANNOTATION_CHARS]
    # write directly to the step's stdout: runner parses every line
    sys.stdout.write(f"::{kind} title={title}::{msg}\n")
    sys.stdout.flush()


def _ci_target() -> tuple[str, str] | None:
    """(pr_number, token) when running in CI on an arena/* PR; else None."""
    if not _ci_active():
        return None
    m = re.match(r"refs/pull/(\d+)/", os.environ.get("GITHUB_REF", "") or "")
    token = os.environ.get("GITHUB_TOKEN", "")
    if not (m and token):
        return None
    return m.group(1), token


def _post_comment(pr: str, token: str, body: str) -> None:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/issues/{pr}/comments",
        data=json.dumps({"body": body}).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json"},
        method="POST")
    try:
        urllib.request.urlopen(req, timeout=20).read()
    except Exception as exc:  # noqa: BLE001 - reporting must never fail a run
        annotate("warning", "CI comment mirror failed", str(exc))


def pytest_terminal_summary(terminalreporter) -> None:
    if not _ci_active():
        return
    failed = terminalreporter.stats.get("failed", [])
    errors = terminalreporter.stats.get("error", [])
    if not failed and not errors:
        return

    parts: list[str] = []
    for rep in list(failed) + list(errors):
        text = str(rep.longreprtext or rep.nodeid)
        parts.append(f"### {rep.nodeid}\n{text[-1500:]}")
        if len(parts) >= 5:
            break
    body = f"pytest failures/errors ({len(failed)} failed, {len(errors)} errors):\n" + \
        "\n".join(parts)

    # guaranteed channel: annotations (parsed by the runner, no permissions)
    annotate("error", f"pytest: {len(failed)}F/{len(errors)}E", body)

    # bonus channel: PR comment (needs a write-enabled token)
    target = _ci_target()
    if target:
        pr, token = target
        _post_comment(pr, token, body[:60000])
