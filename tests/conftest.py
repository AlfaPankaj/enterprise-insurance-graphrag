"""CI visibility hook — post test failures to the PR from inside Actions.

The sandbox that drives this branch cannot reach the Actions log storage
host, so CI failures would be invisible. Runners have full network access:
this hook reports the terminal summary (any failures) to the PR thread.

Active ONLY when: GITHUB_ACTIONS + arena/* head branch (never local runs,
never post-merge runs on main).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

REPO = "AlfaPankaj/enterprise-insurance-graphrag"


def _ci_target() -> tuple[str, str] | None:
    """(pr_number, token) when running in CI on an arena/* PR; else None."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return None
    head = os.environ.get("GITHUB_HEAD_REF", "")
    if not head.startswith("arena/"):
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
        print(f"[ci-report] could not post comment: {exc}")


def pytest_terminal_summary(terminalreporter) -> None:
    target = _ci_target()
    if target is None:
        return
    pr, token = target
    failed = terminalreporter.stats.get("failed", [])
    if not failed:
        return
    lines = [f"**CI test failures** (run {os.environ.get('GITHUB_RUN_ID', '?')}):", ""]
    for rep in failed[:6]:
        text = str(rep.longreprtext or rep.nodeid)
        lines.append(f"<details><summary><code>{rep.nodeid}</code></summary>")
        lines.append("\n```\n" + text[-2500:] + "\n```\n</details>\n")
    _post_comment(pr, token, "\n".join(lines))
