"""Fraud ground-truth comparison for the dashboard (real-dataset validation).

Every real dataset we can ingest carries a fraud label column (except the two
that don't — see below). This module loads those labels, detects which dataset
is currently loaded in Neo4j, and compares an answer's fraud verdict against
the ground truth so the dashboard can show a per-claim ``✓ / ✗`` column.

Datasets:

  * ``fraud_oracle``      — ``FraudFound_P`` == "1"  (15,420 claims)
  * ``insurance_claims``  — ``fraud_reported`` == "Y" (1,000 claims)
  * ``synthetic`` / None  — ``data/samples/claims.json`` ``fraud_flag`` (demo)
  * ``insurance_dataset``, ``data_synthetic`` — **no fraud labels** -> None

Claim ids use the exact id scheme the ingest adapters create
(``CLM-{i:05d}`` for the CSVs, ``CLM-0003`` for the samples), so a claim id
found in the query/pruned context keys straight into the ground-truth table.

The verdict parser is deliberately conservative: a refusal such as "Not
determinable … no information" maps to ``UNKNOWN`` (shown as "—", never
counted as a wrong answer), and explicit negations ("not flagged as fraud")
are checked before any affirmative mention of the word "fraud".
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "Real_datasets"
SAMPLES_CLAIMS = PROJECT_ROOT / "data" / "samples" / "claims.json"

_CLAIM_RE = re.compile(r"CLM-\d{3,}")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Phrase checks are ordered: UNKNOWN (refusals/absence) -> NO (explicit
# negation) -> YES (any remaining "fraud" mention). "not determinable … no
# information that would indicate whether … fraudulent or not" must not be
# read as YES because it contains the word "fraudulent".
_UNKNOWN_PHRASES = (
    "not determinable", "cannot determine", "unable to determine",
    "no information", "does not contain", "does not provide",
    "does not mention", "would indicate whether", "cannot confirm",
    "no evidence", "no indication", "not mentioned",
)
_NO_PHRASES = (
    "not flagged", "not fraudulent", "is not fraud", "isn't fraud",
    "no fraud", "not considered fraud", "not marked as fraud",
    "no fraud flag", "not fraud", "not related to fraud",
    "unrelated to fraud", "no relation to fraud",
)

VERDICT_LABEL = {"YES": "Fraud", "NO": "Not fraud", "UNKNOWN": "—"}
CHECK_LABEL = {"correct": "✅ Correct", "wrong": "❌ Wrong", "no-verdict": "— No verdict"}


# ---------------------------------------------------------------------------
# dataset detection (which labels apply to the loaded graph)
# ---------------------------------------------------------------------------

def detect_dataset(driver) -> str | None:
    """Name of the dataset currently loaded in Neo4j (via a ``(:Dataset)`` marker).

    ``ingest_real_dataset.py`` stamps ``(:Dataset {name})`` on ingest and
    ``seed_graph.py`` stamps ``synthetic`` for the demo graph; returns None for
    a graph that predates the marker (treated as synthetic by callers).
    """
    with driver.session() as session:
        row = session.run(
            "MATCH (d:Dataset) RETURN d.name AS name ORDER BY d.name LIMIT 1"
        ).single()
        return row["name"] if row else None


# ---------------------------------------------------------------------------
# ground-truth loaders: claim id -> is-fraud
# ---------------------------------------------------------------------------

def _load_real_csv(dataset: str, label_col: str, yes_values: set[str]) -> dict[str, bool]:
    path = DATA_DIR / f"{dataset}.csv"
    if not path.exists():
        return {}
    out: dict[str, bool] = {}
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            out[f"CLM-{i:05d}"] = str(row.get(label_col, "")).strip().upper() in yes_values
    return out


def load_ground_truth(dataset: str | None) -> dict[str, bool] | None:
    """claim id -> is-fraud for the given dataset; None when it has no labels.

    The label predicates mirror the ingest adapters exactly (FraudFound_P == "1",
    fraud_reported == "Y"), so ground truth can never disagree with which
    claims the graph actually carries a FraudFlag for.
    """
    if dataset == "fraud_oracle":
        return _load_real_csv("fraud_oracle", "FraudFound_P", {"1"})
    if dataset == "insurance_claims":
        return _load_real_csv("insurance_claims", "fraud_reported", {"Y"})
    if dataset in (None, "synthetic"):
        if not SAMPLES_CLAIMS.exists():
            return {}
        claims = json.loads(SAMPLES_CLAIMS.read_text(encoding="utf-8"))
        return {rec["claim"]["id"]: bool(rec.get("fraud_flag")) for rec in claims}
    return None  # insurance_dataset / data_synthetic carry no fraud labels


# ---------------------------------------------------------------------------
# verdict parsing (from the free-text answer)
# ---------------------------------------------------------------------------

def parse_fraud_verdict(text: str) -> str:
    """Overall fraud verdict of an answer snippet: YES | NO | UNKNOWN."""
    a = text.lower()
    if any(p in a for p in _UNKNOWN_PHRASES):
        return "UNKNOWN"
    if any(p in a for p in _NO_PHRASES):
        return "NO"
    if "fraud" in a:
        return "YES"
    return "UNKNOWN"


def verdict_for_claim(answer: str, claim_id: str) -> str:
    """Verdict the answer gives for one specific claim.

    Parses only the sentence(s) that mention the claim, so a list answer
    ("The fraud claims are CLM-00042, CLM-00117 … Claim CLM-00099 is not
    flagged.") evaluates each claim on its own and a claim the answer never
    mentions is UNKNOWN (no verdict).
    """
    sentences = [s for s in _SENT_SPLIT.split(answer) if claim_id.lower() in s.lower()]
    if not sentences:
        return "UNKNOWN"
    return parse_fraud_verdict(" ".join(sentences))


def evaluate_verdict(ground: bool, verdict: str) -> str:
    """Compare a verdict to ground truth: correct | wrong | no-verdict."""
    if verdict == "UNKNOWN":
        return "no-verdict"
    return "correct" if (verdict == "YES") == ground else "wrong"


# ---------------------------------------------------------------------------
# comparison rows for the dashboard
# ---------------------------------------------------------------------------

def claim_ids_in(text: str) -> list[str]:
    """Unique CLM-* ids mentioned in a string, in order of appearance."""
    return list(dict.fromkeys(_CLAIM_RE.findall(text)))


def build_comparison(query: str, pruned: dict, answer: str,
                     ground_truth: dict[str, bool]) -> list[dict]:
    """Rows for the ground-truth panel: [claim, verdict, ground, check].

    Targets = claim ids named in the query first, then claims in the pruned
    context (kept + dropped). Only ids present in ``ground_truth`` are kept —
    a claim with no label is not a comparison, it is noise.
    """
    if not ground_truth:
        return []  # no labels for this dataset — never crash the dashboard

    targets: list[str] = []
    for cid in claim_ids_in(query) + list(pruned.get("kept", [])) + list(pruned.get("dropped", [])):
        if cid not in targets:
            targets.append(cid)

    rows: list[dict] = []
    for cid in targets:
        if cid not in ground_truth:
            continue
        ground = ground_truth[cid]
        verdict = verdict_for_claim(answer, cid)
        check = evaluate_verdict(ground, verdict)
        rows.append({
            "claim": cid,
            "llm_verdict": VERDICT_LABEL[verdict],
            "ground_truth": "Fraud" if ground else "Not fraud",
            "check": CHECK_LABEL[check],
        })
    return rows
