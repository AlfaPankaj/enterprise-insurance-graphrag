#!/usr/bin/env python3
"""
GraphRAG Insurance Claims System — Phase 1: Synthetic Data Pipeline.

Generates deterministic, ground-truth-verifiable commercial insurance data:
  * data/samples/policies.json      (default 100 policies, each with policyholder + coverages)
  * data/samples/claims.json        (default 200 claims, optional fraud_flag + investigator)
  * data/samples/endorsements.json  (default 50 endorsements)
  * data/samples/ground_truth.json  (PDF file -> canonical entities + relationships)
  * data/pdfs/*.pdf                 (synthetic insurance PDFs via reportlab)

Because every PDF is generated FROM a JSON record, the "correct" entities are
known by construction — this is what makes Phase 2 CDC tests and Phase 3
accuracy benchmarks measurable.

Usage:
    python scripts/data_pipeline.py                     # defaults per the plan
    python scripts/data_pipeline.py --no-pdfs           # JSON samples only
    python scripts/data_pipeline.py --num-claims 5000   # scale test
    python scripts/data_pipeline.py --seed 7 --pdfs-dir data/pdfs

Dependencies (see requirements.txt): faker, reportlab
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Domain constants (must match docs/graph_schema.md)
# ---------------------------------------------------------------------------

POLICY_TYPES = [
    "COMMERCIAL_GENERAL_LIABILITY",
    "COMMERCIAL_PROPERTY",
    "WORKERS_COMPENSATION",
    "AUTO_FLEET",
    "PROFESSIONAL_LIABILITY",
]

COVERAGES_BY_TYPE: dict[str, list[tuple[str, str, float]]] = {
    "COMMERCIAL_GENERAL_LIABILITY": [("CGL-A", "LIABILITY", 1_000_000), ("CGL-B", "LIABILITY", 2_000_000)],
    "COMMERCIAL_PROPERTY": [("CPP-A", "PROPERTY", 5_000_000), ("CPP-B", "PROPERTY", 2_000_000)],
    "WORKERS_COMPENSATION": [("WC-A", "EMPLOYEE_INJURY", 1_000_000)],
    "AUTO_FLEET": [("AF-A", "AUTO", 1_000_000), ("AF-B", "AUTO", 500_000)],
    "PROFESSIONAL_LIABILITY": [("PL-A", "ERRORS_OMISSIONS", 3_000_000)],
}

CLAIM_CAUSES = [
    "Fire damage",
    "Water damage",
    "Theft / burglary",
    "Third-party bodily injury",
    "Property damage",
    "Employee injury",
    "Auto collision",
    "Product liability",
    "Professional negligence",
    "Vandalism",
]

FRAUD_REASONS = [
    "Multiple claims on same vehicle within 90 days",
    "Claimant unable to provide documentation",
    "Inconsistent loss narrative across documents",
    "Loss predates policy inception",
]

ENDORSEMENT_TYPES = [
    "ADDITIONAL_INSURED",
    "LIMIT_INCREASE",
    "DEDUCTIBLE_CHANGE",
    "EXCLUSION_ADDED",
    "NAMED_PARTY_ADDED",
    "LOCATION_ADDED",
]

INVESTIGATOR_ROLES = ["SENIOR_INVESTIGATOR", "INVESTIGATOR", "FRAUD_SPECIALIST", "FIELD_ADJUSTER"]


# ---------------------------------------------------------------------------
# Randomness helpers
# ---------------------------------------------------------------------------

def _rand_date(rng: random.Random, start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, max(delta, 0)))


def _type_abbr(policy_type: str) -> str:
    return "".join(w[0] for w in policy_type.split("_"))  # CGL, CPP, WC, AF, PL


# ---------------------------------------------------------------------------
# Record factories
# ---------------------------------------------------------------------------

def make_investigator(fake: Any, rng: random.Random, i: int) -> dict:
    return {
        "id": f"INV-{i:04d}",
        "name": fake.name(),
        "role": rng.choice(INVESTIGATOR_ROLES),
        "email": fake.email(),
    }


def make_policyholder(fake: Any, rng: random.Random, i: int) -> dict:
    return {
        "id": f"PH-{i:04d}",
        "name": fake.name(),
        "dob": fake.date_of_birth(minimum_age=25, maximum_age=70).isoformat(),
        "address": fake.address().replace("\n", ", "),
        "phone": fake.phone_number(),
        "email": fake.email(),
        "risk_score": round(rng.uniform(5.0, 95.0), 1),
    }


def make_policy(rng: random.Random, i: int) -> dict:
    ptype = rng.choice(POLICY_TYPES)
    start = _rand_date(rng, date(2023, 1, 1), date(2025, 6, 30))
    return {
        "id": f"POL-{i:04d}",
        "policy_number": f"{_type_abbr(ptype)}-{start.year}-{i:04d}",
        "type": ptype,
        "start_date": start.isoformat(),
        "end_date": (start + timedelta(days=365)).isoformat(),
        "premium": float(rng.randint(4_000, 60_000)),
        "deductible": float(rng.choice([1_000, 2_500, 5_000, 10_000])),
        "status": rng.choices(["ACTIVE", "EXPIRED", "CANCELLED", "LAPSED"], weights=[70, 20, 6, 4])[0],
    }


def make_coverages(rng: random.Random, policy: dict, start_index: int) -> list[dict]:
    coverages: list[dict] = []
    for j, (code, category, limit) in enumerate(COVERAGES_BY_TYPE[policy["type"]]):
        coverages.append(
            {
                "id": f"COV-{start_index + j:04d}",
                "code": code,
                "category": category,
                "limit": float(limit),
                "deductible": policy["deductible"],
                "exclusions": ["Intentional acts", "War and terrorism"] if j == 0 else ["Intentional acts"],
            }
        )
    return coverages


def make_claim(rng: random.Random, policy: dict, i: int) -> dict:
    start = date.fromisoformat(policy["start_date"])
    claim_date = _rand_date(rng, start, start + timedelta(days=540))
    return {
        "id": f"CLM-{i:04d}",
        "claim_number": f"CLM-{claim_date.year}-{i:04d}",
        "policy_id": policy["id"],
        "date": claim_date.isoformat(),
        "amount": float(rng.randint(1_000, 250_000)),
        "status": rng.choices(
            ["SUBMITTED", "IN_REVIEW", "APPROVED", "DENIED", "PAID"], weights=[15, 30, 25, 10, 20]
        )[0],
        "cause": rng.choice(CLAIM_CAUSES),
        "description": f"{rng.choice(CLAIM_CAUSES)} reported on {claim_date.isoformat()} under policy "
        f"{policy['policy_number']}. Estimated loss {rng.randint(1, 250)}k USD.",
    }


def detect_fraud(rng: random.Random, policy: dict, claim: dict, fraud_rate: float) -> dict | None:
    """Deterministic fraud rules first, random rate as fallback."""
    start = date.fromisoformat(policy["start_date"])
    claim_date = date.fromisoformat(claim["date"])

    if (claim_date - start).days <= 7:
        return _fraud_flag(claim, "Claim filed within 7 days of policy inception", "HIGH", 0.92, claim_date)
    if claim["amount"] > policy["premium"] * 10:
        return _fraud_flag(claim, "Claim amount exceeds 10x annual premium", "MEDIUM", 0.81, claim_date)
    if rng.random() < fraud_rate:
        return _fraud_flag(claim, rng.choice(FRAUD_REASONS), "LOW", 0.60, claim_date)
    return None


def _fraud_flag(claim: dict, reason: str, severity: str, confidence: float, when: date) -> dict:
    return {
        "id": f"FRD-{claim['id']}",  # derived from the claim id: stable + unique across seeds
        "reason": reason,
        "confidence": confidence,
        "severity": severity,
        "created_by": "SYSTEM",
        "created_at": when.isoformat(),
    }


def make_endorsement(rng: random.Random, policy: dict, i: int) -> dict:
    etype = rng.choice(ENDORSEMENT_TYPES)
    effective = _rand_date(rng, date.fromisoformat(policy["start_date"]), date.fromisoformat(policy["end_date"]))
    return {
        "id": f"END-{i:04d}",
        "endorsement_number": f"END-{effective.year}-{i:04d}",
        "policy_id": policy["id"],
        "type": etype,
        "effective_date": effective.isoformat(),
        "clause": (
            f"This endorsement amends policy {policy['policy_number']}. "
            f"{etype.replace('_', ' ').title()} effective {effective.isoformat()}. "
            f"All other terms and conditions remain unchanged."
        ),
        "premium_adjustment": float(rng.choice([-250, 0, 250, 500, 750, 1000])),
    }


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def generate_dataset(
    seed: int, num_policies: int, num_claims: int, num_endorsements: int, fraud_rate: float
) -> dict:
    rng = random.Random(seed)
    fake = None
    try:
        from faker import Faker

        fake = Faker()
        fake.seed_instance(seed)
    except ImportError:
        raise SystemExit(
            "faker is not installed. Run:  pip install -r requirements.txt"
        ) from None

    investigators = [make_investigator(fake, rng, i) for i in range(1, 7)]

    policies: list[dict] = []
    for i in range(1, num_policies + 1):
        policy = make_policy(rng, i)
        policy["policyholder"] = make_policyholder(fake, rng, i)
        policy["coverages"] = make_coverages(rng, policy, (i - 1) * 4 + 1)
        policy["endorsements"] = []
        policies.append(policy)

    claims: list[dict] = []
    for i in range(1, num_claims + 1):
        policy = rng.choice(policies)
        claim = make_claim(rng, policy, i)
        record = {"doc_id": claim["id"], "claim": claim}
        record["fraud_flag"] = detect_fraud(rng, policy, claim, fraud_rate)
        if claim["status"] in {"IN_REVIEW", "APPROVED", "PAID"} and rng.random() < 0.6:
            record["investigator"] = rng.choice(investigators)
        claims.append(record)

    endorsements: list[dict] = []
    for i in range(1, num_endorsements + 1):
        policy = rng.choice(policies)
        endorsement = make_endorsement(rng, policy, i)
        endorsements.append(endorsement)
        policy["endorsements"].append(endorsement)

    return {
        "policies": policies,
        "claims": claims,
        "endorsements": endorsements,
        "investigators": investigators,
    }


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_samples(dataset: dict, samples_dir: Path) -> None:
    samples_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "policies.json": dataset["policies"],
        "claims.json": dataset["claims"],
        "endorsements.json": dataset["endorsements"],
    }
    for name, records in files.items():
        (samples_dir / name).write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"  wrote {samples_dir / name}  ({len(records)} records)")


# ---------------------------------------------------------------------------
# Ground truth (PDF file -> canonical entities + relationships)
# ---------------------------------------------------------------------------

def _policy_ground_truth(policy: dict) -> dict:
    entities = [
        {"label": "Policyholder", "id": policy["policyholder"]["id"]},
        {"label": "Policy", "id": policy["id"]},
    ]
    relationships = [{"from": policy["policyholder"]["id"], "type": "HAS_POLICY", "to": policy["id"]}]
    for cov in policy["coverages"]:
        entities.append({"label": "Coverage", "id": cov["id"]})
        relationships.append({"from": policy["id"], "type": "COVERS", "to": cov["id"]})
    for end in policy["endorsements"]:
        entities.append({"label": "Endorsement", "id": end["id"]})
        relationships.append({"from": policy["id"], "type": "ENDORSED_BY", "to": end["id"]})
    return {"entities": entities, "relationships": relationships}


def _claim_ground_truth(record: dict) -> dict:
    claim = record["claim"]
    entities = [{"label": "Claim", "id": claim["id"]}]
    # The policy is only referenced (by id) inside claim documents, but the edge
    # is cross-document: Phase 2 CDC must be able to verify the policy->claim link.
    relationships: list[dict] = [{"from": claim["policy_id"], "type": "HAS_CLAIM", "to": claim["id"]}]
    fraud_flag = record.get("fraud_flag")
    if fraud_flag:
        entities.append({"label": "FraudFlag", "id": fraud_flag["id"]})
        relationships.append({"from": claim["id"], "type": "FRAUD_DETECTED", "to": fraud_flag["id"]})
    investigator = record.get("investigator")
    if investigator:
        entities.append({"label": "Investigator", "id": investigator["id"]})
        relationships.append({"from": investigator["id"], "type": "INVESTIGATES_CLAIM", "to": claim["id"]})
    return {"entities": entities, "relationships": relationships}


def _endorsement_ground_truth(endorsement: dict) -> dict:
    return {"entities": [{"label": "Endorsement", "id": endorsement["id"]}], "relationships": []}


# ---------------------------------------------------------------------------
# PDF generation (reportlab)
# ---------------------------------------------------------------------------

def _html(value: Any) -> str:
    from xml.sax.saxutils import escape

    return escape(str(value))


def _build_pdf(path: Path, title: str, blocks: list[Any]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    path.parent.mkdir(parents=True, exist_ok=True)

    header = Table(
        [[Paragraph(f"<b>{_html(title)}</b>", styles["Title"]), "EXL GraphRAG Demo"]],
        colWidths=[100 * mm, 55 * mm],  # keep within the ~159mm printable frame
    )
    header.setStyle(TableStyle([("ALIGN", (1, 0), (1, 0), "RIGHT")]))

    elements: list[Any] = [header, Spacer(1, 6 * mm)]
    for kind, data in blocks:
        if kind == "kv":
            rows = [[Paragraph(f"<b>{_html(k)}</b>", styles["Normal"]), _html(v)] for k, v in data]
            table = Table(rows, colWidths=[55 * mm, 125 * mm])
            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.lightgrey]),
                    ]
                )
            )
            elements.append(table)
        elif kind == "cols":
            rows = [[Paragraph(f"<b>{_html(h)}</b>", styles["Normal"]) for h in data[0]]]
            rows += [[_html(v) for v in row] for row in data[1:]]
            table = Table(rows, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            elements.append(table)
        elif kind == "para":
            elements.append(Paragraph(_html(data), styles["BodyText"]))
        elements.append(Spacer(1, 4 * mm))

    doc = SimpleDocTemplate(str(path), pagesize=A4, title=title, author="GraphRAG Demo")
    doc.build(elements)


def _pdf_policy(policy: dict, pdfs_dir: Path) -> str:
    ph, p = policy["policyholder"], policy
    blocks: list[tuple[str, Any]] = [
        ("kv", [("Policy ID", p["id"]), ("Policy Number", p["policy_number"]), ("Type", p["type"]),
                ("Term", f'{p["start_date"]} to {p["end_date"]}'), ("Status", p["status"]),
                ("Annual Premium", f"${p['premium']:,.0f}"), ("Deductible", f"${p['deductible']:,.0f}")]),
        ("para", "Policyholder"),
        ("kv", [("Policyholder ID", ph["id"]), ("Name", ph["name"]), ("Date of Birth", ph["dob"]),
                ("Address", ph["address"]), ("Phone", ph["phone"]), ("Email", ph["email"]),
                ("Risk Score", ph["risk_score"])]),
        ("para", "Coverages"),
        ("cols", [["ID", "Code", "Category", "Limit", "Deductible", "Exclusions"]] + [
            [c["id"], c["code"], c["category"], f"${c['limit']:,.0f}", f"${c['deductible']:,.0f}",
             "; ".join(c["exclusions"])] for c in p["coverages"]]),
    ]
    if p["endorsements"]:
        blocks.append(("para", "Endorsements"))
        blocks.append(("cols", [["ID", "Number", "Type", "Effective", "Premium Adj"]] + [
            [e["id"], e["endorsement_number"], e["type"], e["effective_date"],
             f"${e['premium_adjustment']:,.0f}"] for e in p["endorsements"]]))
    _build_pdf(pdfs_dir / f"policy_{p['id']}.pdf", f"Commercial Insurance Policy — {p['policy_number']}", blocks)
    return f"policy_{p['id']}.pdf"


def _pdf_claim(record: dict, pdfs_dir: Path) -> str:
    claim, policy = record["claim"], record["_policy"]
    blocks: list[tuple[str, Any]] = [
        ("kv", [("Claim ID", claim["id"]), ("Claim Number", claim["claim_number"]),
                ("Policy", policy["policy_number"]), ("Policy ID", claim["policy_id"]),
                ("Date of Loss", claim["date"]),
                ("Amount", f"${claim['amount']:,.0f}"), ("Status", claim["status"]),
                ("Cause", claim["cause"])]),
        ("para", claim["description"]),
    ]
    if record.get("fraud_flag"):
        f = record["fraud_flag"]
        blocks.append(("para", "Fraud Flag"))
        blocks.append(("kv", [("Flag ID", f["id"]), ("Reason", f["reason"]),
                              ("Confidence", f["confidence"]),
                              ("Severity", f["severity"]), ("Created By", f["created_by"])]))
    if record.get("investigator"):
        inv = record["investigator"]
        blocks.append(("para", "Investigator"))
        blocks.append(("kv", [("Investigator ID", inv["id"]), ("Name", inv["name"]),
                              ("Role", inv["role"]), ("Email", inv["email"])]))
    _build_pdf(pdfs_dir / f"claim_{claim['id']}.pdf", f"Claim Report — {claim['claim_number']}", blocks)
    return f"claim_{claim['id']}.pdf"


def _pdf_endorsement(endorsement: dict, pdfs_dir: Path) -> str:
    e = endorsement
    blocks: list[tuple[str, Any]] = [
        ("kv", [("Endorsement ID", e["id"]), ("Endorsement Number", e["endorsement_number"]),
                ("Policy", e["policy_id"]), ("Type", e["type"]), ("Effective Date", e["effective_date"]),
                ("Premium Adjustment", f"${e['premium_adjustment']:,.0f}")]),
        ("para", e["clause"]),
    ]
    _build_pdf(pdfs_dir / f"endorsement_{e['id']}.pdf", f"Policy Endorsement — {e['endorsement_number']}", blocks)
    return f"endorsement_{e['id']}.pdf"


def export_pdfs(dataset: dict, pdfs_dir: Path, n_policies: int, n_claims: int, n_endorsements: int) -> dict:
    """Render PDFs from the dataset and return {pdf_file: ground_truth}.

    NOTE: mutates claim records with a temporary `_policy` key — must be called
    AFTER export_samples() so the key never reaches the serialized JSON.
    """
    try:
        from reportlab.platypus import SimpleDocTemplate  # noqa: F401  (verify install early)
    except ImportError:
        raise SystemExit("reportlab is not installed. Run:  pip install -r requirements.txt") from None

    # Claim records need their policy for the PDF; attach it (not serialized to JSON).
    policy_by_id = {p["id"]: p for p in dataset["policies"]}
    for record in dataset["claims"]:
        record["_policy"] = policy_by_id[record["claim"]["policy_id"]]

    ground_truth: dict[str, dict] = {}
    for policy in dataset["policies"][:n_policies]:
        ground_truth[_pdf_policy(policy, pdfs_dir)] = _policy_ground_truth(policy)
    for record in dataset["claims"][:n_claims]:
        ground_truth[_pdf_claim(record, pdfs_dir)] = _claim_ground_truth(record)
    for endorsement in dataset["endorsements"][:n_endorsements]:
        ground_truth[_pdf_endorsement(endorsement, pdfs_dir)] = _endorsement_ground_truth(endorsement)
    return ground_truth


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic insurance samples (JSON + PDFs) with known ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples-dir", type=Path, default=PROJECT_ROOT / "data" / "samples",
                        help="Where to write the sample JSON files")
    parser.add_argument("--pdfs-dir", type=Path, default=PROJECT_ROOT / "data" / "pdfs",
                        help="Where to write the generated PDFs")
    parser.add_argument("--num-policies", type=int, default=100, help="Number of policies to generate")
    parser.add_argument("--num-claims", type=int, default=200, help="Number of claims to generate")
    parser.add_argument("--num-endorsements", type=int, default=50, help="Number of endorsements to generate")
    parser.add_argument("--fraud-rate", type=float, default=0.10,
                        help="Fraction of claims randomly flagged beyond the deterministic rules")
    parser.add_argument("--pdf-policies", type=int, default=50, help="How many policy PDFs to render")
    parser.add_argument("--pdf-claims", type=int, default=30, help="How many claim PDFs to render")
    parser.add_argument("--pdf-endorsements", type=int, default=20, help="How many endorsement PDFs to render")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (deterministic output)")
    parser.add_argument("--no-pdfs", action="store_true", help="Only write JSON samples, skip PDF generation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        f"Generating dataset: {args.num_policies} policies, {args.num_claims} claims, "
        f"{args.num_endorsements} endorsements (seed={args.seed}, fraud_rate={args.fraud_rate})"
    )

    dataset = generate_dataset(args.seed, args.num_policies, args.num_claims, args.num_endorsements, args.fraud_rate)
    export_samples(dataset, args.samples_dir)

    n_flagged = sum(1 for r in dataset["claims"] if r["fraud_flag"])
    print(f"  fraud flags: {n_flagged}/{len(dataset['claims'])} claims "
          f"({n_flagged / len(dataset['claims']) * 100:.1f}%)")

    if args.no_pdfs:
        print("  PDF generation skipped; ground_truth.json left untouched (if present)")
        return 0

    ground_truth = export_pdfs(dataset, args.pdfs_dir, args.pdf_policies, args.pdf_claims, args.pdf_endorsements)
    (args.samples_dir / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2), encoding="utf-8"
    )
    print(f"  wrote {args.samples_dir / 'ground_truth.json'}  ({len(ground_truth)} PDF mappings)")
    print(f"  wrote {len(ground_truth)} PDFs -> {args.pdfs_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
