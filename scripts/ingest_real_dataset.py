"""Ingest real insurance datasets (data/Real_datasets/*.csv) into the graph.

Each dataset is adapted to the project's schema:

    (Policyholder)-[:HAS_POLICY]->(Policy)-[:HAS_CLAIM]->(Claim)
    (Claim)-[:FRAUD_DETECTED]->(FraudFlag)   (only where fraud is flagged)
    (Policy)-[:COVERS]->(Coverage)           (where coverage data exists)

Dataset adapters map real columns onto the schema's property names so the
existing retrieval/reranking/pruning pipeline works unchanged:

  * fraud_oracle.csv        — 15,420 auto claims with FraudFound_P ground truth
  * insurance_claims.csv    — 1,000 claims with fraud_reported ground truth
  * insurance_dataset.csv   — 13,000 customers + claim amounts (no fraud label)
  * data_synthetic.csv      — 53,503 customers with policies + coverages

Entity ids are prefixed to match the pipeline's id regex (POL|CLM|PH|FRD-\\d{3,})
so natural-language queries anchor on them exactly like the synthetic demo.

Usage:
    .venv/Scripts/python.exe scripts/ingest_real_dataset.py fraud_oracle --reset
    .venv/Scripts/python.exe scripts/ingest_real_dataset.py data_synthetic --reset --limit 10000
    .venv/Scripts/python.exe scripts/ingest_real_dataset.py --list

``--limit`` only applies to data_synthetic and defaults to *all* 53,503 rows.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Real_datasets"
sys.path.insert(0, str(PROJECT_ROOT))          # for `scripts.*` imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # for `graphrag.*` imports

from graphrag.config import settings  # noqa: E402
from scripts.seed_graph import load_nodes, load_relationships  # noqa: E402

DATASETS = ["fraud_oracle", "insurance_claims", "insurance_dataset", "data_synthetic"]


def policy_id(value, i: int, pad: int = 4) -> str:
    """Shared policy-id rule so ingest and benchmark can never drift.

    Matches the graph's id scheme (POL-<NNNN>): digit strings keep their
    digits (zero-padded to ``pad`` width, matching the ingest adapter),
    anything else falls back to the zero-padded row index.
    """
    s = _ident(value).replace(" ", "") or str(i)
    return f"POL-{(s if s.isdigit() else i):>0{pad}}"


def _num(value) -> float | None:
    """Parse a number from a CSV cell (money, counts); None when not numeric."""
    if value is None:
        return None
    s = str(value).strip().replace("$", "").replace(",", "").replace("%", "")
    if not s or s.lower() in ("nan", "none", "?", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _ident(s: str) -> str:
    """Keep only [A-Za-z0-9 _-], collapse spaces (Neo4j-safe, ASCII ids)."""
    return "".join(ch if ch.isalnum() or ch in " _-" else " " for ch in str(s)).strip()


# ---------------------------------------------------------------------------
# adapters: return (nodes_by_label, rels) — same shape as seed_graph.seed
# ---------------------------------------------------------------------------

def _adapt_fraud_oracle(path: Path, limit: int | None = None) -> tuple[dict[str, list[dict]], list]:
    nodes: dict[str, list[dict]] = defaultdict(list)
    rels: list[tuple] = []
    pol_ids: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            pol_id = policy_id(row.get("PolicyNumber", i), i, pad=4)
            if pol_id not in pol_ids:
                pol_ids.add(pol_id)
                nodes["Policyholder"].append({
                    "id": f"PH-{pol_id[4:]}",
                    "props": {
                        "name": f"{row.get('Sex', '')} driver",
                        "age": _num(row.get("Age")),
                        "marital_status": row.get("MaritalStatus"),
                        "risk_score": _num(row.get("DriverRating")),
                    },
                })
                nodes["Policy"].append({
                    "id": pol_id,
                    "props": {
                        "policy_number": pol_id,
                        "type": row.get("PolicyType"),
                        "deductible": _num(row.get("Deductible")),
                        "status": row.get("BasePolicy"),
                    },
                })
                rels.append(("Policyholder", "Policy", f"PH-{pol_id[4:]}", pol_id, "HAS_POLICY"))

            clm_id = f"CLM-{i:05d}"
            nodes["Claim"].append({
                "id": clm_id,
                "props": {
                    "claim_number": clm_id,
                    "policy_id": pol_id,
                    "cause": row.get("Fault"),
                    "status": row.get("AccidentArea"),
                    "amount": _num(row.get("VehiclePrice")),
                    "date": f"{row.get('MonthClaimed')} {row.get('DayOfWeekClaimed')}",
                    "make": row.get("Make"),
                    "vehicle_category": row.get("VehicleCategory"),
                    "police_report": row.get("PoliceReportFiled"),
                },
            })
            rels.append(("Policy", "Claim", pol_id, clm_id, "HAS_CLAIM"))

            if str(row.get("FraudFound_P", "0")).strip() == "1":
                nodes["FraudFlag"].append({
                    "id": f"FRD-{i:05d}",
                    "props": {
                        "claim_id": clm_id,
                        "reason": f"{row.get('Fault', '')} / {row.get('AccidentArea', '')}",
                        "severity": "CONFIRMED",
                        "confidence": 1.0,
                    },
                })
                rels.append(("Claim", "FraudFlag", clm_id, f"FRD-{i:05d}", "FRAUD_DETECTED"))
    return nodes, rels


def _adapt_insurance_claims(path: Path, limit: int | None = None) -> tuple[dict[str, list[dict]], list]:
    nodes: dict[str, list[dict]] = defaultdict(list)
    rels: list[tuple] = []
    ph_ids: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            pol_id = policy_id(row.get("policy_number", i), i, pad=4)
            ph_id = f"PH-{pol_id[4:]}"
            if ph_id not in ph_ids:
                ph_ids.add(ph_id)
                nodes["Policyholder"].append({
                    "id": ph_id,
                    "props": {
                        "name": f"{row.get('insured_sex', '')} customer",
                        "occupation": row.get("insured_occupation"),
                        "education_level": row.get("insured_education_level"),
                        "age": _num(row.get("age")),
                        "months_as_customer": _num(row.get("months_as_customer")),
                    },
                })
                nodes["Policy"].append({
                    "id": pol_id,
                    "props": {
                        "policy_number": pol_id,
                        "policyholder_id": ph_id,
                        "type": row.get("policy_csl"),
                        "status": row.get("policy_state"),
                        "premium": _num(row.get("policy_annual_premium")),
                        "deductible": _num(row.get("policy_deductable")),
                        "start_date": row.get("policy_bind_date"),
                    },
                })
                rels.append(("Policyholder", "Policy", ph_id, pol_id, "HAS_POLICY"))

            clm_id = f"CLM-{i:05d}"
            nodes["Claim"].append({
                "id": clm_id,
                "props": {
                    "claim_number": clm_id,
                    "policy_id": pol_id,
                    "amount": _num(row.get("total_claim_amount")),
                    "status": row.get("incident_severity"),
                    "cause": row.get("incident_type"),
                    "date": row.get("incident_date"),
                    "collision_type": row.get("collision_type"),
                    "auto_make": row.get("auto_make"),
                    "city": row.get("incident_city"),
                },
            })
            rels.append(("Policy", "Claim", pol_id, clm_id, "HAS_CLAIM"))

            if str(row.get("fraud_reported", "N")).strip().upper() == "Y":
                nodes["FraudFlag"].append({
                    "id": f"FRD-{i:05d}",
                    "props": {
                        "claim_id": clm_id,
                        "reason": row.get("incident_type"),
                        "severity": row.get("incident_severity"),
                        "confidence": 1.0,
                    },
                })
                rels.append(("Claim", "FraudFlag", clm_id, f"FRD-{i:05d}", "FRAUD_DETECTED"))
    return nodes, rels


def _adapt_insurance_dataset(path: Path, limit: int | None = None) -> tuple[dict[str, list[dict]], list]:
    nodes: dict[str, list[dict]] = defaultdict(list)
    rels: list[tuple] = []
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            ph_id = f"PH-{i:05d}"
            pol_id = f"POL-{i:05d}"
            clm_id = f"CLM-{i:05d}"
            nodes["Policyholder"].append({
                "id": ph_id,
                "props": {
                    "name": f"{row.get('Gender', '')} customer",
                    "occupation": row.get("Occupation"),
                    "education_level": row.get("Education"),
                    "age": _num(row.get("Age")),
                    "income": _num(row.get("Income")),
                    "marital_status": row.get("Marital_Status"),
                },
            })
            nodes["Policy"].append({
                "id": pol_id,
                "props": {"policy_number": pol_id, "policyholder_id": ph_id,
                          "status": "ACTIVE", "premium": _num(row.get("Income"))},
            })
            rels.append(("Policyholder", "Policy", ph_id, pol_id, "HAS_POLICY"))
            nodes["Claim"].append({
                "id": clm_id,
                "props": {"claim_number": clm_id, "policy_id": pol_id,
                          "amount": _num(row.get("Claim_Amount"))},
            })
            rels.append(("Policy", "Claim", pol_id, clm_id, "HAS_CLAIM"))
    return nodes, rels


def _adapt_data_synthetic(path: Path, limit: int | None = None) -> tuple[dict[str, list[dict]], list]:
    nodes: dict[str, list[dict]] = defaultdict(list)
    rels: list[tuple] = []
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            if limit and i > limit:
                break
            cust = _ident(row.get("Customer ID", i)).replace(" ", "")
            ph_id = f"PH-{(cust if cust.isdigit() else i):>05}"
            pol_id = policy_id(row.get("Customer ID", i), i, pad=5)
            nodes["Policyholder"].append({
                "id": ph_id,
                "props": {
                    "name": f"Customer {cust}",
                    "occupation": row.get("Occupation"),
                    "age": _num(row.get("Age")),
                    "risk_score": _num(row.get("Risk Profile")),
                    "credit_score": _num(row.get("Credit Score")),
                    "driving_record": row.get("Driving Record"),
                },
            })
            nodes["Policy"].append({
                "id": pol_id,
                "props": {
                    "policy_number": pol_id,
                    "policyholder_id": ph_id,
                    "type": row.get("Policy Type"),
                    "premium": _num(row.get("Premium Amount")),
                    "deductible": _num(row.get("Deductible")),
                    "status": row.get("Segmentation Group"),
                },
            })
            rels.append(("Policyholder", "Policy", ph_id, pol_id, "HAS_POLICY"))
            cov_id = f"COV-{cust if cust.isdigit() else i:>05}"
            nodes["Coverage"].append({
                "id": cov_id,
                "props": {
                    "code": cov_id,
                    "policy_id": pol_id,
                    "category": row.get("Policy Type"),
                    "limit": _num(row.get("Coverage Amount")),
                    "deductible": _num(row.get("Deductible")),
                },
            })
            rels.append(("Policy", "Coverage", pol_id, cov_id, "COVERS"))
    return nodes, rels


_ADAPTERS = {
    "fraud_oracle": _adapt_fraud_oracle,
    "insurance_claims": _adapt_insurance_claims,
    "insurance_dataset": _adapt_insurance_dataset,
    "data_synthetic": _adapt_data_synthetic,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest a real insurance CSV into the graph")
    p.add_argument("dataset", nargs="?", choices=DATASETS, help="which dataset to ingest")
    p.add_argument("--list", action="store_true", help="list available datasets")
    p.add_argument("--reset", action="store_true", help="clear the whole graph first")
    p.add_argument("--limit", type=int, default=None, help="max rows for data_synthetic "
                                                             "(default: all 53,503)")
    p.add_argument("--uri", default=settings.NEO4J_URI)
    p.add_argument("--user", default=settings.NEO4J_USER)
    p.add_argument("--password", default=settings.NEO4J_PASSWORD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list or not args.dataset:
        print("available datasets:")
        for name in DATASETS:
            csv_file = DATA_DIR / f"{name}.csv"
            print(f"  {name:<18} {'OK' if csv_file.exists() else 'MISSING'}  ({csv_file.name})")
        return 0

    if args.limit is not None and args.limit < 1:
        print("ERROR: --limit must be a positive integer (or omit it for all rows)")
        return 2

    adapter = _ADAPTERS[args.dataset]
    csv_path = DATA_DIR / f"{args.dataset}.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return 1

    print(f"ingesting {args.dataset} from {csv_path.name} ...")
    t0 = time.perf_counter()
    nodes, rels = adapter(csv_path, limit=args.limit)
    parse_s = time.perf_counter() - t0
    print(f"  parsed {sum(len(v) for v in nodes.values()):,} nodes (queued) / "
          f"{len(rels):,} relationships in {parse_s:.1f}s")

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session() as session:
            if args.reset:
                from scripts.seed_graph import clear_graph_batched

                clear_graph_batched(session)
                print("  graph cleared (--reset)")
            # autocommit batches (load_nodes/load_relationships chunk at 500):
            # one explicit transaction would risk the server's 60s transaction
            # timeout on the 53k-row data_synthetic load. MERGEs are idempotent,
            # so a partial failure is safe to re-run.
            t1 = time.perf_counter()
            load_nodes(session, nodes)
            load_relationships(session, rels)
            load_s = time.perf_counter() - t1
            print(f"  loaded in {load_s:.1f}s (total {load_s + parse_s:.1f}s)")
            # stamp which dataset is loaded so the dashboard can pick the right
            # fraud ground-truth table (labels live in the CSVs, not the graph)
            session.run("MERGE (d:Dataset {name: $name})", name=args.dataset)
            counts = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY label"
            ).data()
            print("  graph nodes:", ", ".join(f"{r['label']}={r['c']}" for r in counts))
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
