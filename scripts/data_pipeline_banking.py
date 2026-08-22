#!/usr/bin/env python3
"""Generate the banking demo dataset (v2 — WS-E, G18).

Deterministic (seed=42), ground-truth-by-construction, mirroring the
insurance ``data_pipeline.py`` philosophy:

    (Customer)-[:HOLDS]->(Account)-[:POSTED]->(Transaction)
    (Account)-[:HAS_DISPUTE]->(Dispute)-[:ABOUT]->(Transaction)
    (Account)-[:HAS_ALERT]->(AMLAlert)

Writes ``data/samples/banking.json``, which
``scripts/ingest_banking_dataset.py`` loads into Neo4j and
``scripts/benchmark_banking_dataset.py`` uses to build ground-truth queries.

Usage:  python scripts/data_pipeline_banking.py [--customers 60]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "data" / "samples" / "banking.json"

FIRST_NAMES = ["Aarav", "Priya", "Rohan", "Sneha", "Vikram", "Ananya",
               "Karthik", "Meera", "Aditya", "Nisha", "Rahul", "Divya"]
LAST_NAMES = ["Sharma", "Patel", "Iyer", "Reddy", "Mehta", "Nair", "Gupta",
              "Rao", "Khan", "Joshi", "Das", "Menon"]
MERCHANTS = ["BlueSky Airlines", "MetroGrocer", "Northwind Traders",
             "Cafe Delight", "TechNova Store", "GreenLeaf Pharmacy",
             "Summit Hotels", "QuickFuel Station", "Orbit Electronics",
             "Harbor Freight Co."]
ACCOUNT_TYPES = ("SAVINGS", "CHECKING", "CORPORATE", "JOINT")
TXN_TYPES = ("POS", "ATM", "TRANSFER", "CASH", "INTERNATIONAL")
DISPUTE_REASONS = ("unauthorized charge", "duplicate processing",
                   "goods not received", "wrong amount charged")
AML_REASONS = ("structuring detected", "rapid movement of funds",
               "money laundering pattern", "sanctions list match",
               "unusual transaction velocity")


def _pct(value: float) -> str:
    return f"{value:,.2f}"


def generate(customers_n: int = 60, seed: int = 42) -> dict:
    rng = random.Random(seed)
    customers: list[dict] = []
    for i in range(1, customers_n + 1):
        customers.append({
            "id": f"CUST-{i:04d}",
            "name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
            "dob": f"{rng.randint(1950, 2002)}-{rng.randint(1, 12):02d}-"
                   f"{rng.randint(1, 28):02d}",
            "address": f"{rng.randint(1, 400)} {rng.choice(['Maple', 'Oak', 'Cedar', 'Pine'])} St",
            "phone": f"+91 98{rng.randint(10000000, 99999999)}",
            "email": f"customer{i}@example.com",
            "risk_tier": rng.choice(["LOW", "LOW", "MEDIUM", "HIGH"]),
        })

    accounts: list[dict] = []
    # 1-2 accounts per customer, round-robin so every customer holds at least one
    for i in range(1, customers_n + 1):
        for _ in range(2 if i % 3 == 0 else 1):
            accounts.append({
                "id": f"ACC-{len(accounts) + 1:04d}",
                "account_number": f"IB-{rng.randint(10**11, 10**12 - 1)}",
                "type": rng.choice(ACCOUNT_TYPES),
                "status": rng.choice(["ACTIVE"] * 8 + ["FROZEN", "CLOSED"]),
                "balance": round(rng.uniform(500, 250_000), 2),
                "currency": "USD",
                "opened_date": f"{rng.randint(2010, 2025)}-{rng.randint(1, 12):02d}-01",
                "customer_id": f"CUST-{i:04d}",
            })
    accounts_n = len(accounts)

    transactions: list[dict] = []
    txn_merchants: dict[str, set[str]] = {}  # merchant -> accounts posted to
    for i in range(1, 401):
        merchant = rng.choice(MERCHANTS)
        account = accounts[rng.randrange(accounts_n)]
        txn = {
            "id": f"TXN-{i:06d}",
            "transaction_id": f"TRX{i:08d}",
            "account_id": account["id"],
            "type": rng.choice(TXN_TYPES),
            "amount": round(rng.uniform(5, 45_000), 2),
            "date": f"2026-0{rng.randint(1, 8)}-{rng.randint(1, 28):02d}",
            "merchant": merchant,
            "status": rng.choice(["POSTED"] * 9 + ["PENDING", "REVERSED"]),
            "currency": "USD",
        }
        transactions.append(txn)
        txn_merchants.setdefault(merchant, set()).add(account["id"])

    # disputes: one per disputed transaction (30 of them)
    disputed_txns = rng.sample(transactions, 30)
    disputes: list[dict] = []
    for i, txn in enumerate(disputed_txns, start=1):
        disputes.append({
            "id": f"DSP-{i:04d}",
            "dispute_id": f"DSP-2026-{i:05d}",
            "transaction_id": txn["id"],
            "account_id": txn["account_id"],
            "reason": rng.choice(DISPUTE_REASONS),
            "status": rng.choice(["OPEN", "UNDER_REVIEW", "RESOLVED", "REJECTED"]),
            "amount": txn["amount"],
            "opened_date": f"2026-08-{rng.randint(1, 20):02d}",
        })

    # AML alerts on 18 accounts with the highest balances (money-movement risk)
    alert_accounts = sorted(accounts, key=lambda a: a["balance"], reverse=True)[:18]
    aml_alerts: list[dict] = []
    for i, account in enumerate(alert_accounts, start=1):
        aml_alerts.append({
            "id": f"AML-{i:04d}",
            "alert_id": f"AML-2026-{i:05d}",
            "account_id": account["id"],
            "customer_id": account["customer_id"],
            "reason": rng.choice(AML_REASONS),
            "severity": rng.choice(["LOW", "MEDIUM", "HIGH"]),
            "status": rng.choice(["OPEN", "CLOSED"]),
            "amount": account["balance"],
            "raised_at": f"2026-08-{rng.randint(1, 20):02d}",
        })

    # ground truth for the benchmark: designate one transaction as the anchor
    # for a no-id paraphrase query by giving it a merchant name used nowhere
    # else in the dataset
    anchor_txn = transactions[rng.randrange(len(transactions))]
    unique_merchant = "Aurora Rare Goods Co."
    anchor_txn["merchant"] = unique_merchant
    unique_accounts = [anchor_txn["account_id"]]

    return {
        "seed": seed,
        "counts": {
            "customers": len(customers), "accounts": accounts_n,
            "transactions": len(transactions), "disputes": len(disputes),
            "aml_alerts": len(aml_alerts),
        },
        "unique_merchant": unique_merchant,
        "unique_merchant_account": sorted(unique_accounts)[0],
        "customers": customers,
        "accounts": accounts,
        "transactions": transactions,
        "disputes": disputes,
        "aml_alerts": aml_alerts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=60)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    data = generate(customers_n=args.customers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote banking demo dataset to {args.out.relative_to(PROJECT_ROOT)}")
    print("  counts:", ", ".join(f"{k}={v}" for k, v in data["counts"].items()))
    print(f"  unique merchant (benchmark anchor): {data['unique_merchant']} "
          f"-> {data['unique_merchant_account']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
