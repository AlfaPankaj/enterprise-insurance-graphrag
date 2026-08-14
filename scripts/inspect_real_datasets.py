"""Inspect the real insurance datasets in data/Real_datasets/."""
import csv
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Real_datasets"

FILES = ["fraud_oracle.csv", "insurance_claims.csv", "insurance_dataset.csv",
         "data_synthetic.csv"]

for fn in FILES:
    path = DATA / fn
    if not path.exists():
        print(f"=== {fn}: MISSING ===")
        continue
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = list(rd)
    print(f"=== {fn} ({len(rows)} rows x {len(header)} cols) ===")
    print("  cols:", ", ".join(header))
    # value cardinality for a few interesting columns
    for col in header:
        lo = col.lower()
        if any(k in lo for k in ("fraud", "claim", "policy", "status", "state",
                                 "type", "severity", "cause", "incident")):
            vals = Counter(r[i] for r in rows for i, c in enumerate(header)
                           if c == col and i < len(r))
            print(f"  {col}: {len(vals)} distinct | top: "
                  f"{', '.join(f'{v}={n}' for v, n in vals.most_common(5))}")
    print()

    # sample row
    if rows:
        print("  sample:")
        for c, v in zip(header, rows[0]):
            print(f"    {c} = {v[:50]}")
    print()
