"""Banking domain spec (v2 — WS-E, G18).

The second business domain for the digital-operations pitch: transaction
disputes and AML alerting.

    (Customer)-[:HOLDS]->(Account)-[:POSTED]->(Transaction)
    (Account)-[:HAS_DISPUTE]->(Dispute)-[:ABOUT]->(Transaction)
    (Account)-[:HAS_ALERT]->(AMLAlert)

Entity ids: ``CUST-``, ``ACC-``, ``TXN-``, ``DSP-``, ``AML-`` + digits.
Data: ``scripts/data_pipeline_banking.py`` generates the demo dataset with
ground truth by construction (``data/samples/banking.json``), ingested by
``scripts/ingest_banking_dataset.py``, benchmarked by
``scripts/benchmark_banking_dataset.py``.
"""

from __future__ import annotations

from graphrag.domains.base import DomainSpec

BANKING = DomainSpec(
    name="banking",
    description="Retail banking: accounts, transactions, disputes, AML alerts.",
    required_fields={
        "Customer": ("id", "name", "dob", "risk_tier"),
        "Account": ("id", "account_number", "type", "status", "balance",
                    "currency", "opened_date"),
        "Transaction": ("id", "transaction_id", "type", "amount", "date",
                        "status", "currency"),
        "Dispute": ("id", "dispute_id", "reason", "status", "amount",
                    "opened_date"),
        "AMLAlert": ("id", "alert_id", "reason", "severity", "status",
                     "amount"),
    },
    relationships=(
        ("Customer", "Account", "HOLDS"),
        ("Account", "Transaction", "POSTED"),
        ("Account", "Dispute", "HAS_DISPUTE"),
        ("Dispute", "Transaction", "ABOUT"),
        ("Account", "AMLAlert", "HAS_ALERT"),
    ),
    id_pattern=r"(?:CUST|ACC|TXN|DSP|AML)-\d{3,}",
    id_patterns={
        "Customer": r"^CUST-\d{3,}$",
        "Account": r"^ACC-\d{3,}$",
        "Transaction": r"^TXN-\d{3,}$",
        "Dispute": r"^DSP-\d{3,}$",
        "AMLAlert": r"^AML-\d{3,}$",
    },
    keyword_props=(
        "name", "address", "email", "phone", "risk_tier",
        "account_number", "type", "status", "currency",
        "transaction_id", "merchant", "reason",
        "dispute_id", "alert_id", "severity", "opened_date", "raised_at",
    ),
    numeric_props=(
        ("balance", "Account"),
        ("amount", "Transaction"),
        ("amount", "Dispute"),
    ),
    prop_focus={
        "balance": (("balance", "Account"),),
    },
    text_props={
        "Customer": ("name", "risk_tier"),
        "Account": ("account_number", "type", "status", "balance", "currency"),
        "Transaction": ("transaction_id", "type", "amount", "date",
                        "merchant", "status"),
        "Dispute": ("dispute_id", "reason", "status", "amount", "opened_date"),
        "AMLAlert": ("alert_id", "reason", "severity", "status", "amount"),
    },
    node_kinds={
        "Customer": "banking customer",
        "Account": "bank account",
        "Transaction": "banking transaction",
        "Dispute": "account dispute",
        "AMLAlert": "anti-money-laundering alert",
    },
    label_hints=(
        (("dispute", "disputes", "chargeback", "chargebacks"), "Dispute"),
        (("aml", "laundering", "anti-money", "structuring"), "AMLAlert"),
        (("transaction", "transactions", "payment", "payments"), "Transaction"),
        (("account", "accounts"), "Account"),
        (("customer", "customers", "client", "clients"), "Customer"),
    ),
    # banking schema nouns — added to the seeding stopwords so "account" /
    # "transaction" never seed generic nodes (value tokens do the real work)
    stopwords=frozenset({
        "account", "accounts", "transaction", "transactions",
        "dispute", "disputes", "alert", "alerts", "balance", "balances",
        "merchant", "merchants", "aml", "currency", "currencies",
    }),
    pii={
        ("Customer", "name"): "PII_IDENTITY",
        ("Customer", "dob"): "PII_IDENTITY",
        ("Customer", "address"): "PII_CONTACT",
        ("Customer", "phone"): "PII_CONTACT",
        ("Customer", "email"): "PII_CONTACT",
    },
)
