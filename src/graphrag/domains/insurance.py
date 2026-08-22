"""Insurance domain spec (v2 — WS-E, G18).

Encodes the v1 ontology EXACTLY — every constant below was previously
hardcoded in ``graph_retriever`` / ``reranker`` / ``entity_extractor`` /
``pii``. Moving them into a spec (and merging across domains) must not change
a single insurance behavior; the full v1 suite + the 10,200-query benchmark
guard that.
"""

from __future__ import annotations

from graphrag.domains.base import DomainSpec

INSURANCE = DomainSpec(
    name="insurance",
    description="Commercial insurance: policies, claims, fraud flags, endorsements.",
    required_fields={
        "Policyholder": ("id", "name", "dob", "address", "phone", "email",
                         "risk_score"),
        "Policy": ("id", "policy_number", "type", "start_date", "end_date",
                   "premium", "deductible", "status"),
        "Coverage": ("id", "code", "category", "limit", "deductible"),
        "Claim": ("id", "claim_number", "date", "amount", "status", "cause"),
        "FraudFlag": ("id", "reason", "confidence", "severity", "created_by"),
        "Investigator": ("id", "name", "role", "email"),
        "Endorsement": ("id", "endorsement_number", "type", "effective_date",
                        "premium_adjustment"),
    },
    relationships=(
        ("Policyholder", "Policy", "HAS_POLICY"),
        ("Policy", "Claim", "HAS_CLAIM"),
        ("Policy", "Coverage", "COVERS"),
        ("Policy", "Endorsement", "ENDORSED_BY"),
        ("Claim", "FraudFlag", "FRAUD_DETECTED"),
        ("Investigator", "Claim", "INVESTIGATES_CLAIM"),
    ),
    id_pattern=r"(?:POL|CLM|PH|END|FRD|INV)-\d{3,}",
    id_patterns={
        "Policyholder": r"^PH-\d{3,}$",
        "Policy": r"^POL-\d{3,}$",
        "Coverage": r"^COV-\d{3,}$",
        "Claim": r"^CLM-\d{3,}$",
        "FraudFlag": r"^FRD-\d{3,}$",
        "Investigator": r"^INV-\d{3,}$",
        "Endorsement": r"^END-\d{3,}$",
    },
    keyword_props=(
        "name", "address", "email", "policy_number", "type", "status",
        "claim_number", "cause", "reason", "severity", "endorsement_number",
        "role", "category", "occupation",
    ),
    numeric_props=(
        ("amount", "Claim"), ("limit", "Coverage"), ("premium", "Policy"),
        ("deductible", "Policy"), ("risk_score", "Policyholder"),
        ("confidence", "FraudFlag"),
    ),
    prop_focus={
        "premium": (("premium", "Policy"),),
        "deductible": (("deductible", "Policy"), ("deductible", "Coverage")),
        "amount": (("amount", "Claim"),),
        "limit": (("limit", "Coverage"),),
        "risk": (("risk_score", "Policyholder"),),
        "confidence": (("confidence", "FraudFlag"),),
    },
    text_props={
        "Policyholder": ("name", "risk_score"),
        "Policy": ("policy_number", "type", "status", "premium", "deductible",
                   "start_date", "end_date"),
        "Claim": ("claim_number", "status", "amount", "date", "cause"),
        "FraudFlag": ("severity", "confidence", "reason", "created_by"),
        "Endorsement": ("endorsement_number", "type", "effective_date",
                        "premium_adjustment"),
        "Investigator": ("name", "role", "email"),
        "Coverage": ("code", "category", "limit", "deductible"),
    },
    node_kinds={
        "Policyholder": "policyholder",
        "Policy": "insurance policy",
        "Claim": "insurance claim",
        "FraudFlag": "fraud flag",
        "Endorsement": "policy endorsement",
        "Investigator": "claims investigator",
        "Coverage": "coverage",
    },
    label_hints=(
        (("coverage", "coverages", "covers"), "Coverage"),
        (("fraud", "flag", "flagged"), "FraudFlag"),
        (("investigat", "handled by", "assigned"), "Investigator"),
        (("endorsement", "endorsements"), "Endorsement"),
        (("policy", "policies"), "Policy"),
        (("claim", "claims"), "Claim"),
        (("policyholder", "holder", "held by"), "Policyholder"),
    ),
    stopwords=frozenset(),
    pii={
        ("Policyholder", "name"): "PII_IDENTITY",
        ("Policyholder", "dob"): "PII_IDENTITY",
        ("Policyholder", "address"): "PII_CONTACT",
        ("Policyholder", "phone"): "PII_CONTACT",
        ("Policyholder", "email"): "PII_CONTACT",
        ("Investigator", "name"): "PII_IDENTITY",
        ("Investigator", "email"): "PII_CONTACT",
        ("Policy", "policyholder_name"): "PII_IDENTITY",
    },
)
