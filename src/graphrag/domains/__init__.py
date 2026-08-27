"""Domain registry (v2 — WS-E, G18).

Registers the business domains the pipeline understands and merges their
specs into the constants the retriever/reranker/extractor consult. The
insurance spec carries the EXACT v1 values, and insurance is always first in
every merge — so all merged constants are pure supersets and v1 behavior is
bit-identical.

Adding a domain = adding one ``DomainSpec`` here (see ``banking.py``); every
pipeline stage picks it up automatically.
"""

from __future__ import annotations

import re

from graphrag.domains.banking import BANKING
from graphrag.domains.base import DomainSpec
from graphrag.domains.insurance import INSURANCE

# insurance first: its regex alternative wins on the merged pattern, and its
# entries keep their positions in every merged collection
DOMAINS: tuple[DomainSpec, ...] = (INSURANCE, BANKING)
BY_NAME: dict[str, DomainSpec] = {d.name: d for d in DOMAINS}


def get_domain(name: str) -> DomainSpec | None:
    return BY_NAME.get(name)


def specs() -> tuple[DomainSpec, ...]:
    return DOMAINS


def merged_entity_id_re() -> re.Pattern:
    """Combined entity-id regex: ``\\b((?:POL|CLM|...|ACC|TXN|...)-\\d{3,})\\b``."""
    fragments = "|".join(d.id_pattern for d in DOMAINS)
    return re.compile(rf"\b((?:{fragments}))\b")


def merged_keyword_props() -> list[str]:
    return sorted({p for d in DOMAINS for p in d.keyword_props})


def merged_numeric_props() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for d in DOMAINS:
        for pair in d.numeric_props:
            if pair not in out:
                out.append(pair)
    return out


def merged_prop_focus() -> dict[str, list[tuple[str, str]]]:
    merged: dict[str, list[tuple[str, str]]] = {}
    for d in DOMAINS:
        for word, pairs in d.prop_focus.items():
            merged[word] = list(pairs)
    return merged


def merged_text_props() -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for d in DOMAINS:
        for label, props in d.text_props.items():
            merged[label] = list(props)
    return merged


def merged_node_kinds() -> dict[str, str]:
    return {label: kind for d in DOMAINS
            for label, kind in d.node_kinds.items()}


def merged_label_hints() -> list[tuple[tuple[str, ...], str]]:
    merged: list[tuple[tuple[str, ...], str]] = []
    seen: set[tuple] = set()
    for d in DOMAINS:
        for hint in d.label_hints:
            key = (tuple(hint[0]), hint[1])
            if key not in seen:
                seen.add(key)
                merged.append(hint)
    return merged


def merged_stopwords() -> frozenset[str]:
    return frozenset(w for d in DOMAINS for w in d.stopwords)


def merged_required_fields() -> dict[str, list[str]]:
    return {label: list(fields) for d in DOMAINS
            for label, fields in d.required_fields.items()}


def merged_pii_classes() -> dict[tuple[str, str], str]:
    return {key: cls for d in DOMAINS for key, cls in d.pii.items()}


def merged_id_patterns() -> dict[str, re.Pattern]:
    """Strict per-label id patterns from every domain spec."""
    return {label: re.compile(pattern) for d in DOMAINS
            for label, pattern in d.id_patterns.items()}
