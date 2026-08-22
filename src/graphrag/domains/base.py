"""Domain specification dataclass (v2 — WS-E, G18).

A domain declares the ontology a GraphRAG pipeline needs:

* node labels + their required fields (drives extraction confidence scoring)
* relationship verbs (schema documentation + adapters)
* entity-id pattern (drives seed anchoring for that domain's ids)
* retrieval surfaces: keyword properties, numeric properties, serialization
  text props, natural-language node kinds, answer-type label hints,
  numeric prop-focus words, and extra schema-noun stopwords
* PII classification for the domain's properties

The registry (``graphrag.domains``) merges all registered specs into the
constants the retriever/reranker/extractor consult — so adding a domain is
purely additive and the insurance behavior stays bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainSpec:
    """One business domain's ontology contract."""

    name: str
    description: str
    # label -> required properties (mirrors the domain's schema doc)
    required_fields: dict[str, tuple[str, ...]]
    # (from_label, to_label, relationship_verb) catalog
    relationships: tuple[tuple[str, str, str], ...]
    # regex fragment matching this domain's entity ids (e.g. r"POL-\d{3,}")
    id_pattern: str
    # property names scanned for keyword seeding
    keyword_props: tuple[str, ...]
    # (property, owning_label) pairs for threshold/numeric seeding
    numeric_props: tuple[tuple[str, str], ...]
    # query word -> (property, label) pairs narrowing the threshold scan
    prop_focus: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    # label -> properties serialized into LLM context lines
    text_props: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # label -> natural-language descriptor ("Policy" -> "insurance policy")
    node_kinds: dict[str, str] = field(default_factory=dict)
    # query phrases -> node label (the answer-type prior)
    label_hints: tuple[tuple[tuple[str, ...], str], ...] = ()
    # extra schema-noun stopwords for keyword seeding
    stopwords: frozenset[str] = frozenset()
    # (label, property) -> PII class (PII_IDENTITY / PII_CONTACT / PII_HEALTH)
    pii: dict[tuple[str, str], str] = field(default_factory=dict)
    # label -> strict id regex for confidence scoring (demo schemes)
    id_patterns: dict[str, str] = field(default_factory=dict)
