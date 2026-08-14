"""Token accounting for the cost dashboard (Shot 2).

Uses tiktoken (``cl100k_base``) as a fast, dependency-light proxy for the
Llama tokenizer — the savings *ratio* is what matters, and before/after use the
same tokenizer. Falls back to a ~4 chars/token heuristic if tiktoken is absent.
"""

from __future__ import annotations

import functools

from graphrag.config import settings


@functools.lru_cache(maxsize=4)
def _encoding(model: str):
    """Lazily load the tiktoken encoding (cache the BPE tables once per model)."""
    import tiktoken

    return tiktoken.get_encoding(model)


def count_tokens(text: str, model: str | None = None) -> int:
    """Return the number of tokens in ``text`` for the given (or configured) model."""
    model = model or settings.TOKENIZER_MODEL
    try:
        return len(_encoding(model).encode(text))
    except Exception:
        # heuristic fallback: ~4 characters per token (BPE average)
        return 0 if not text else max(1, int(len(text) / 4))
