"""Token-cost estimation for provider-reported usage (v2).

When a provider returns a ``usage`` block, we can price the call. Prices are
per-1k-token USD and are *config defaults*, not a source of truth — enterprises
should set their contracted prices via env:

    LLM_PRICE_PER_1K_INPUT=0.0030   # gpt-4o-mini list, for example
    LLM_PRICE_PER_1K_OUTPUT=0.0120
"""

from __future__ import annotations

from graphrag.config import settings

# per-1k-token USD, input/output — list prices for common enterprise models
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.0100),
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4.1": (0.0020, 0.0080),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gpt-4.1-nano": (0.0001, 0.0004),
    "o3-mini": (0.0011, 0.0044),
    "llama-3.1-70b": (0.00059, 0.00079),
    "llama-3.3-70b": (0.00059, 0.00079),
    "mixtral-8x7b": (0.0006, 0.0006),
}

def _unknown_price() -> tuple[float, float]:
    """Live env prices (read per call so config changes are honoured)."""
    return (settings.LLM_PRICE_PER_1K_INPUT, settings.LLM_PRICE_PER_1K_OUTPUT)


def price_for(model: str) -> tuple[float, float]:
    """(input, output) USD per 1k tokens for a model name (fuzzy prefix match).

    Exact match wins; otherwise the longest matching table key wins
    ("gpt-4o-mini" must price as gpt-4o-mini, not gpt-4o).
    """
    name = (model or "").lower()
    if name in _PRICES:
        return _PRICES[name]
    for key in sorted(_PRICES, key=len, reverse=True):
        if name.startswith(key) or key.startswith(name):
            return _PRICES[key]
    return _unknown_price()


def estimate_cost(model: str, input_tokens: int | None,
                  output_tokens: int | None) -> float | None:
    """USD cost of one call; None when usage is unknown or prices are zero/unset."""
    if input_tokens is None and output_tokens is None:
        return None
    in_price, out_price = price_for(model)
    if in_price <= 0 and out_price <= 0:
        return None
    cost = (input_tokens or 0) / 1000 * in_price + \
           (output_tokens or 0) / 1000 * out_price
    return round(cost, 6)
