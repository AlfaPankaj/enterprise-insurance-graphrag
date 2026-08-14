"""Token counter sanity tests."""

from graphrag.token_counter import count_tokens


def test_count_tokens_positive():
    assert count_tokens("This is a short sentence about insurance claims.") > 0


def test_count_tokens_monotonic():
    short = count_tokens("hello world")
    long = count_tokens("hello world " * 100)
    assert long > short


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_consistent():
    # same text must always yield the same count (deterministic accounting)
    text = "What is the status of policy POL-0005?"
    assert count_tokens(text) == count_tokens(text)
