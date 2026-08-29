"""Shared HTTP transport pooling and payload-parity controls."""

from __future__ import annotations

from contextlib import nullcontext

from graphrag import http_client


class _FakeClient:
    def __init__(self):
        self.calls: list[tuple] = []

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return "get-response"

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return "post-response"

    def stream(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return nullcontext("stream-response")


def test_request_helpers_reuse_shared_client(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(http_client, "_CLIENT", client)

    assert http_client.request_get("https://provider.test/health", timeout=2) == \
        "get-response"
    assert http_client.request_post(
        "https://provider.test/generate", json={"prompt": "q"}, timeout=90
    ) == "post-response"
    with http_client.request_stream(
        "POST", "https://provider.test/stream", json={"stream": True}
    ) as response:
        assert response == "stream-response"

    assert [call[0] for call in client.calls] == ["GET", "POST", "POST"]
    assert client.calls[1][2]["json"] == {"prompt": "q"}


def test_get_http_client_is_process_wide():
    http_client.close_http_client()
    try:
        first = http_client.get_http_client()
        second = http_client.get_http_client()
        assert first is second
    finally:
        http_client.close_http_client()
