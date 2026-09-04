from __future__ import annotations

import json

import httpx
import pytest

from quarterly_rag.errors import ModelServerError
from quarterly_rag.indexing.base import Embedder
from quarterly_rag.indexing.openai_compatible import OpenAICompatibleEmbedder

BASE = "http://ai-server.local:11434/v1"


def make_embedder(handler) -> OpenAICompatibleEmbedder:
    return OpenAICompatibleEmbedder(
        BASE, "secret", "nomic-embed-text", transport=httpx.MockTransport(handler)
    )


def test_embed_posts_inputs_and_orders_by_index() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            },
        )

    embedder = make_embedder(handler)
    assert isinstance(embedder, Embedder)
    assert embedder.label == "openai_compatible/nomic-embed-text"

    vectors = embedder.embed(["first", "second"])

    assert seen["url"] == f"{BASE}/embeddings"
    assert seen["body"] == {"model": "nomic-embed-text", "input": ["first", "second"]}
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_empty_input_makes_no_request() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"data": []})

    assert make_embedder(handler).embed([]) == []
    assert calls == []


def test_count_mismatch_is_an_error() -> None:
    embedder = make_embedder(
        lambda request: httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1]}]})
    )
    with pytest.raises(ModelServerError, match="asked for 2 embeddings, got 1"):
        embedder.embed(["a", "b"])


def test_bad_shape_is_an_error() -> None:
    embedder = make_embedder(lambda request: httpx.Response(200, json={"data": [{"oops": 1}]}))
    with pytest.raises(ModelServerError, match="unexpected embeddings response shape"):
        embedder.embed(["a"])


def test_http_error_is_an_error() -> None:
    embedder = make_embedder(lambda request: httpx.Response(404, text="model not found"))
    with pytest.raises(ModelServerError) as excinfo:
        embedder.embed(["a"])
    assert excinfo.value.status_code == 404
