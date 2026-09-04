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


def test_task_prefixes_are_applied_to_each_side() -> None:
    """nomic-embed-text is trained with these markers; omitting them costs real recall."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        count = len(seen[-1]["input"])
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": [0.1]} for i in range(count)]}
        )

    embedder = OpenAICompatibleEmbedder(
        BASE,
        "secret",
        "nomic-embed-text",
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        transport=httpx.MockTransport(handler),
    )
    embedder.embed_documents(["net sales rose"])
    embedder.embed_query("what were net sales?")

    assert seen[0]["input"] == ["search_document: net sales rose"]
    assert seen[1]["input"] == ["search_query: what were net sales?"]


def test_no_prefixes_by_default() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1]}]})

    make_embedder(handler).embed_documents(["plain"])
    assert seen[0]["input"] == ["plain"]


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

    vectors = embedder.embed_documents(["first", "second"])

    assert seen["url"] == f"{BASE}/embeddings"
    assert seen["body"] == {"model": "nomic-embed-text", "input": ["first", "second"]}
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_empty_input_makes_no_request() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"data": []})

    assert make_embedder(handler).embed_documents([]) == []
    assert calls == []


def test_count_mismatch_is_an_error() -> None:
    embedder = make_embedder(
        lambda request: httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1]}]})
    )
    with pytest.raises(ModelServerError, match="asked for 2 embeddings, got 1"):
        embedder.embed_documents(["a", "b"])


def test_bad_shape_is_an_error() -> None:
    embedder = make_embedder(lambda request: httpx.Response(200, json={"data": [{"oops": 1}]}))
    with pytest.raises(ModelServerError, match="unexpected embeddings response shape"):
        embedder.embed_documents(["a"])


def test_http_error_is_an_error() -> None:
    embedder = make_embedder(lambda request: httpx.Response(404, text="model not found"))
    with pytest.raises(ModelServerError) as excinfo:
        embedder.embed_documents(["a"])
    assert excinfo.value.status_code == 404
