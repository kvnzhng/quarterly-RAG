from __future__ import annotations

import json

import httpx
import pytest

from quarterly_rag.errors import ModelServerError
from quarterly_rag.generation.base import LLM, ChatMessage
from quarterly_rag.generation.openai_compatible import OpenAICompatibleLLM

BASE = "http://ai-server.local:11434/v1"


def make_llm(handler, api_key: str = "secret") -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(BASE, api_key, "llama3.1:8b", transport=httpx.MockTransport(handler))


def chat_reply(text: str = "pong") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "llama3.1:8b",
            "choices": [
                {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1},
        },
    )


def test_chat_posts_openai_shape_and_parses_reply() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return chat_reply()

    llm = make_llm(handler)
    response = llm.chat(
        [ChatMessage(role="system", content="be terse"), ChatMessage(role="user", content="ping")],
        temperature=0.2,
        max_tokens=8,
    )

    assert seen["url"] == f"{BASE}/chat/completions"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["model"] == "llama3.1:8b"
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "ping"},
    ]
    assert seen["body"]["temperature"] == 0.2
    assert seen["body"]["max_tokens"] == 8
    assert response.text == "pong"
    assert response.model == "llama3.1:8b"
    assert response.stop_reason == "stop"
    assert (response.input_tokens, response.output_tokens) == (12, 1)


def test_satisfies_protocol_and_label() -> None:
    llm = make_llm(lambda request: chat_reply())
    assert isinstance(llm, LLM)
    assert llm.label == "openai_compatible/llama3.1:8b"


def test_no_auth_header_when_key_empty() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return chat_reply()

    make_llm(handler, api_key="").chat([ChatMessage(role="user", content="ping")])
    assert seen["auth"] is None


def test_http_error_becomes_model_server_error() -> None:
    llm = make_llm(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(ModelServerError) as excinfo:
        llm.chat([ChatMessage(role="user", content="ping")])
    assert excinfo.value.status_code == 500
    assert "boom" in str(excinfo.value)
    assert f"{BASE}/chat/completions" in str(excinfo.value)


def test_connection_error_becomes_model_server_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ModelServerError, match="ConnectError"):
        make_llm(handler).chat([ChatMessage(role="user", content="ping")])


def test_unexpected_shape_becomes_model_server_error() -> None:
    llm = make_llm(lambda request: httpx.Response(200, json={"nope": 1}))
    with pytest.raises(ModelServerError, match="unexpected chat response shape"):
        llm.chat([ChatMessage(role="user", content="ping")])


def test_missing_usage_is_tolerated() -> None:
    llm = make_llm(
        lambda request: httpx.Response(
            200, json={"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
        )
    )
    response = llm.chat([ChatMessage(role="user", content="ping")])
    assert response.text == "hi"
    assert response.model == "llama3.1:8b"
    assert response.input_tokens is None


def test_list_models_returns_sorted_ids() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["method"] = request.method
        return httpx.Response(
            200,
            json={"data": [{"id": "nomic-embed-text:latest"}, {"id": "llama3.1:8b"}]},
        )

    assert make_llm(handler).list_models() == ["llama3.1:8b", "nomic-embed-text:latest"]
    assert seen["url"] == f"{BASE}/models"
    assert seen["method"] == "GET"
