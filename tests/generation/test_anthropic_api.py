from __future__ import annotations

import contextlib
from types import SimpleNamespace

import anthropic
import pytest

from quarterly_rag.errors import ModelServerError
from quarterly_rag.generation.anthropic_api import AnthropicLLM
from quarterly_rag.generation.base import LLM, ChatMessage


class StubMessages:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class StubModels:
    def __init__(self, ids=(), error: Exception | None = None) -> None:
        self.ids = ids
        self.error = error

    def list(self):
        if self.error is not None:
            raise self.error
        return [SimpleNamespace(id=i) for i in self.ids]


def text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def thinking_block():
    return SimpleNamespace(type="thinking", thinking="")


def api_response(blocks, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        content=blocks,
        model="claude-opus-5",
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=3),
    )


def make_llm(messages: StubMessages | None = None, models: StubModels | None = None):
    client = SimpleNamespace(messages=messages or StubMessages(), models=models or StubModels())
    return AnthropicLLM("sk-test", "claude-opus-5", client=client)


def status_error(status_code: int, message: str) -> anthropic.APIStatusError:
    response = SimpleNamespace(status_code=status_code, headers={}, request=None)
    return anthropic.APIStatusError(message, response=response, body=None)


def test_system_goes_to_system_kwarg_and_text_blocks_are_joined() -> None:
    messages = StubMessages(
        response=api_response([thinking_block(), text_block("po"), text_block("ng")])
    )
    llm = make_llm(messages)
    assert isinstance(llm, LLM)
    assert llm.label == "anthropic/claude-opus-5"

    response = llm.chat(
        [ChatMessage(role="system", content="terse"), ChatMessage(role="user", content="ping")],
        temperature=0.7,
        max_tokens=16,
    )

    assert messages.kwargs["model"] == "claude-opus-5"
    assert messages.kwargs["system"] == "terse"
    assert messages.kwargs["messages"] == [{"role": "user", "content": "ping"}]
    assert messages.kwargs["max_tokens"] == 16
    assert "temperature" not in messages.kwargs
    assert response.text == "pong"
    assert response.model == "claude-opus-5"
    assert response.stop_reason == "end_turn"
    assert (response.input_tokens, response.output_tokens) == (10, 3)


def test_without_system_message_system_is_not_given() -> None:
    messages = StubMessages(response=api_response([text_block("hi")]))
    make_llm(messages).chat([ChatMessage(role="user", content="ping")])
    assert messages.kwargs["system"] is anthropic.NOT_GIVEN


def test_refusal_stop_reason_passes_through() -> None:
    messages = StubMessages(response=api_response([], stop_reason="refusal"))
    response = make_llm(messages).chat([ChatMessage(role="user", content="ping")])
    assert response.text == ""
    assert response.stop_reason == "refusal"


def test_api_status_error_becomes_model_server_error() -> None:
    llm = make_llm(StubMessages(error=status_error(401, "invalid x-api-key")))
    with pytest.raises(ModelServerError) as excinfo:
        llm.chat([ChatMessage(role="user", content="ping")])
    assert excinfo.value.status_code == 401
    assert "invalid x-api-key" in str(excinfo.value)


def test_connection_error_becomes_model_server_error() -> None:
    error = anthropic.APIConnectionError(request=SimpleNamespace())
    llm = make_llm(StubMessages(error=error))
    with pytest.raises(ModelServerError, match="APIConnectionError"):
        llm.chat([ChatMessage(role="user", content="ping")])


def test_list_models_sorted_and_errors_mapped() -> None:
    assert make_llm(models=StubModels(ids=("claude-sonnet-5", "claude-opus-5"))).list_models() == [
        "claude-opus-5",
        "claude-sonnet-5",
    ]
    llm = make_llm(models=StubModels(error=status_error(403, "forbidden")))
    with pytest.raises(ModelServerError) as excinfo:
        llm.list_models()
    assert excinfo.value.status_code == 403


def test_construction_never_raises_anything_but_model_server_error(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with contextlib.suppress(ModelServerError):
        AnthropicLLM("", "claude-opus-5")
