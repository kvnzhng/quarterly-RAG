"""The tracer, and the promise that it never breaks an answer (RAG-013).

Nothing here touches a Langfuse server. `LangfuseTracer` is exercised against a fake client
that records what it was asked to do, and against one that raises on every call, because the
second is the case that matters: a broken tracer must be invisible to the caller.
"""

from __future__ import annotations

from typing import Any

import pytest

from quarterly_rag.config import Settings
from quarterly_rag.observability.tracing import (
    BOOLEAN,
    GENERATION,
    LangfuseTracer,
    NullTracer,
    Span,
    Tracer,
    build_tracer,
    configured,
    reachable,
    trace_metadata,
)


class FakeObservation:
    def __init__(self, name: str, recorder: list, trace_id: str = "trace-1") -> None:
        self.name = name
        self.trace_id = trace_id
        self._recorder = recorder

    def update(self, **attributes: Any) -> None:
        self._recorder.append(("update", self.name, attributes))

    def __enter__(self) -> FakeObservation:
        return self

    def __exit__(self, *exc: object) -> None:
        self._recorder.append(("end", self.name, {}))


class FakeClient:
    def __init__(self) -> None:
        self.calls: list = []

    def start_as_current_observation(self, *, name: str, as_type: str, **attributes: Any):
        self.calls.append(("start", name, {"as_type": as_type, **attributes}))
        return FakeObservation(name, self.calls)

    def create_score(self, **kwargs: Any) -> None:
        self.calls.append(("score", kwargs.get("name", ""), kwargs))

    def flush(self) -> None:
        self.calls.append(("flush", "", {}))


class BrokenClient:
    """Every call fails, the way a client pointed at a dead server eventually does."""

    def start_as_current_observation(self, **kwargs: Any):
        raise RuntimeError("no server")

    def create_score(self, **kwargs: Any) -> None:
        raise RuntimeError("no server")

    def flush(self) -> None:
        raise RuntimeError("no server")


def test_the_null_tracer_satisfies_the_protocol_and_does_nothing() -> None:
    tracer = NullTracer()
    assert isinstance(tracer, Tracer)
    assert not tracer.enabled
    with tracer.span("anything", input={"a": 1}) as span:
        assert isinstance(span, Span)
        assert span.trace_id == ""
        span.update(output="ignored")
    tracer.score("", "name", 1.0)
    tracer.flush()


def test_spans_reach_the_client_with_their_kind_and_attributes() -> None:
    client = FakeClient()
    tracer = LangfuseTracer(client)
    with tracer.span("generation", kind=GENERATION, model="m", input={"q": "why?"}) as span:
        assert span.trace_id == "trace-1"
        span.update(output="because", usage_details={"input": 10})

    starts = [c for c in client.calls if c[0] == "start"]
    assert starts[0][1] == "generation"
    assert starts[0][2]["as_type"] == GENERATION
    assert starts[0][2]["model"] == "m"
    assert ("update", "generation", {"output": "because", "usage_details": {"input": 10}}) in (
        client.calls
    )
    assert client.calls[-1][0] == "end"


def test_scores_carry_their_data_type() -> None:
    client = FakeClient()
    LangfuseTracer(client).score("trace-1", "refused_correctly", True, data_type=BOOLEAN)
    (call,) = (c for c in client.calls if c[0] == "score")
    assert call[2]["trace_id"] == "trace-1"
    assert call[2]["value"] is True
    assert call[2]["data_type"] == BOOLEAN


def test_a_score_without_a_trace_is_dropped_rather_than_sent() -> None:
    client = FakeClient()
    LangfuseTracer(client).score("", "faithfulness", 1.0)
    assert not [c for c in client.calls if c[0] == "score"]


def test_a_broken_client_never_reaches_the_caller() -> None:
    """The whole promise of this module: tracing cannot turn an answer into a traceback."""
    tracer = LangfuseTracer(BrokenClient())
    with tracer.span("retrieval") as span:
        assert span.trace_id == ""
        span.update(output={"chunks": []})
    tracer.score("trace-1", "faithfulness", 1.0)
    tracer.flush()


def test_tracing_is_off_unless_the_host_and_both_keys_are_set(settings: Settings) -> None:
    assert not configured(settings)
    assert isinstance(build_tracer(settings), NullTracer)

    half = settings.model_copy(update={"langfuse_public_key": "pk"})
    assert not configured(half)
    assert isinstance(build_tracer(half), NullTracer)

    whole = settings.model_copy(update={"langfuse_public_key": "pk", "langfuse_secret_key": "sk"})
    assert configured(whole)


def test_trace_metadata_names_the_model_and_never_the_endpoint(settings: Settings) -> None:
    """The server address is nobody's business but the operator's, so it is not in a trace."""
    configured_settings = settings.model_copy(
        update={"llm_base_url": "http://ai-server.local:11434/v1", "llm_model": "qwen3.8-27b-64k"}
    )
    metadata = trace_metadata(configured_settings, {"context": "gold"})
    assert metadata["llm_model"] == "qwen3.8-27b-64k"
    assert metadata["context"] == "gold"
    assert "ai-server.local" not in str(metadata)
    assert not any("url" in key for key in metadata)


def test_a_failure_inside_a_span_still_reaches_the_caller() -> None:
    """Tracing must not eat the error it was watching.

    A `@contextmanager` that catches the exception thrown into its `yield` and does not
    re-raise suppresses it for the caller. That turned a model-server error into a silent
    `None` from `Pipeline.ask`, with the server's message gone.
    """
    tracer = LangfuseTracer(FakeClient())
    with pytest.raises(ValueError, match="model server down"), tracer.span("generation"):
        raise ValueError("model server down")


def test_the_span_is_closed_even_when_the_body_raises() -> None:
    client = FakeClient()
    with pytest.raises(ValueError), LangfuseTracer(client).span("generation"):
        raise ValueError("boom")
    assert client.calls[-1][0] == "end"


def test_a_server_that_does_not_answer_gives_a_tracer_that_does_nothing(settings: Settings) -> None:
    """Otherwise the exporter retries inside flush and an answer takes eleven seconds longer."""
    dead = settings.model_copy(
        update={
            "langfuse_host": "http://127.0.0.1:9",
            "langfuse_public_key": "pk",
            "langfuse_secret_key": "sk",
        }
    )
    assert configured(dead)
    assert not reachable(dead, timeout=0.5)
    assert isinstance(build_tracer(dead), NullTracer)
