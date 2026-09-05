"""The pipeline's stages, and the trace they leave behind (RAG-011, RAG-013).

No model and no Langfuse server: a fake retriever, a fake LLM, and a tracer that records
what it was asked to record. The point is that the traced path and the untraced path are the
same code, so both are checked here.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from quarterly_rag.generation.base import ChatMessage, ChatResponse
from quarterly_rag.generation.refusal import CorpusScope, GateSettings
from quarterly_rag.pipeline import Pipeline
from quarterly_rag.retrieval.base import RetrievedChunk

TABLE = "(In millions)\nTotal net sales | 109,417 | 94,036"


class RecordingSpan:
    def __init__(self, name: str, calls: list) -> None:
        self.name = name
        self.trace_id = "trace-1"
        self._calls = calls

    def update(self, **attributes: Any) -> None:
        self._calls.append(("update", self.name, attributes))


class RecordingTracer:
    """A tracer that keeps everything, so a test can assert on the shape of a trace."""

    enabled = True

    def __init__(self) -> None:
        self.calls: list = []

    @contextmanager
    def span(self, name: str, *, kind: str = "span", **attributes: Any) -> Iterator[RecordingSpan]:
        self.calls.append(("start", name, {"kind": kind, **attributes}))
        yield RecordingSpan(name, self.calls)
        self.calls.append(("end", name, {}))

    def score(self, trace_id: str, name: str, value: Any, **kwargs: Any) -> None:
        self.calls.append(("score", name, {"trace_id": trace_id, "value": value, **kwargs}))

    def flush(self) -> None:
        self.calls.append(("flush", "", {}))

    def names(self, kind: str = "start") -> list[str]:
        return [name for call, name, _ in self.calls if call == kind]

    def attributes(self, call: str, name: str) -> dict:
        return next(a for c, n, a in self.calls if c == call and n == name)


class FakeLLM:
    label = "fake/llm"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def chat(self, messages: Sequence[ChatMessage], *, temperature=0.0, max_tokens=1024):
        return ChatResponse(
            text=self.reply,
            model="fake",
            stop_reason="stop",
            input_tokens=120,
            output_tokens=30,
        )

    def list_models(self) -> list[str]:
        return ["fake"]


class FakeRetriever:
    name = "fake"

    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results

    def retrieve(self, question: str, k: int = 5, where: dict | None = None):
        return self.results[:k]


@pytest.fixture
def hit(make_chunk):
    return RetrievedChunk(chunk=make_chunk("a:1-2", TABLE), score=0.9, rank=1, retriever="fake")


def build(retriever, llm, tracer=None, tickers=("AAPL",)) -> Pipeline:
    return Pipeline(
        retriever=retriever,
        llm=llm,
        scope=CorpusScope(
            tickers=frozenset(tickers),
            company_words=frozenset({"apple"}),
            fiscal_years=frozenset({2026}),
        ),
        gate=GateSettings(min_retrieval_score=0.0),
        **({"tracer": tracer} if tracer else {}),
    )


def test_an_answered_question_traces_every_stage_in_order(hit) -> None:
    tracer = RecordingTracer()
    pipeline = build(FakeRetriever([hit]), FakeLLM("Net sales were $109,417 million [c1]."), tracer)
    outcome = pipeline.ask("What were Apple's net sales?", k=5)

    assert not outcome.refused
    assert outcome.trace_id == "trace-1"
    assert tracer.names() == [
        "rag ask",
        "scope-gate",
        "retrieval",
        "retrieval-gate",
        "generation",
        "verification",
        "answer-gate",
    ]


def test_the_generation_span_records_the_model_and_its_tokens(hit) -> None:
    tracer = RecordingTracer()
    build(FakeRetriever([hit]), FakeLLM("Net sales were $109,417 million [c1]."), tracer).ask("q")

    start = tracer.attributes("start", "generation")
    assert start["kind"] == "generation"
    assert start["model"] == "fake/llm"
    update = tracer.attributes("update", "generation")
    assert update["usage_details"] == {"input": 120, "output": 30}


def test_the_verification_span_records_what_the_checks_found(hit) -> None:
    tracer = RecordingTracer()
    build(FakeRetriever([hit]), FakeLLM("Net sales were $109,417 million [c1]."), tracer).ask("q")

    found = tracer.attributes("update", "verification")["output"]
    assert found["fully_grounded"] is True
    assert found["citations"] == 1
    assert found["cited_sentences"] == 1
    assert found["unsupported_sentences"] == 0


def test_a_refusal_stops_the_trace_early_and_marks_it(hit) -> None:
    """An out-of-scope question never reaches a model, and the trace shows exactly that."""
    tracer = RecordingTracer()
    pipeline = build(FakeRetriever([hit]), FakeLLM("never used"), tracer)
    outcome = pipeline.ask("What were Microsoft's net sales?", k=5)

    assert outcome.refused
    assert tracer.names() == ["rag ask", "scope-gate"]
    root = tracer.attributes("update", "rag ask")
    assert root["level"] == "WARNING"
    assert root["output"]["refused"] is True
    assert root["output"]["reason"] == "out_of_scope"


def test_the_untraced_path_is_the_same_path(hit) -> None:
    """With no tracer the pipeline still answers, and carries no trace id."""
    pipeline = build(FakeRetriever([hit]), FakeLLM("Net sales were $109,417 million [c1]."))
    outcome = pipeline.ask("What were Apple's net sales?", k=5)
    assert not outcome.refused
    assert outcome.trace_id == ""
    assert outcome.answer is not None
    assert outcome.answer.fully_grounded


def test_no_retrieved_passages_means_no_model_call(make_chunk) -> None:
    tracer = RecordingTracer()
    pipeline = build(FakeRetriever([]), FakeLLM("never used"), tracer)
    outcome = pipeline.ask("What were Apple's net sales?", k=5)
    assert outcome.refused
    assert "generation" not in tracer.names()


def test_a_model_failure_reaches_the_caller_even_when_traced(hit) -> None:
    """The bug this guards: a traced `ask` used to swallow the model's error and return None.

    `RecordingTracer` cannot catch it, because it has no try around its yield. Only the real
    `LangfuseTracer` shape does, so this test uses that against a fake client.
    """
    from quarterly_rag.errors import ModelServerError
    from quarterly_rag.observability.tracing import LangfuseTracer

    class MinimalClient:
        """Enough of the SDK for the tracer to think it is recording."""

        def start_as_current_observation(self, **kwargs):
            return nullcontext(SimpleNamespace(trace_id="t", update=lambda **_: None))

        def create_score(self, **kwargs) -> None:
            return None

        def flush(self) -> None:
            return None

    class BrokenLLM(FakeLLM):
        def chat(self, messages, *, temperature=0.0, max_tokens=1024):
            raise ModelServerError("the model server is down")

    pipeline = build(FakeRetriever([hit]), BrokenLLM("unused"), LangfuseTracer(MinimalClient()))
    with pytest.raises(ModelServerError, match="the model server is down"):
        pipeline.ask("What were Apple's net sales?", k=5)
