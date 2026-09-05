"""`POST /ask` against a fake pipeline (RAG-014).

No model, no index, no network. The pipeline is injected, so what is tested here is the
contract: what the endpoint returns for an answer, for a refusal, and when the model server
is down.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from quarterly_rag.api.app import create_app
from quarterly_rag.api.models import MAX_EXCERPT_CHARS, MAX_PASSAGE_CHARS
from quarterly_rag.config import Settings
from quarterly_rag.errors import ModelServerError
from quarterly_rag.generation.answer import verify
from quarterly_rag.generation.refusal import GateOutcome, Refusal
from quarterly_rag.observability.tracing import NullTracer
from quarterly_rag.retrieval.base import RetrievedChunk

TABLE = "(In millions)\nTotal net sales | 109,417 | 94,036\nServices | 27,423 | 24,213"


class FakePipeline:
    """Enough of a pipeline for the endpoint: an outcome, a label, and a tracer."""

    prompt_version = "2"

    def __init__(self, outcome, *, raises: Exception | None = None) -> None:
        self.outcome = outcome
        self.raises = raises
        self.tracer = NullTracer()
        self.llm = type("L", (), {"label": "fake/llm"})()
        self.calls: list[tuple] = []

    def ask(self, question: str, k: int = 5, where: dict | None = None):
        self.calls.append((question, k, where))
        if self.raises:
            raise self.raises
        return self.outcome


@pytest.fixture
def hit(make_chunk):
    return RetrievedChunk(chunk=make_chunk("a:1-2", TABLE), score=0.91, rank=1, retriever="hybrid")


def client(pipeline) -> TestClient:
    app = create_app(Settings(_env_file=None), pipeline_factory=lambda s: pipeline)
    return TestClient(app)


def test_an_answered_question_returns_its_citations_and_checks(hit) -> None:
    answer = verify("Total net sales were $109,417 million [c1].", [hit.chunk], model="fake/llm")
    outcome = GateOutcome(answer=answer, results=[hit], trace_id="trace-1")
    with client(FakePipeline(outcome)) as http:
        response = http.post("/ask", json={"question": "What were net sales?", "k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["refusal"] is None
    assert body["trace_id"] == "trace-1"
    assert body["answer"]["fully_grounded"] is True
    (citation,) = body["answer"]["citations"]
    assert citation["tag"] == "c1"
    assert citation["ticker"] == "AAPL"
    assert citation["source_url"].startswith("https://www.sec.gov/")


def test_the_citation_carries_the_whole_passage_not_the_short_quote(hit) -> None:
    """`Citation.quote` is 200 characters, which is not enough to see the table."""
    answer = verify("Total net sales were $109,417 million [c1].", [hit.chunk], model="fake/llm")
    outcome = GateOutcome(answer=answer, results=[hit])
    with client(FakePipeline(outcome)) as http:
        (citation,) = http.post("/ask", json={"question": "q"}).json()["answer"]["citations"]
    assert citation["text"] == TABLE
    assert "Services | 27,423" in citation["text"]


def test_a_long_passage_is_capped(make_chunk) -> None:
    long_hit = RetrievedChunk(
        chunk=make_chunk("a:1-2", "109,417 " + "x" * 5000), score=0.9, rank=1, retriever="hybrid"
    )
    answer = verify("Net sales were 109,417 [c1].", [long_hit.chunk], model="fake/llm")
    with client(FakePipeline(GateOutcome(answer=answer, results=[long_hit]))) as http:
        (citation,) = http.post("/ask", json={"question": "q"}).json()["answer"]["citations"]
    assert len(citation["text"]) == MAX_PASSAGE_CHARS


def test_a_refusal_is_a_200_with_its_reason(hit) -> None:
    """Refusing is an answer to "can you answer this?", not a failed request."""
    refusal = Refusal(
        reason="out_of_scope",
        detail="Microsoft is not in the corpus, which holds AAPL, NVDA.",
        best_chunks=[hit],
    )
    with client(FakePipeline(GateOutcome(refusal=refusal, results=[hit]))) as http:
        response = http.post("/ask", json={"question": "What were Microsoft's net sales?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] is None
    assert body["refusal"]["reason"] == "out_of_scope"
    (passage,) = body["refusal"]["best_chunks"]
    assert passage["score"] == pytest.approx(0.91)
    assert len(passage["excerpt"]) <= MAX_EXCERPT_CHARS


def test_a_model_server_failure_is_a_502_carrying_the_reason(hit) -> None:
    pipeline = FakePipeline(None, raises=ModelServerError("the model server is down"))
    with client(pipeline) as http:
        response = http.post("/ask", json={"question": "What were net sales?"})
    assert response.status_code == 502
    assert "the model server is down" in response.json()["detail"]


def test_the_ticker_filter_reaches_the_pipeline_upper_cased(hit) -> None:
    answer = verify("Net sales were $109,417 million [c1].", [hit.chunk], model="fake/llm")
    pipeline = FakePipeline(GateOutcome(answer=answer, results=[hit]))
    with client(pipeline) as http:
        http.post("/ask", json={"question": "q", "k": 7, "ticker": "aapl"})
    assert pipeline.calls == [("q", 7, {"ticker": "AAPL"})]


def test_a_question_out_of_range_is_rejected_before_a_model_is_called() -> None:
    pipeline = FakePipeline(None)
    with client(pipeline) as http:
        assert http.post("/ask", json={"question": ""}).status_code == 422
        assert http.post("/ask", json={"question": "q", "k": 99}).status_code == 422
    assert pipeline.calls == []


def test_health_reports_the_model_and_never_the_endpoint(hit) -> None:
    pipeline = FakePipeline(GateOutcome(results=[hit]))
    with client(pipeline) as http:
        body = http.get("/health").json()
    assert body == {"status": "ok", "model": "fake/llm", "prompt_version": "2", "tracing": False}


def test_a_pipeline_that_could_not_be_built_answers_503_rather_than_refusing_to_start() -> None:
    """A missing index should not stop the server; it should say what is wrong."""

    def broken(settings):
        raise RuntimeError("the index is empty")

    with TestClient(create_app(Settings(_env_file=None), pipeline_factory=broken)) as http:
        assert http.get("/health").status_code == 503
        assert http.post("/ask", json={"question": "q"}).status_code == 503
