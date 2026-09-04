from __future__ import annotations

import math
from collections.abc import Sequence

from quarterly_rag.chunking.base import Chunk
from quarterly_rag.indexing.base import SearchHit
from quarterly_rag.retrieval.base import Retriever
from quarterly_rag.retrieval.dense import DenseRetriever


class FakeEmbedder:
    """Records what it was asked to embed, so the query/document split is testable."""

    label = "fake/embedder"

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.documents: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.documents.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]

    def list_models(self) -> list[str]:
        return ["fake"]


class FakeStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[int, dict | None]] = []

    @property
    def name(self) -> str:
        return "fake"

    def add(self, chunks, vectors) -> None: ...

    def query(self, vector, k=5, where=None) -> list[SearchHit]:
        self.calls.append((k, where))
        return [SearchHit(chunk=c, score=1.0 - i * 0.1) for i, c in enumerate(self.chunks[:k])]

    def count(self) -> int:
        return len(self.chunks)

    def persist(self) -> None: ...


def test_ranks_results_and_keeps_provenance(make_chunk) -> None:
    chunks = [make_chunk(f"c:{i}-{i + 1}", f"passage {i}") for i in range(3)]
    embedder = FakeEmbedder()
    retriever = DenseRetriever(embedder, FakeStore(chunks))
    assert isinstance(retriever, Retriever)
    assert retriever.name == "dense"

    results = retriever.retrieve("What were Apple's total net sales?", k=3)

    assert [r.rank for r in results] == [1, 2, 3]
    assert [r.retriever for r in results] == ["dense"] * 3
    assert results[0].score > results[-1].score
    assert results[0].chunk.ticker == "AAPL"
    assert results[0].chunk.section == "Part I.Item 1"
    assert results[0].chunk.source_url.endswith("aapl.htm")


def test_the_question_is_embedded_as_a_query_not_a_document(make_chunk) -> None:
    embedder = FakeEmbedder()
    DenseRetriever(embedder, FakeStore([make_chunk("a:1-2", "x")])).retrieve("a question")
    assert embedder.queries == ["a question"]
    assert embedder.documents == []


def test_filters_are_passed_through_and_blank_questions_short_circuit(make_chunk) -> None:
    store = FakeStore([make_chunk("a:1-2", "x")])
    retriever = DenseRetriever(FakeEmbedder(), store)

    retriever.retrieve("net sales", k=7, where={"ticker": "NVDA"})
    assert store.calls == [(7, {"ticker": "NVDA"})]

    assert retriever.retrieve("   ") == []
    assert len(store.calls) == 1  # no request made


def test_score_is_a_similarity_not_a_distance(make_chunk) -> None:
    chunks = [make_chunk("a:1-2", "x")]
    (result,) = DenseRetriever(FakeEmbedder(), FakeStore(chunks)).retrieve("q", k=1)
    assert math.isclose(result.score, 1.0)
