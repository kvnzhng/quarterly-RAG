from __future__ import annotations

import pytest

from quarterly_rag.retrieval.base import RetrievedChunk
from quarterly_rag.retrieval.bm25 import BM25Retriever
from quarterly_rag.retrieval.filtered import FilteredRetriever
from quarterly_rag.retrieval.hybrid import HybridRetriever
from quarterly_rag.retrieval.rerank import LLMReranker


class Scripted:
    """Returns a fixed ranking, so fusion arithmetic is checked against known input."""

    def __init__(self, label: str, chunks) -> None:
        self.label = label
        self.chunks = chunks
        self.calls: list[tuple[int, dict | None]] = []

    @property
    def name(self) -> str:
        return self.label

    def retrieve(self, question, k=5, where=None):
        self.calls.append((k, where))
        return [
            RetrievedChunk(chunk=c, score=1.0 - i * 0.1, retriever=self.label, rank=i + 1)
            for i, c in enumerate(self.chunks[:k])
        ]


@pytest.fixture
def chunks(make_chunk):
    return [make_chunk(f"c{i}:1-2", f"passage {i}") for i in range(4)]


def test_appearing_in_both_rankings_beats_appearing_in_one(chunks) -> None:
    # A ranks c0 first and c1 second; B only knows c1. c1 collects two contributions.
    a = Scripted("a", [chunks[0], chunks[1]])
    b = Scripted("b", [chunks[1]])
    fused = HybridRetriever([a, b], pool=2, fusion_k=60).retrieve("q", k=2)

    assert [f.chunk.chunk_id for f in fused] == ["c1:1-2", "c0:1-2"]
    assert [f.rank for f in fused] == [1, 2]
    assert all(f.retriever == "hybrid" for f in fused)
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)


def test_a_higher_rank_still_wins_when_both_lists_agree(chunks) -> None:
    a = Scripted("a", [chunks[0], chunks[1]])
    b = Scripted("b", [chunks[0], chunks[1]])
    fused = HybridRetriever([a, b], pool=2).retrieve("q", k=2)
    assert [f.chunk.chunk_id for f in fused] == ["c0:1-2", "c1:1-2"]


def test_each_retriever_is_asked_for_the_pool_not_the_final_k(chunks) -> None:
    a = Scripted("a", chunks)
    HybridRetriever([a], pool=4, fusion_k=60).retrieve("q", k=2)
    assert a.calls == [(4, None)]


def test_contributions_are_reported(chunks) -> None:
    a = Scripted("a", chunks[:2])
    b = Scripted("b", chunks[2:])
    hybrid = HybridRetriever([a, b], pool=2)
    hybrid.retrieve("q", k=4)
    assert hybrid.contributions == {"a": 2, "b": 2}


def test_a_single_retriever_fuses_to_its_own_order(chunks) -> None:
    a = Scripted("a", chunks)
    fused = HybridRetriever([a], pool=4).retrieve("q", k=4)
    assert [f.chunk.chunk_id for f in fused] == [c.chunk_id for c in chunks]


@pytest.fixture
def two_company_corpus(make_chunk):
    """Five documents, because BM25's IDF is zero on a corpus of two."""
    apple = [
        make_chunk("a1:1-2", "Total net sales | 109,417 | 94,036"),
        make_chunk("a2:1-2", "Americas net sales increased during the quarter"),
        make_chunk("a3:1-2", "Research and development | 34,550"),
    ]
    nvidia = [
        make_chunk("n1:1-2", "Revenue | 96,221 | 46,743", ticker="NVDA", company="NVIDIA CORP"),
        make_chunk("n2:1-2", "Data Center revenue grew", ticker="NVDA", company="NVIDIA CORP"),
    ]
    return apple + nvidia


def test_the_filter_is_inferred_from_the_question(two_company_corpus) -> None:
    filtered = FilteredRetriever(BM25Retriever(two_company_corpus))
    assert filtered.name == "bm25+filter"
    results = filtered.retrieve("What was Nvidia's revenue?", k=5)
    assert {r.chunk.ticker for r in results} == {"NVDA"}
    assert all(r.retriever == "bm25+filter" for r in results)


def test_an_explicit_filter_wins_over_the_inferred_one(two_company_corpus) -> None:
    filtered = FilteredRetriever(BM25Retriever(two_company_corpus))
    # The question says Nvidia; the caller says Apple, and the caller is obeyed.
    results = filtered.retrieve("Nvidia net sales", k=5, where={"ticker": "AAPL"})
    assert results
    assert {r.chunk.ticker for r in results} == {"AAPL"}


def test_an_empty_filtered_result_falls_back_to_no_filter(two_company_corpus) -> None:
    # A filter that removes everything is worse than no filter: the refusal gate would
    # report low confidence for a question the corpus can answer.
    filtered = FilteredRetriever(BM25Retriever(two_company_corpus))
    results = filtered.retrieve("Nvidia net sales in the first quarter of fiscal 1999", k=5)
    assert results  # nothing matches that period, so the unfiltered ranking is used


def test_the_fallback_can_be_switched_off(two_company_corpus) -> None:
    strict = FilteredRetriever(BM25Retriever(two_company_corpus), fall_back=False)
    assert strict.retrieve("Nvidia net sales in the first quarter of fiscal 1999", k=5) == []


def test_filtering_can_be_disabled(two_company_corpus) -> None:
    off = FilteredRetriever(BM25Retriever(two_company_corpus), enabled=False)
    assert off.name == "bm25"
    # Without the inferred filter, Apple passages can come back for an Nvidia question.
    assert {r.chunk.ticker for r in off.retrieve("net sales revenue", k=5)} == {"AAPL", "NVDA"}


class CountingLLM:
    """Scores a passage by the digit its text was seeded with."""

    label = "fake/judge"

    def __init__(self, scores: dict[str, str]) -> None:
        self.scores = scores
        self.calls = 0

    def chat(self, messages, *, temperature=0.0, max_tokens=1024):
        from quarterly_rag.generation.base import ChatResponse

        self.calls += 1
        passage = messages[-1].content
        for key, reply in self.scores.items():
            if key in passage:
                return ChatResponse(text=reply, model="fake", stop_reason="stop")
        return ChatResponse(text="0", model="fake", stop_reason="stop")

    def list_models(self):
        return ["fake"]


def test_the_reranker_reorders_by_judged_relevance(chunks) -> None:
    inner = Scripted("hybrid", chunks)
    llm = CountingLLM({"passage 3": "10", "passage 0": "2", "passage 1": "7"})
    reranked = LLMReranker(inner, llm, pool=4).retrieve("q", k=3)

    assert [r.chunk.chunk_id for r in reranked] == ["c3:1-2", "c1:1-2", "c0:1-2"]
    assert reranked[0].score == pytest.approx(1.0)
    assert reranked[0].retriever == "hybrid+rerank"
    assert llm.calls == 4  # one model call per candidate, which is the cost


def test_an_unparseable_or_failed_judgement_scores_zero(chunks) -> None:
    from quarterly_rag.errors import ModelServerError

    class Broken(CountingLLM):
        def chat(self, messages, *, temperature=0.0, max_tokens=1024):
            raise ModelServerError("endpoint down")

    reranked = LLMReranker(Scripted("hybrid", chunks), Broken({}), pool=2).retrieve("q", k=2)
    # Everything ties at zero, so the inner order survives rather than being scrambled.
    assert [r.chunk.chunk_id for r in reranked] == ["c0:1-2", "c1:1-2"]


def test_the_reranker_short_circuits_on_no_candidates() -> None:
    llm = CountingLLM({})
    assert LLMReranker(Scripted("hybrid", []), llm).retrieve("q") == []
    assert llm.calls == 0
