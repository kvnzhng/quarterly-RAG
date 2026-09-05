"""Filtering and per-company retrieval (RAG-009, RAG-031).

A fake inner retriever records what it was asked for, so what is tested is the decision this
wrapper makes rather than any ranking.
"""

from __future__ import annotations

import pytest

from quarterly_rag.retrieval.base import RetrievedChunk
from quarterly_rag.retrieval.filtered import FilteredRetriever, interleave


class FakeInner:
    """Returns chunks tagged with whichever ticker was asked for, and remembers the asks."""

    name = "hybrid"

    def __init__(
        self,
        by_ticker: dict[str, list[RetrievedChunk]] | None = None,
        unfiltered: list[RetrievedChunk] | None = None,
    ) -> None:
        self.by_ticker = by_ticker or {}
        self.unfiltered = unfiltered or []
        self.calls: list[tuple[str, int, dict | None]] = []

    def retrieve(self, question: str, k: int = 5, where: dict | None = None):
        self.calls.append((question, k, where))
        ticker = (where or {}).get("ticker")
        # No silent fallback here: the wrapper's own fallback is what the tests check.
        results = self.by_ticker.get(ticker, []) if ticker else self.unfiltered
        return results[:k]


def hits(make_chunk, ticker: str, count: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=make_chunk(f"{ticker}:{i}", f"passage {i}", ticker=ticker),
            score=1.0 - i / 100,
            rank=i,
            retriever="hybrid",
        )
        for i in range(1, count + 1)
    ]


@pytest.fixture
def two_companies(make_chunk):
    return FakeInner(
        {"AAPL": hits(make_chunk, "AAPL", 6), "NVDA": hits(make_chunk, "NVDA", 6)},
        unfiltered=hits(make_chunk, "AAPL", 6),
    )


def test_one_company_named_becomes_a_filter(two_companies) -> None:
    FilteredRetriever(two_companies).retrieve("What were Apple's net sales in fiscal 2025?", k=5)
    (_, _, where) = two_companies.calls[0]
    assert where == {"ticker": "AAPL"}


def test_two_companies_named_are_asked_separately(two_companies) -> None:
    """One ranked list cannot represent two companies whose filings use different words."""
    results = FilteredRetriever(two_companies).retrieve(
        "Who made more revenue in 2025, Nvidia or Apple?", k=6
    )
    asked = [where.get("ticker") for _, _, where in two_companies.calls]
    assert sorted(asked) == ["AAPL", "NVDA"]
    assert [r.chunk.ticker for r in results] == ["AAPL", "NVDA", "AAPL", "NVDA", "AAPL", "NVDA"]


def test_each_company_reaches_the_answer_even_at_an_odd_k(two_companies) -> None:
    results = FilteredRetriever(two_companies).retrieve("Apple or Nvidia in 2025?", k=5)
    tickers = [r.chunk.ticker for r in results]
    assert tickers.count("AAPL") == 3
    assert tickers.count("NVDA") == 2


def test_ranks_describe_the_merged_list(two_companies) -> None:
    results = FilteredRetriever(two_companies).retrieve("Apple or Nvidia?", k=4)
    assert [r.rank for r in results] == [1, 2, 3, 4]


def test_an_explicit_caller_filter_wins_over_the_split(two_companies) -> None:
    """`rag ask --ticker AAPL` means one company, whatever the question mentions."""
    results = FilteredRetriever(two_companies).retrieve(
        "Who made more revenue, Nvidia or Apple?", k=4, where={"ticker": "AAPL"}
    )
    assert {r.chunk.ticker for r in results} == {"AAPL"}
    assert len(two_companies.calls) == 1


def test_a_company_with_nothing_to_say_yields_its_slots(make_chunk) -> None:
    inner = FakeInner({"AAPL": hits(make_chunk, "AAPL", 6), "NVDA": []})
    results = FilteredRetriever(inner).retrieve("Apple or Nvidia?", k=4)
    assert [r.chunk.ticker for r in results] == ["AAPL"] * 4


def test_nothing_from_either_company_falls_back_to_no_filter(make_chunk) -> None:
    inner = FakeInner({}, unfiltered=hits(make_chunk, "AAPL", 3))
    results = FilteredRetriever(inner).retrieve("Apple or Nvidia?", k=3)
    assert len(results) == 3
    assert two_asks_then_the_fallback(inner)


def two_asks_then_the_fallback(inner: FakeInner) -> bool:
    return [w.get("ticker") if w else None for _, _, w in inner.calls] == ["AAPL", "NVDA", None]


def test_disabling_the_wrapper_disables_the_split(two_companies) -> None:
    FilteredRetriever(two_companies, enabled=False).retrieve("Apple or Nvidia?", k=4)
    assert two_companies.calls == [("Apple or Nvidia?", 4, None)]


def test_interleave_takes_the_best_of_each_before_the_second_of_any(make_chunk) -> None:
    apple, nvidia = hits(make_chunk, "AAPL", 3), hits(make_chunk, "NVDA", 3)
    merged = interleave([apple, nvidia], 4)
    assert [(r.chunk.ticker, r.chunk.chunk_id) for r in merged] == [
        ("AAPL", "AAPL:1"),
        ("NVDA", "NVDA:1"),
        ("AAPL", "AAPL:2"),
        ("NVDA", "NVDA:2"),
    ]


def test_interleave_does_not_repeat_a_chunk_two_companies_both_returned(make_chunk) -> None:
    shared = hits(make_chunk, "AAPL", 2)
    merged = interleave([shared, shared], 4)
    assert len(merged) == 2
