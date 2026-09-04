from __future__ import annotations

import math

import pytest

from quarterly_rag.evaluation.metrics import (
    QuestionResult,
    evaluate_question,
    group_by,
    mean_reciprocal_rank,
    ndcg_at_k,
    near_miss_rates,
    recall_at_k,
    summarise,
)
from quarterly_rag.retrieval.base import RetrievedChunk

from .test_relevance import question


def result(rank: int | None, *, qid="q1", relevant_in_corpus=1, **overrides) -> QuestionResult:
    fields = {
        "question_id": qid,
        "ticker": "AAPL",
        "form": "10-Q",
        "question_type": "lookup",
        "section": "Part I.Item 1",
        "first_relevant_rank": rank,
        "relevant_retrieved": 1 if rank else 0,
        "relevant_in_corpus": relevant_in_corpus,
        "retrieved": 10,
    }
    return QuestionResult(**(fields | overrides))


def test_recall_counts_a_question_once_and_rises_with_k() -> None:
    results = [result(1), result(4), result(None)]
    assert recall_at_k(results, 1) == pytest.approx(1 / 3)
    assert recall_at_k(results, 3) == pytest.approx(1 / 3)
    assert recall_at_k(results, 5) == pytest.approx(2 / 3)
    assert recall_at_k(results, 10) == pytest.approx(2 / 3)
    assert recall_at_k([], 5) == 0.0


def test_mrr_is_the_mean_of_one_over_the_first_relevant_rank() -> None:
    assert mean_reciprocal_rank([result(1), result(2), result(None)]) == pytest.approx(
        (1.0 + 0.5 + 0.0) / 3
    )
    assert mean_reciprocal_rank([]) == 0.0


def test_ndcg_rewards_a_higher_rank() -> None:
    high = ndcg_at_k([result(1)], 5, {"q1": [1]})
    low = ndcg_at_k([result(3)], 5, {"q1": [3]})
    assert high == pytest.approx(1.0)
    assert 0 < low < high


def test_ndcg_cannot_reach_one_when_the_evidence_needs_two_chunks() -> None:
    """A gold span that straddles two chunks has two relevant chunks in the corpus, so
    finding one is a partial answer and the metric says so."""
    one_of_two = ndcg_at_k([result(1, relevant_in_corpus=2)], 5, {"q1": [1]})
    both = ndcg_at_k([result(1, relevant_in_corpus=2)], 5, {"q1": [1, 2]})
    assert one_of_two < 1.0
    assert both == pytest.approx(1.0)
    assert one_of_two == pytest.approx(1.0 / (1.0 + 1 / math.log2(3)))


def test_ndcg_is_zero_when_nothing_relevant_was_retrieved() -> None:
    assert ndcg_at_k([result(None)], 5, {"q1": []}) == 0.0


def test_near_miss_ladder_is_monotone() -> None:
    results = [
        result(1, filing_rank=1, section_rank=1),
        result(None, qid="q2", filing_rank=2, section_rank=3),
        result(None, qid="q3", filing_rank=4),
        result(None, qid="q4"),
    ]
    rates = near_miss_rates(results, 5)
    assert rates["filing"] == pytest.approx(0.75)
    assert rates["section"] == pytest.approx(0.5)
    assert rates["chunk"] == pytest.approx(0.25)
    assert rates["filing"] >= rates["section"] >= rates["chunk"]


def test_near_miss_respects_the_cutoff() -> None:
    rates = near_miss_rates([result(None, filing_rank=9, section_rank=9)], 5)
    assert rates == {"filing": 0.0, "section": 0.0, "chunk": 0.0}


def test_grouping_partitions_the_questions() -> None:
    results = [
        result(1, qid="a", question_type="lookup"),
        result(None, qid="b", question_type="lookup"),
        result(2, qid="c", question_type="derived"),
    ]
    ranks = {"a": [1], "b": [], "c": [2]}
    grouped = group_by(results, "question_type", ranks, ks=(5,))
    assert sum(m.count for m in grouped.values()) == len(results)
    assert grouped["lookup"].recall[5] == pytest.approx(0.5)
    assert grouped["derived"].recall[5] == pytest.approx(1.0)


def test_summarise_reports_every_cutoff() -> None:
    metrics = summarise([result(2)], {"q1": [2]}, ks=(1, 3))
    assert metrics.count == 1
    assert metrics.recall == {1: 0.0, 3: 1.0}
    assert set(metrics.as_dict()["recall"]) == {"@1", "@3"}


def test_evaluate_question_finds_the_first_relevant_rank(make_chunk) -> None:
    q = question((100, 200))
    retrieved = [
        RetrievedChunk(
            chunk=make_chunk(char_start=900, char_end=1000), score=0.9, retriever="dense", rank=1
        ),
        RetrievedChunk(
            chunk=make_chunk(char_start=150, char_end=250), score=0.8, retriever="dense", rank=2
        ),
        RetrievedChunk(
            chunk=make_chunk(char_start=180, char_end=260), score=0.7, retriever="dense", rank=3
        ),
    ]
    scored = evaluate_question(q, retrieved, relevant_total=2, form="10-Q")
    assert scored.first_relevant_rank == 2
    assert scored.relevant_retrieved == 2
    assert scored.section_rank == 1  # same filing and section, just the wrong offsets
    assert scored.filing_rank == 1
    assert scored.form == "10-Q"
    assert scored.section == "Part I.Item 1"
