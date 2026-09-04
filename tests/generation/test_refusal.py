from __future__ import annotations

import pytest

from quarterly_rag.generation.answer import verify
from quarterly_rag.generation.refusal import (
    CorpusScope,
    GateSettings,
    check_answer,
    check_retrieval,
    check_scope,
)
from quarterly_rag.retrieval.base import RetrievedChunk

SCOPE = CorpusScope(
    tickers=frozenset({"AAPL", "NVDA"}),
    company_words=frozenset({"apple", "nvidia"}),
    fiscal_years=frozenset({2024, 2025, 2026, 2027}),
)
OPEN_GATE = GateSettings(min_retrieval_score=0.0)


def results(make_chunk, *scores: float) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=make_chunk(f"c:{i}-{i + 1}", "text"), score=s, retriever="dense", rank=i
        )
        for i, s in enumerate(scores, start=1)
    ]


def test_a_company_outside_the_corpus_is_refused_before_retrieval() -> None:
    refusal = check_scope("What was Tesla's total revenue in fiscal 2025?", SCOPE, OPEN_GATE)
    assert refusal is not None
    assert refusal.reason == "out_of_scope"
    assert "Tesla" in refusal.detail
    assert "AAPL, NVDA" in refusal.detail


def test_a_company_inside_the_corpus_passes() -> None:
    assert check_scope("What were Apple's total net sales?", SCOPE, OPEN_GATE) is None
    assert check_scope("How much did Nvidia spend on R and D?", SCOPE, OPEN_GATE) is None


def test_a_comparison_naming_both_inside_and_outside_is_let_through() -> None:
    # Half the question is answerable, so refusing before retrieval would be premature;
    # the generator gets to say what it can support.
    assert check_scope("How does AMD's revenue compare with Nvidia's?", SCOPE, OPEN_GATE) is None


def test_a_year_before_the_corpus_is_refused() -> None:
    refusal = check_scope("What were Apple's total net sales in fiscal 2019?", SCOPE, OPEN_GATE)
    assert refusal is not None
    assert refusal.reason == "out_of_scope"
    assert "2024 to 2027" in refusal.detail


def test_a_year_inside_the_corpus_passes() -> None:
    assert check_scope("Apple's net sales in fiscal 2025?", SCOPE, OPEN_GATE) is None
    # A future year is not certainly absent: a filing may be about it.
    assert check_scope("Nvidia's revenue in fiscal 2027?", SCOPE, OPEN_GATE) is None


@pytest.mark.parametrize(
    "question",
    [
        "Should I buy Nvidia shares?",
        "Is Nvidia stock a good investment right now?",
        "What is Nvidia's current share price?",
        "What is Apple's market capitalisation?",
        "What did Nvidia's chief executive say on the earnings call?",
        "What is the weather in Santa Clara today?",
    ],
)
def test_questions_filings_never_answer_are_refused(question: str) -> None:
    refusal = check_scope(question, SCOPE, OPEN_GATE)
    assert refusal is not None and refusal.reason == "out_of_scope"


def test_topic_and_scope_checks_can_be_switched_off() -> None:
    off = GateSettings(check_scope=False, check_topics=False)
    assert check_scope("Should I buy Tesla shares?", SCOPE, off) is None


def test_low_confidence_fires_only_above_the_threshold(make_chunk) -> None:
    hits = results(make_chunk, 0.81, 0.79)
    assert check_retrieval(hits, OPEN_GATE) is None
    assert check_retrieval(hits, GateSettings(min_retrieval_score=0.80)) is None  # best clears it
    refusal = check_retrieval(hits, GateSettings(min_retrieval_score=0.90))
    assert refusal is not None
    assert refusal.reason == "low_confidence"
    assert "0.810" in refusal.detail
    assert len(refusal.best_chunks) == 2  # a refusal still shows its work


def test_empty_retrieval_is_low_confidence() -> None:
    refusal = check_retrieval([], OPEN_GATE)
    assert refusal is not None and refusal.reason == "low_confidence"


def test_the_generators_own_signal_becomes_a_reason(make_chunk) -> None:
    hits = results(make_chunk, 0.8)
    answer = verify("INSUFFICIENT_EVIDENCE", [h.chunk for h in hits])
    refusal = check_answer(answer, hits)
    assert refusal is not None and refusal.reason == "insufficient_evidence"
    assert refusal.best_chunks


def test_an_answer_with_no_resolvable_citation_fails_verification(make_chunk) -> None:
    hits = results(make_chunk, 0.8)
    answer = verify("Net sales were $109,417 million [c9].", [h.chunk for h in hits])
    refusal = check_answer(answer, hits)
    assert refusal is not None and refusal.reason == "verification_failed"


def test_a_verified_answer_is_not_refused(make_chunk) -> None:
    chunk = make_chunk("a:1-2", "(In millions)\nTotal net sales | 109,417")
    hits = [RetrievedChunk(chunk=chunk, score=0.9, retriever="dense", rank=1)]
    answer = verify("Net sales were $109,417 million [c1].", [chunk])
    assert check_answer(answer, hits) is None


def test_scope_is_derived_from_the_corpus(make_chunk) -> None:
    scope = CorpusScope.from_chunks(
        [
            make_chunk("a:1-2", "x", ticker="AAPL", company="Apple Inc.", fiscal_year=2025),
            make_chunk("b:1-2", "x", ticker="NVDA", company="NVIDIA CORP", fiscal_year=2027),
        ]
    )
    assert scope.tickers == {"AAPL", "NVDA"}
    assert scope.company_words == {"apple", "nvidia"}  # Inc and CORP are dropped
    assert scope.year_range == (2025, 2027)


def test_a_calculation_citing_a_passage_does_not_satisfy_the_answer_gate(make_chunk) -> None:
    """A verified calculation is not a grounded claim: the sentence still has to cite."""
    chunk = make_chunk("a:1-2", "(In millions)\nTotal net sales | 109,417 | 94,036")
    answer = verify("Net sales rose a lot.\nCALC: 109,417 [c1] - 94,036 [c1] = 15,381", [chunk])
    refusal = check_answer(answer, [])
    assert refusal is not None
    assert refusal.reason == "verification_failed"
