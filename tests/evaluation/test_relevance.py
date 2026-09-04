from __future__ import annotations

from quarterly_rag.evaluation.questions import EvalQuestion, EvidenceSpan
from quarterly_rag.evaluation.relevance import (
    OverlapRule,
    is_relevant,
    overlap_chars,
    relevant_in_corpus,
)

ACCESSION = "0000320193-26-000020"


def question(*spans: tuple[int, int], accession: str = ACCESSION, ticker: str = "AAPL"):
    return EvalQuestion(
        id="q001",
        question="What were Apple's total net sales?",
        ticker=ticker,
        type="lookup",
        gold_answer="$109,417 million",
        evidence=[
            EvidenceSpan(
                accession=accession,
                section="Part I.Item 1",
                char_start=start,
                char_end=end,
                quote="x" * (end - start),
            )
            for start, end in spans
        ],
    )


def test_overlap_is_measured_in_characters(make_chunk) -> None:
    span = question((100, 200)).evidence[0]
    assert overlap_chars(make_chunk(char_start=150, char_end=250), span) == 50
    assert overlap_chars(make_chunk(char_start=0, char_end=100), span) == 0  # touching, not over
    assert overlap_chars(make_chunk(char_start=0, char_end=101), span) == 1
    assert overlap_chars(make_chunk(char_start=120, char_end=130), span) == 10  # contained


def test_a_different_filing_never_overlaps(make_chunk) -> None:
    span = question((100, 200)).evidence[0]
    other = make_chunk(accession="0001045810-26-000021", char_start=100, char_end=200)
    assert overlap_chars(other, span) == 0


def test_any_overlap_counts_by_default(make_chunk) -> None:
    q = question((100, 200))
    assert is_relevant(make_chunk(char_start=199, char_end=400), q)
    assert not is_relevant(make_chunk(char_start=200, char_end=400), q)


def test_a_stricter_rule_rejects_a_clipped_chunk(make_chunk) -> None:
    q = question((100, 200))
    clipped = make_chunk(char_start=199, char_end=400)  # covers one character
    assert is_relevant(clipped, q)
    assert not is_relevant(clipped, q, OverlapRule(min_chars=10))
    assert not is_relevant(clipped, q, OverlapRule(min_fraction=0.5))
    assert is_relevant(make_chunk(char_start=100, char_end=200), q, OverlapRule(min_fraction=1.0))


def test_a_chunk_covering_any_one_span_is_relevant(make_chunk) -> None:
    q = question((100, 200), (500, 600))
    assert is_relevant(make_chunk(char_start=550, char_end=700), q)
    assert not is_relevant(make_chunk(char_start=300, char_end=400), q)


def test_the_wrong_company_is_never_relevant(make_chunk) -> None:
    q = question((100, 200), ticker="NVDA")
    assert not is_relevant(make_chunk(char_start=100, char_end=200), q)


def test_counting_relevant_chunks_in_the_corpus(make_chunk) -> None:
    q = question((100, 300))
    corpus = [
        make_chunk(char_start=0, char_end=150),
        make_chunk(char_start=150, char_end=280),
        make_chunk(char_start=280, char_end=400),
        make_chunk(char_start=400, char_end=500),
    ]
    assert relevant_in_corpus(corpus, q) == 3


def test_the_rule_describes_itself() -> None:
    assert OverlapRule().describe() == "any overlap (min_chars=1)"
    assert "min_fraction=0.5" in OverlapRule(min_chars=5, min_fraction=0.5).describe()
