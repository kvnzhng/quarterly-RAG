"""The rates a generation report publishes (RAG-010, RAG-021).

These numbers end up in `docs/`, so the arithmetic behind them is tested without a model.
"""

from __future__ import annotations

import pytest

from quarterly_rag.evaluation.generation_eval import (
    GOLD,
    AnswerResult,
    GenerationReport,
    _verified_sentences,
)
from quarterly_rag.evaluation.retrieval_eval import RunRecord
from quarterly_rag.generation.answer import verify

RUN = RunRecord(
    timestamp="2026-09-04T00:00:00+00:00",
    git_commit="abc1234",
    git_dirty=False,
    corpus_hash="c",
    eval_set_hash="e",
    parser_version="1",
    chunk_strategy="section-aware",
    chunk_words=350,
    chunk_overlap_words=60,
    embed_variant="context",
    embedder="fake/embed",
    embed_query_prefix="",
    embed_document_prefix="",
    vector_store="chroma",
    indexed_chunks=10,
    retriever="gold-chunks",
    k_values=[5],
    overlap_rule="any overlap (min_chars=1)",
    filters=None,
    question_count=2,
)


def result(
    question_id: str,
    *,
    derived: int = 0,
    derived_verified: int = 0,
    calculations: int = 0,
    calculations_verified: int = 0,
    reasons: list[str] | None = None,
    refused: bool = False,
) -> AnswerResult:
    return AnswerResult(
        question_id=question_id,
        question_type="derived",
        ticker="AAPL",
        passages=5,
        insufficient_evidence=refused,
        citations=1,
        invalid_tags=0,
        unsupported_sentences=0,
        derived_numbers=derived,
        derived_verified=derived_verified,
        calculations=calculations,
        calculations_verified=calculations_verified,
        calculation_reasons=reasons or [],
        truncated=False,
        fully_grounded=derived == derived_verified,
        gold_answer_figures_present=True,
        answer="an answer",
    )


def report(*results: AnswerResult) -> GenerationReport:
    return GenerationReport(run=RUN, context=GOLD, results=list(results))


def test_a_question_counts_as_accounted_when_every_figure_recomputes() -> None:
    rates = report(
        result("q007", derived=1, derived_verified=1, calculations=1, calculations_verified=1),
        result("q008", derived=2, derived_verified=1, calculations=2, calculations_verified=1),
    ).rates()
    assert rates["figures_accounted"] == pytest.approx(0.5)
    assert rates["derived_verified"] == pytest.approx(2 / 3)  # 2 of 3 derived figures
    assert rates["calculations_verified"] == pytest.approx(2 / 3)  # 2 of 3 CALC lines


def test_the_presence_only_rate_keeps_its_old_meaning() -> None:
    """`figures_verified` still counts a recomputed figure as not present, because it is not.

    Changing what it means would make every number recorded before RAG-021 incomparable.
    """
    rates = report(
        result("q007", derived=1, derived_verified=1, calculations=1, calculations_verified=1)
    ).rates()
    assert rates["figures_verified"] == 0.0
    assert rates["figures_accounted"] == 1.0
    assert rates["fully_grounded"] == 1.0


def test_an_answer_set_with_no_arithmetic_reports_no_arithmetic_rate() -> None:
    """Absent, not zero: a set that derived nothing did not fail at deriving."""
    rates = report(result("q001")).rates()
    assert "derived_verified" not in rates
    assert "calculations_verified" not in rates
    assert rates["figures_accounted"] == 1.0


def test_refused_answers_are_excluded_from_the_arithmetic_rates() -> None:
    rates = report(
        result("q007", refused=True),
        result("q008", derived=1, derived_verified=1),
    ).rates()
    assert rates["insufficient_evidence"] == pytest.approx(0.5)
    assert rates["figures_accounted"] == 1.0


def test_failure_reasons_are_counted_so_one_check_cannot_hide_behind_another() -> None:
    failures = report(
        result("q007", reasons=["unparsed", "an operand with no citation"]),
        result("q008", reasons=["unparsed", ""]),
    ).calculation_failures()
    assert failures == {"unparsed": 2, "an operand with no citation": 1}


def test_the_judge_is_calibrated_against_prose_only(make_chunk) -> None:
    """A CALC line is not a sentence, so it must not enter the calibration set."""
    passages = [make_chunk("a:1-2", "(In millions)\nTotal net sales | 109,417 | 94,036")]
    answer = verify(
        "Net sales rose $15,381 million [c1].\nCALC: 109,417 [c1] - 94,036 [c1] = 15,381",
        passages,
    )
    assert _verified_sentences(answer) == {"Net sales rose $15,381 million [c1]."}
