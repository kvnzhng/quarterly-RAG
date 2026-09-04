from __future__ import annotations

import pytest

from quarterly_rag.evaluation.refusal_eval import (
    AbstentionMetrics,
    RefusalResult,
    score,
    sweep_threshold,
)


def result(qid: str, *, should: bool, refused: bool, best: float | None = 0.8, reason=None):
    return RefusalResult(
        question_id=qid,
        question_type="unanswerable" if should else "lookup",
        should_refuse=should,
        refused=refused,
        reason=reason or ("insufficient_evidence" if refused else None),
        expected_reason="insufficient_evidence" if should else None,
        best_score=best,
    )


def test_precision_and_recall_are_about_refusals() -> None:
    results = [
        result("a", should=True, refused=True),  # correct refusal
        result("b", should=True, refused=False),  # a leak: answered the unanswerable
        result("c", should=False, refused=True),  # over-refusal: lost an answer
        result("d", should=False, refused=False),  # correctly answered
    ]
    metrics = score(results)
    assert metrics.total == 4
    assert metrics.should_refuse == 2
    assert metrics.refused == 2
    assert metrics.true_refusals == 1
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    # Two answerable questions, one wrongly refused.
    assert metrics.answerable_coverage == pytest.approx(0.5)


def test_refusing_everything_maximises_recall_and_destroys_coverage() -> None:
    results = [result(str(i), should=i < 2, refused=True) for i in range(4)]
    metrics = score(results)
    assert metrics.recall == 1.0
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.answerable_coverage == 0.0


def test_refusing_nothing_gives_no_precision_and_full_coverage() -> None:
    results = [result(str(i), should=i < 2, refused=False) for i in range(4)]
    metrics = score(results)
    assert metrics.refused == 0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.answerable_coverage == 1.0


def test_empty_input_does_not_divide_by_zero() -> None:
    metrics = AbstentionMetrics(refused=0, should_refuse=0, true_refusals=0, total=0)
    assert (metrics.precision, metrics.recall, metrics.f1) == (0.0, 0.0, 0.0)
    assert metrics.answerable_coverage == 0.0


def test_the_sweep_only_adds_refusals_as_the_threshold_rises() -> None:
    results = [
        result("answerable-high", should=False, refused=False, best=0.90),
        result("answerable-low", should=False, refused=False, best=0.70),
        result("unanswerable-low", should=True, refused=False, best=0.72),
    ]
    rows = sweep_threshold(results, (0.0, 0.75, 0.95))
    assert [r["refused"] for r in rows] == [0, 2, 3]
    assert rows[0]["abstention_recall"] == 0.0
    assert rows[1]["abstention_recall"] == 1.0  # the unanswerable one is now caught
    assert rows[1]["answerable_coverage"] == pytest.approx(0.5)  # at the cost of one answer
    assert rows[2]["answerable_coverage"] == 0.0


def test_a_question_already_refused_stays_refused_in_the_sweep() -> None:
    results = [result("scope", should=True, refused=True, best=0.99, reason="out_of_scope")]
    rows = sweep_threshold(results, (0.0, 0.5))
    assert [r["refused"] for r in rows] == [1, 1]
    assert rows[0]["abstention_precision"] == 1.0


def test_reason_matching_accepts_any_stage_two_reason() -> None:
    assert result("a", should=True, refused=True).reason_matches
    lenient = result("b", should=True, refused=True, reason="verification_failed")
    assert lenient.reason_matches  # both mean "the evidence did not hold up"
    wrong_way = result("c", should=False, refused=True)
    assert not wrong_way.reason_matches
