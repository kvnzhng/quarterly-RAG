from __future__ import annotations

import json

import pytest

from quarterly_rag.config import Settings
from quarterly_rag.evaluation.baseline import (
    DEFAULT_TOLERANCE,
    baseline_path,
    compare,
    load_baseline,
    save_baseline,
)

METRICS = {"recall@5": 0.485, "faithfulness": 0.79, "abstention_f1": 0.80}


def test_a_baseline_round_trips_with_its_run_record(settings: Settings) -> None:
    path = save_baseline(settings, METRICS, {"git_commit": "abc1234", "retriever": "hybrid"})
    assert path == baseline_path(settings)
    stored = load_baseline(settings)
    assert stored["metrics"]["recall@5"] == 0.485
    assert stored["run_record"]["git_commit"] == "abc1234"
    assert stored["tolerance"] == DEFAULT_TOLERANCE
    # Committed alongside the eval set, so it must be readable as plain JSON.
    assert json.loads(path.read_text())["metrics"]


def test_no_baseline_reads_as_none(settings: Settings) -> None:
    assert load_baseline(settings) is None


def test_an_unchanged_run_passes(settings: Settings) -> None:
    save_baseline(settings, METRICS, {})
    check = compare(load_baseline(settings), METRICS)
    assert check.passed
    assert not check.regressions
    assert all(not c.improved for c in check.comparisons)


def test_a_drop_inside_the_tolerance_passes(settings: Settings) -> None:
    # 33 questions means one question is three points, so a small drop is noise.
    save_baseline(settings, METRICS, {}, tolerance=0.05)
    check = compare(load_baseline(settings), {**METRICS, "recall@5": 0.455})
    assert check.passed


def test_a_drop_beyond_the_tolerance_fails(settings: Settings) -> None:
    save_baseline(settings, METRICS, {}, tolerance=0.05)
    check = compare(load_baseline(settings), {**METRICS, "recall@5": 0.40})
    assert not check.passed
    (regression,) = check.regressions
    assert regression.metric == "recall@5"
    assert regression.delta == pytest.approx(-0.085)


def test_an_improvement_is_reported_and_does_not_fail(settings: Settings) -> None:
    save_baseline(settings, METRICS, {}, tolerance=0.05)
    check = compare(load_baseline(settings), {**METRICS, "faithfulness": 0.95})
    assert check.passed
    improved = [c for c in check.comparisons if c.improved]
    assert [c.metric for c in improved] == ["faithfulness"]


def test_a_metric_the_run_did_not_produce_fails(settings: Settings) -> None:
    # A gate that silently skips what it cannot find is not a gate: dropping the judge
    # would otherwise make a faithfulness regression invisible.
    save_baseline(settings, METRICS, {})
    check = compare(load_baseline(settings), {"recall@5": 0.485, "abstention_f1": 0.80})
    assert not check.passed
    assert check.missing == ["faithfulness"]


def test_a_new_metric_not_in_the_baseline_is_ignored(settings: Settings) -> None:
    save_baseline(settings, METRICS, {})
    check = compare(load_baseline(settings), {**METRICS, "brand_new": 0.1})
    assert check.passed
    assert "brand_new" not in {c.metric for c in check.comparisons}


def test_the_tolerance_is_read_from_the_baseline(settings: Settings) -> None:
    save_baseline(settings, METRICS, {}, tolerance=0.01)
    check = compare(load_baseline(settings), {**METRICS, "recall@5": 0.47})
    assert not check.passed  # a 1.5 point drop fails a 1 point tolerance
