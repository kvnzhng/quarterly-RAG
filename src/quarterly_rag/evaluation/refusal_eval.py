"""Abstention evaluation: is the system refusing the right questions? (RAG-011)

Two error types, and they trade against each other. Refusing an answerable question
throws away an answer the corpus contains. Answering an unanswerable one is the failure
this whole project exists to avoid. The threshold sweep prices that trade so the operating
point is chosen from a curve rather than picked.

Precision and recall here are about *refusals*:
- abstention precision: of the questions refused, how many should have been
- abstention recall: of the questions that should have been refused, how many were
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from quarterly_rag.config import Settings
from quarterly_rag.evaluation.questions import load_questions, questions_path
from quarterly_rag.evaluation.retrieval_eval import REPORTS_DIRNAME, RunRecord, build_run_record
from quarterly_rag.generation.answer import PROMPT_VERSION
from quarterly_rag.generation.refusal import GateSettings
from quarterly_rag.pipeline import Pipeline


@dataclass
class RefusalResult:
    question_id: str
    question_type: str
    should_refuse: bool
    refused: bool
    reason: str | None
    expected_reason: str | None
    best_score: float | None
    answer: str = ""

    @property
    def correct(self) -> bool:
        return self.refused == self.should_refuse

    @property
    def reason_matches(self) -> bool:
        """Only meaningful for a correct refusal; the two stage-2 reasons are
        interchangeable with `insufficient_evidence` in the labels."""
        if not (self.refused and self.should_refuse):
            return False
        if self.reason == self.expected_reason:
            return True
        stage_two = {"insufficient_evidence", "verification_failed", "low_confidence"}
        return self.reason in stage_two and self.expected_reason == "insufficient_evidence"


@dataclass
class AbstentionMetrics:
    refused: int
    should_refuse: int
    true_refusals: int
    total: int

    @property
    def precision(self) -> float:
        return self.true_refusals / self.refused if self.refused else 0.0

    @property
    def recall(self) -> float:
        return self.true_refusals / self.should_refuse if self.should_refuse else 0.0

    @property
    def answerable_coverage(self) -> float:
        """Share of answerable questions the system was willing to attempt."""
        answerable = self.total - self.should_refuse
        wrongly_refused = self.refused - self.true_refusals
        return (answerable - wrongly_refused) / answerable if answerable else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "questions": self.total,
            "should_refuse": self.should_refuse,
            "refused": self.refused,
            "correct_refusals": self.true_refusals,
            "abstention_precision": round(self.precision, 4),
            "abstention_recall": round(self.recall, 4),
            "abstention_f1": round(self.f1, 4),
            "answerable_coverage": round(self.answerable_coverage, 4),
        }


def score(results: Sequence[RefusalResult]) -> AbstentionMetrics:
    return AbstentionMetrics(
        refused=sum(r.refused for r in results),
        should_refuse=sum(r.should_refuse for r in results),
        true_refusals=sum(r.refused and r.should_refuse for r in results),
        total=len(results),
    )


@dataclass
class RefusalReport:
    run: RunRecord
    results: list[RefusalResult] = field(default_factory=list)
    sweep: list[dict[str, object]] = field(default_factory=list)

    def by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.results:
            if result.reason:
                counts[result.reason] = counts.get(result.reason, 0) + 1
        return dict(sorted(counts.items()))

    def leaks(self) -> list[RefusalResult]:
        """Unanswerable questions the system answered anyway. The failures that matter."""
        return [r for r in self.results if r.should_refuse and not r.refused]

    def as_dict(self) -> dict[str, object]:
        return {
            "run_record": asdict(self.run),
            "overall": score(self.results).as_dict(),
            "by_reason": self.by_reason(),
            "reason_matches_label": sum(r.reason_matches for r in self.results),
            "threshold_sweep": self.sweep,
            "leaked": [r.question_id for r in self.leaks()],
            "per_question": [asdict(r) for r in self.results],
        }

    def write(self, settings: Settings) -> Path:
        stamp = self.run.timestamp.replace(":", "").replace("-", "")
        path = Path(settings.data_dir).parent / REPORTS_DIRNAME / f"refusal-{stamp}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
        return path


def sweep_threshold(
    results: Sequence[RefusalResult], thresholds: Sequence[float]
) -> list[dict[str, object]]:
    """Replay one run at other thresholds, using each question's best retrieval score.

    Cheap and exact for the `low_confidence` check: raising the threshold can only turn an
    answer into a refusal, and the score that decides it was recorded.
    """
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        replayed = [
            RefusalResult(
                question_id=r.question_id,
                question_type=r.question_type,
                should_refuse=r.should_refuse,
                refused=r.refused or (r.best_score is not None and r.best_score < threshold),
                reason=r.reason
                or (
                    "low_confidence"
                    if r.best_score is not None and r.best_score < threshold
                    else None
                ),
                expected_reason=r.expected_reason,
                best_score=r.best_score,
            )
            for r in results
        ]
        rows.append({"min_retrieval_score": threshold, **score(replayed).as_dict()})
    return rows


def run_refusal_eval(
    settings: Settings,
    pipeline: Pipeline,
    *,
    k: int = 5,
    store: str = "chroma",
    strategy: str = "fixed",
    variant: str = "context",
    thresholds: Sequence[float] = (0.0, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.90),
) -> RefusalReport:
    questions = load_questions(questions_path(settings))
    run = build_run_record(
        settings,
        retriever_name=pipeline.retriever.name,
        store=store,
        strategy=strategy,
        variant=variant,
        ks=[k],
        rule=__import__(
            "quarterly_rag.evaluation.relevance", fromlist=["DEFAULT_RULE"]
        ).DEFAULT_RULE,
        question_count=len(questions),
    )
    run.prompt_version = PROMPT_VERSION
    report = RefusalReport(run=run)

    for question in questions:
        outcome = pipeline.ask(question.question, k=k)
        best = max((r.score for r in outcome.results), default=None)
        report.results.append(
            RefusalResult(
                question_id=question.id,
                question_type=question.type,
                should_refuse=question.type == "unanswerable",
                refused=outcome.refused,
                reason=outcome.reason,
                expected_reason=question.refusal_reason,
                best_score=best,
                answer=outcome.answer.text[:300] if outcome.answer else "",
            )
        )
    report.sweep = sweep_threshold(report.results, thresholds)
    return report


def gate_settings(settings: Settings, min_score: float | None = None) -> GateSettings:
    return GateSettings(
        min_retrieval_score=(settings.min_retrieval_score if min_score is None else min_score)
    )
