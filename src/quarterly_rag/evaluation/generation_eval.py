"""Score grounded generation: do citations resolve, and do figures check out? (RAG-010)

Two contexts, because they answer different questions. `gold` hands the generator the
chunks that actually hold the evidence, so the numbers describe the generator and the
verifier alone. `retrieved` runs the real pipeline, so the numbers are end to end and
carry retrieval's recall with them. Reporting only the second would blame the generator
for retrieval's misses.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from quarterly_rag.chunking.base import Chunk
from quarterly_rag.chunking.build import iter_chunks
from quarterly_rag.config import Settings
from quarterly_rag.evaluation.calibration import Calibration, calibrate
from quarterly_rag.evaluation.judge import ClaimJudgement, Judge
from quarterly_rag.evaluation.questions import EvalQuestion, load_questions, questions_path
from quarterly_rag.evaluation.relevance import DEFAULT_RULE, OverlapRule, is_relevant
from quarterly_rag.evaluation.retrieval_eval import REPORTS_DIRNAME, RunRecord, build_run_record
from quarterly_rag.generation.answer import (
    PROMPT_VERSION,
    Answer,
    answer_question,
    parse_tags,
    split_sentences,
    tag_for,
)
from quarterly_rag.generation.base import LLM
from quarterly_rag.retrieval.base import Retriever

GOLD = "gold"
RETRIEVED = "retrieved"
CONTEXTS = (GOLD, RETRIEVED)
TICKERS = ("AAPL", "NVDA")


@dataclass
class AnswerResult:
    question_id: str
    question_type: str
    ticker: str
    passages: int
    insufficient_evidence: bool
    citations: int
    invalid_tags: int
    unsupported_sentences: int
    derived_numbers: int
    fully_grounded: bool
    gold_answer_figures_present: bool
    """Whether the answer states the figures the gold answer states. A weak proxy kept
    alongside the judge so the two can be compared (RAG-012)."""
    answer: str
    faithfulness: float | None = None
    """Share of cited sentences a judge found supported by the passage they cited."""
    correctness: str | None = None
    """correct | partial | incorrect, from the judge (RAG-012)."""


def _verified_sentences(answer: Answer) -> set[str]:
    """Sentences the deterministic verifier passed: cited, and every figure found.

    This is the partial ground truth the judge is calibrated against.
    """
    flagged = {d.sentence for d in answer.derived_numbers} | set(answer.unsupported_sentences)
    return {
        sentence
        for sentence in split_sentences(answer.raw_text.strip() or answer.text)
        if parse_tags(sentence) and sentence not in flagged
    }


@dataclass
class GenerationReport:
    run: RunRecord
    context: str
    results: list[AnswerResult] = field(default_factory=list)
    claims: list[ClaimJudgement] = field(default_factory=list)
    verified_sentences: set[str] = field(default_factory=set)

    def calibration(self) -> Calibration:
        """How the judge compares with the deterministic verifier."""
        return calibrate(self.claims, self.verified_sentences)

    @property
    def answered(self) -> list[AnswerResult]:
        return [r for r in self.results if not r.insufficient_evidence]

    def rates(self, results: Sequence[AnswerResult] | None = None) -> dict[str, float]:
        rows = list(results if results is not None else self.results)
        if not rows:
            return {}
        answered = [r for r in rows if not r.insufficient_evidence]
        divisor = len(answered) or 1
        return {
            "insufficient_evidence": sum(r.insufficient_evidence for r in rows) / len(rows),
            **_judged_rates(answered),
            "citation_resolution": sum(r.invalid_tags == 0 for r in answered) / divisor,
            "all_sentences_cited": sum(r.unsupported_sentences == 0 for r in answered) / divisor,
            "figures_verified": sum(r.derived_numbers == 0 for r in answered) / divisor,
            "fully_grounded": sum(r.fully_grounded for r in answered) / divisor,
            "gold_figures_present": sum(r.gold_answer_figures_present for r in answered) / divisor,
        }

    def judged(self) -> bool:
        return any(r.faithfulness is not None for r in self.results)

    def by_type(self) -> dict[str, dict[str, float]]:
        buckets: dict[str, list[AnswerResult]] = {}
        for result in self.results:
            buckets.setdefault(result.question_type, []).append(result)
        return {
            name: {"questions": len(rows), **self.rates(rows)}
            for name, rows in sorted(buckets.items())
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "run_record": asdict(self.run),
            "context": self.context,
            "overall": {"questions": len(self.results), **self.rates()},
            "by_type": self.by_type(),
            "judge_calibration": self.calibration().as_dict() if self.claims else None,
            "per_question": [asdict(r) for r in self.results],
        }

    def write(self, settings: Settings) -> Path:
        stamp = self.run.timestamp.replace(":", "").replace("-", "")
        path = (
            Path(settings.data_dir).parent
            / REPORTS_DIRNAME
            / f"generation-{self.context}-{stamp}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
        return path


def _gold_figures_present(question: EvalQuestion, answer: Answer) -> bool:
    from quarterly_rag.generation.numbers import parse_figures

    wanted = parse_figures(question.gold_answer)
    if not wanted:
        return question.gold_answer.strip(". ").lower() in answer.text.lower()
    written = {round(f.absolute, 4) for f in parse_figures(answer.text)}
    return all(round(f.absolute, 4) in written for f in wanted)


def run_generation_eval(
    settings: Settings,
    llm: LLM,
    *,
    context: str = GOLD,
    retriever: Retriever | None = None,
    k: int = 5,
    strategy: str = "fixed",
    variant: str = "context",
    store: str = "chroma",
    rule: OverlapRule = DEFAULT_RULE,
    question_types: Sequence[str] = ("lookup",),
    judge: Judge | None = None,
) -> GenerationReport:
    if context not in CONTEXTS:
        raise ValueError(f"unknown context {context!r}; expected one of {CONTEXTS}")
    if context == RETRIEVED and retriever is None:
        raise ValueError("the retrieved context needs a retriever")

    questions = [q for q in load_questions(questions_path(settings)) if q.type in question_types]
    corpus: list[Chunk] = [c for t in TICKERS for c in iter_chunks(settings, t, strategy)]
    if not corpus:
        raise FileNotFoundError(f"no chunks for strategy {strategy!r}; run `rag chunk build` first")

    run = build_run_record(
        settings,
        retriever_name=retriever.name if retriever else "gold-chunks",
        store=store,
        strategy=strategy,
        variant=variant,
        ks=[k],
        rule=rule,
        question_count=len(questions),
    )
    run.prompt_version = PROMPT_VERSION
    report = GenerationReport(run=run, context=context)

    for question in questions:
        if context == GOLD:
            chunks = [c for c in corpus if is_relevant(c, question, rule)][:k]
        else:
            chunks = [r.chunk for r in retriever.retrieve(question.question, k=k)]
        answer = answer_question(
            llm, question.question, chunks, max_tokens=settings.answer_max_tokens
        )
        faithfulness = correctness = None
        if judge is not None and not answer.insufficient_evidence:
            passages = {tag_for(i): c.text for i, c in enumerate(chunks, start=1)}
            judged = judge.faithfulness(answer, passages)
            faithfulness = judged.score if judged.claims else None
            correctness = judge.correctness(question.question, question.gold_answer, answer.text)
            report.claims.extend(judged.claims)
            report.verified_sentences.update(_verified_sentences(answer))
        report.results.append(
            AnswerResult(
                question_id=question.id,
                question_type=question.type,
                ticker=question.ticker,
                passages=len(chunks),
                insufficient_evidence=answer.insufficient_evidence,
                citations=len(answer.citations),
                invalid_tags=len(answer.invalid_tags),
                unsupported_sentences=len(answer.unsupported_sentences),
                derived_numbers=len(answer.derived_numbers),
                fully_grounded=answer.fully_grounded,
                gold_answer_figures_present=_gold_figures_present(question, answer),
                faithfulness=faithfulness,
                correctness=correctness,
                answer=answer.text,
            )
        )
    return report


def _judged_rates(answered: Sequence[AnswerResult]) -> dict[str, float]:
    """Judge-derived rates, absent when no judge ran rather than reported as zero."""
    judged = [r for r in answered if r.faithfulness is not None]
    if not judged:
        return {}
    scored = [r for r in answered if r.correctness is not None]
    rates: dict[str, float] = {
        "faithfulness": sum(r.faithfulness or 0.0 for r in judged) / len(judged),
    }
    if scored:
        rates["correct"] = sum(r.correctness == "correct" for r in scored) / len(scored)
        rates["correct_or_partial"] = sum(
            r.correctness in {"correct", "partial"} for r in scored
        ) / len(scored)
    return rates
