"""Retrieval metrics over the eval set (RAG-008).

All three are computed from one relevance judgement (`relevance.is_relevant`), so a change
to the overlap rule moves every metric together rather than one of them quietly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from quarterly_rag.evaluation.questions import EvalQuestion
from quarterly_rag.evaluation.relevance import DEFAULT_RULE, OverlapRule, is_relevant
from quarterly_rag.retrieval.base import RetrievedChunk

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    ticker: str
    form: str
    question_type: str
    section: str
    """Section of the first gold span. A question with spans in two sections is filed
    under the first; the per-question rows keep the detail."""
    first_relevant_rank: int | None
    relevant_retrieved: int
    relevant_in_corpus: int
    retrieved: int
    # How close a miss was, which is what says whether the fix is filtering, ranking, or
    # chunking. Each is "within the top k", filled by the caller.
    filing_rank: int | None = None
    """Rank of the first chunk from the right filing, relevant or not."""
    section_rank: int | None = None
    """Rank of the first chunk from the right filing and the right section."""

    def hit_at(self, k: int) -> bool:
        return self.first_relevant_rank is not None and self.first_relevant_rank <= k

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_relevant_rank if self.first_relevant_rank else 0.0


def evaluate_question(
    question: EvalQuestion,
    results: Sequence[RetrievedChunk],
    *,
    relevant_total: int,
    form: str = "",
    rule: OverlapRule = DEFAULT_RULE,
) -> QuestionResult:
    ranks = [result.rank for result in results if is_relevant(result.chunk, question, rule)]
    accessions = {span.accession for span in question.evidence}
    sections = {(span.accession, span.section) for span in question.evidence}
    filing_ranks = [r.rank for r in results if r.chunk.accession in accessions]
    section_ranks = [r.rank for r in results if (r.chunk.accession, r.chunk.section) in sections]
    return QuestionResult(
        question_id=question.id,
        ticker=question.ticker,
        form=form,
        question_type=question.type,
        section=question.evidence[0].section if question.evidence else "",
        first_relevant_rank=min(ranks) if ranks else None,
        relevant_retrieved=len(ranks),
        relevant_in_corpus=relevant_total,
        retrieved=len(results),
        filing_rank=min(filing_ranks) if filing_ranks else None,
        section_rank=min(section_ranks) if section_ranks else None,
    )


def near_miss_rates(results: Sequence[QuestionResult], k: int) -> dict[str, float]:
    """How far retrieval got: right filing, right section, right chunk.

    The three together say which lever to pull. Finding the filing but not the section
    means ranking or chunking; not finding the filing means filtering or the query itself.
    """
    if not results:
        return {"filing": 0.0, "section": 0.0, "chunk": 0.0}
    within = lambda rank: rank is not None and rank <= k  # noqa: E731
    return {
        "filing": sum(1 for r in results if within(r.filing_rank)) / len(results),
        "section": sum(1 for r in results if within(r.section_rank)) / len(results),
        "chunk": recall_at_k(results, k),
    }


def recall_at_k(results: Sequence[QuestionResult], k: int) -> float:
    """Share of questions with at least one relevant chunk in the top k.

    A hit rate, not fractional recall: a question can have more than one relevant chunk and
    this does not ask for all of them. Kept binary because that is what matters downstream,
    where one good chunk in the prompt is what lets the generator answer.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r.hit_at(k)) / len(results)


def mean_reciprocal_rank(results: Sequence[QuestionResult]) -> float:
    if not results:
        return 0.0
    return sum(r.reciprocal_rank for r in results) / len(results)


def ndcg_at_k(results: Sequence[QuestionResult], k: int, all_ranks: dict[str, list[int]]) -> float:
    """Binary-gain nDCG. The ideal ranking puts every relevant chunk in the corpus first,
    so a question whose evidence straddles two chunks cannot score 1.0 on a single hit."""
    if not results:
        return 0.0
    total = 0.0
    for result in results:
        ranks = [rank for rank in all_ranks.get(result.question_id, []) if rank <= k]
        dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks)
        ideal_hits = min(k, result.relevant_in_corpus)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        total += dcg / ideal if ideal else 0.0
    return total / len(results)


@dataclass
class MetricSet:
    count: int
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: dict[int, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "questions": self.count,
            "recall": {f"@{k}": round(v, 4) for k, v in self.recall.items()},
            "mrr": round(self.mrr, 4),
            "ndcg": {f"@{k}": round(v, 4) for k, v in self.ndcg.items()},
        }


def summarise(
    results: Sequence[QuestionResult],
    all_ranks: dict[str, list[int]],
    ks: Sequence[int] = DEFAULT_KS,
) -> MetricSet:
    return MetricSet(
        count=len(results),
        recall={k: recall_at_k(results, k) for k in ks},
        mrr=mean_reciprocal_rank(results),
        ndcg={k: ndcg_at_k(results, k, all_ranks) for k in ks},
    )


def group_by(
    results: Sequence[QuestionResult],
    key: str,
    all_ranks: dict[str, list[int]],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, MetricSet]:
    buckets: dict[str, list[QuestionResult]] = {}
    for result in results:
        buckets.setdefault(getattr(result, key) or "(none)", []).append(result)
    return {name: summarise(group, all_ranks, ks) for name, group in sorted(buckets.items())}
