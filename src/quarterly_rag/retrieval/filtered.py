"""Metadata filtering inferred from the question (RAG-009), and one query per company
when the question names more than one (RAG-031).

A question naming one company can skip the other company's half of the index. The filter is
never applied on the period, because a filing quotes prior years for comparison and a period
filter would discard the passage that answers a cross-period question.

A question naming *two* companies used to be left unfiltered and asked once, which sounds
even-handed and is not. Nvidia's income statement line is "Revenue" and Apple's is "Net
sales", so a question asking who made more revenue matched six Nvidia passages and no Apple
ones, and the answer was refused for lack of evidence about Apple. One ranked list cannot
represent two companies when the question's wording belongs to one of them.

So each named company is asked separately and the results are interleaved by rank: the best
Apple passage, then the best Nvidia one, then the second of each. Merging by score would
reproduce the original problem, because the scores were what was lopsided.
"""

from __future__ import annotations

from dataclasses import dataclass

from quarterly_rag.retrieval.base import RetrievedChunk, Retriever
from quarterly_rag.retrieval.query import parse_facets


def interleave(per_company: list[list[RetrievedChunk]], k: int) -> list[RetrievedChunk]:
    """Round-robin by rank, so every company reaches the answer.

    With two companies and k=5 the split is 3 and 2, whatever the scores say. A company
    that returned fewer passages than its share simply stops contributing, and the rest of
    the slots go to the others rather than being wasted.
    """
    merged: list[RetrievedChunk] = []
    seen: set[str] = set()
    for position in range(max((len(results) for results in per_company), default=0)):
        for results in per_company:
            if position >= len(results):
                continue
            hit = results[position]
            if hit.chunk.chunk_id in seen:
                continue
            seen.add(hit.chunk.chunk_id)
            merged.append(hit)
            if len(merged) == k:
                return _renumbered(merged)
    return _renumbered(merged)


def _renumbered(results: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Ranks describe the merged list, not the per-company one they came from."""
    return [hit.model_copy(update={"rank": i}) for i, hit in enumerate(results, start=1)]


@dataclass
class FilteredRetriever:
    """Wraps any retriever and adds a filter the question implies."""

    inner: Retriever
    enabled: bool = True
    fall_back: bool = True
    """Retry unfiltered when the filter leaves nothing. A filter that empties the result
    is worse than no filter: the refusal gate would report low confidence for a question
    the corpus can answer."""

    @property
    def name(self) -> str:
        return f"{self.inner.name}+filter" if self.enabled else self.inner.name

    def retrieve(
        self, question: str, k: int = 5, where: dict[str, object] | None = None
    ) -> list[RetrievedChunk]:
        caller = dict(where or {})
        if self.enabled and "ticker" not in caller:
            facets = parse_facets(question)
            if len(facets.tickers) > 1:
                return self._per_company(question, k, caller, facets.tickers)

        combined = dict(caller)
        if self.enabled and (inferred := parse_facets(question).as_filter()):
            # An explicit caller filter wins; the inference only fills a gap.
            combined = {**inferred, **caller}
        results = self.inner.retrieve(question, k=k, where=combined or None)
        if not results and self.fall_back and combined != caller:
            results = self.inner.retrieve(question, k=k, where=where)
        return [r.model_copy(update={"retriever": self.name}) for r in results]

    def _per_company(
        self, question: str, k: int, caller: dict[str, object], tickers: tuple[str, ...]
    ) -> list[RetrievedChunk]:
        """One query per company named, merged so each of them reaches the answer."""
        per_company = [
            self.inner.retrieve(question, k=k, where={**caller, "ticker": ticker})
            for ticker in tickers
        ]
        merged = interleave([r for r in per_company if r], k)
        if not merged and self.fall_back:
            merged = self.inner.retrieve(question, k=k, where=caller or None)
        return [r.model_copy(update={"retriever": self.name}) for r in merged]
