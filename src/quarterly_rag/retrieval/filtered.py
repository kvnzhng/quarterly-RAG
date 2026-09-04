"""Metadata filtering inferred from the question (RAG-009).

A question naming one company can skip the other company's half of the index. It is
applied only when exactly one company is named and never on the period, because a filing
quotes prior years for comparison and a period filter would discard the passage that
answers a cross-period question.
"""

from __future__ import annotations

from dataclasses import dataclass

from quarterly_rag.retrieval.base import RetrievedChunk, Retriever
from quarterly_rag.retrieval.query import parse_facets


@dataclass
class FilteredRetriever:
    """Wraps any retriever and adds a filter the question implies."""

    inner: Retriever
    enabled: bool = True

    @property
    def name(self) -> str:
        return f"{self.inner.name}+filter" if self.enabled else self.inner.name

    def retrieve(
        self, question: str, k: int = 5, where: dict[str, object] | None = None
    ) -> list[RetrievedChunk]:
        combined = dict(where or {})
        if self.enabled and (inferred := parse_facets(question).as_filter()):
            # An explicit caller filter wins; the inference only fills a gap.
            combined = {**inferred, **combined}
        results = self.inner.retrieve(question, k=k, where=combined or None)
        return [r.model_copy(update={"retriever": self.name}) for r in results]
