"""Reciprocal rank fusion of several retrievers (RAG-009).

Cosine similarity and BM25 scores are not commensurable: one is bounded and clustered,
the other unbounded and corpus-dependent. RRF ignores both and fuses on rank alone, which
is why it needs no tuning per retriever and why it is the standard answer here.

Each retriever contributes `1 / (k + rank)`. The constant damps the top of every list so
one retriever's confident first result cannot outvote agreement further down.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from quarterly_rag.retrieval.base import RetrievedChunk, Retriever

DEFAULT_FUSION_K = 60
DEFAULT_POOL = 50
"""Candidates taken from each retriever before fusing. Measured: widening the pool from
20 to 50 moved recall@5 from 39% to 46%, because a chunk both retrievers rank mid-list
beats one that only the leader ranks highly."""


@dataclass
class HybridRetriever:
    retrievers: Sequence[Retriever]
    pool: int = DEFAULT_POOL
    fusion_k: int = DEFAULT_FUSION_K
    label: str = "hybrid"
    contributions: dict[str, int] = field(default_factory=dict)
    """How many of the last fused results each retriever supplied; a diagnostic, not state
    anything depends on."""

    @property
    def name(self) -> str:
        return self.label

    def retrieve(
        self, question: str, k: int = 5, where: dict[str, object] | None = None
    ) -> list[RetrievedChunk]:
        scores: dict[str, float] = {}
        best: dict[str, RetrievedChunk] = {}
        sources: dict[str, set[str]] = {}
        for retriever in self.retrievers:
            for result in retriever.retrieve(question, k=self.pool, where=where):
                chunk_id = result.chunk.chunk_id
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.fusion_k + result.rank)
                sources.setdefault(chunk_id, set()).add(result.retriever)
                # Keep the best-ranked copy so the returned chunk carries a real score.
                if chunk_id not in best or result.rank < best[chunk_id].rank:
                    best[chunk_id] = result

        ordered = sorted(scores, key=lambda cid: -scores[cid])[:k]
        self.contributions = {}
        for chunk_id in ordered:
            for source in sources[chunk_id]:
                self.contributions[source] = self.contributions.get(source, 0) + 1
        return [
            RetrievedChunk(
                chunk=best[chunk_id].chunk,
                score=scores[chunk_id],
                retriever=self.name,
                rank=rank,
            )
            for rank, chunk_id in enumerate(ordered, start=1)
        ]
