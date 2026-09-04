"""Dense retrieval: embed the question, take the nearest chunks (RAG-006).

The baseline everything else is measured against. BM25, fusion, metadata filters inferred
from the question, and reranking are RAG-009.
"""

from __future__ import annotations

from quarterly_rag.indexing.base import Embedder, VectorStore
from quarterly_rag.retrieval.base import RetrievedChunk


class DenseRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    @property
    def name(self) -> str:
        return "dense"

    def retrieve(
        self, question: str, k: int = 5, where: dict[str, object] | None = None
    ) -> list[RetrievedChunk]:
        if not question.strip():
            return []
        vector = self._embedder.embed_query(question)
        hits = self._store.query(vector, k=k, where=where)
        return [
            RetrievedChunk(chunk=hit.chunk, score=hit.score, retriever=self.name, rank=rank)
            for rank, hit in enumerate(hits, start=1)
        ]
