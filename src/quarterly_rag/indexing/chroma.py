"""ChromaDB adapter behind the `VectorStore` protocol (RAG-006).

Chroma is embedded, persists to a directory, and filters on metadata, which is everything
the pipeline needs from a store. FAISS goes behind the same protocol in RAG-007 and the two
are benchmarked there.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb

from quarterly_rag.chunking.base import Chunk
from quarterly_rag.indexing.base import SearchHit

COLLECTION = "chunks"
NO_QUARTER = 0
"""Chroma metadata cannot hold None, so an annual filing's quarter is stored as 0."""
BATCH = 500


def _metadata(chunk: Chunk) -> dict[str, Any]:
    fields = chunk.model_dump(exclude={"text"})
    fields["fiscal_quarter"] = NO_QUARTER if chunk.fiscal_quarter is None else chunk.fiscal_quarter
    return fields


def _chunk(document: str, metadata: dict[str, Any]) -> Chunk:
    fields = dict(metadata)
    if fields.get("fiscal_quarter") == NO_QUARTER:
        fields["fiscal_quarter"] = None
    return Chunk(**fields, text=document)


class ChromaStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        # Vectors from the embedding endpoint arrive unit-normalised, so cosine is the
        # matching metric and 1 - distance is a similarity in [0, 1].
        self._collection = self._client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    @property
    def name(self) -> str:
        return "chroma"

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        for start in range(0, len(chunks), BATCH):
            window = chunks[start : start + BATCH]
            self._collection.upsert(
                ids=[c.chunk_id for c in window],
                embeddings=[list(v) for v in vectors[start : start + BATCH]],
                documents=[c.text for c in window],
                metadatas=[_metadata(c) for c in window],
            )

    def query(
        self,
        vector: Sequence[float],
        k: int = 5,
        where: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(vector)],
            n_results=min(k, self.count()),
            where=where or None,
        )
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            SearchHit(chunk=_chunk(doc, meta), score=1.0 - float(distance))
            for doc, meta, distance in zip(documents, metadatas, distances, strict=True)
        ]

    def count(self) -> int:
        return self._collection.count()

    def persist(self) -> None:
        """Chroma's persistent client writes as it goes; kept for protocol parity."""
