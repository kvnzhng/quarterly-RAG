"""Retrieval interfaces. A retriever turns a question into ranked chunks (RAG-006)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from quarterly_rag.chunking.base import Chunk


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    retriever: str
    """`dense` today; `bm25`, `hybrid` and `rerank` follow in RAG-009."""
    rank: int


@runtime_checkable
class Retriever(Protocol):
    @property
    def name(self) -> str: ...

    def retrieve(
        self, question: str, k: int = 5, where: dict[str, object] | None = None
    ) -> list[RetrievedChunk]: ...
