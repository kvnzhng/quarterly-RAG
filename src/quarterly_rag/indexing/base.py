"""Indexing-layer interfaces: `Embedder` (RAG-002) and `VectorStore` (RAG-006)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from quarterly_rag.chunking.base import Chunk


@runtime_checkable
class Embedder(Protocol):
    @property
    def label(self) -> str:
        """`provider/model`, recorded with every index and eval number (ADR-005)."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per passage, in input order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """The vector for a question.

        Separate from `embed_documents` because asymmetric models embed the two sides
        differently: `nomic-embed-text` is trained with `search_query:` and
        `search_document:` prefixes, and omitting them costs real recall on this corpus.
        """
        ...

    def list_models(self) -> list[str]:
        """Model ids the endpoint serves. Raises `ModelServerError` if it cannot say."""
        ...


@dataclass(frozen=True)
class SearchHit:
    """One chunk the store matched, with its similarity in [0, 1] where 1 is identical."""

    chunk: Chunk
    score: float


@runtime_checkable
class VectorStore(Protocol):
    @property
    def name(self) -> str: ...

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Upsert chunks with their vectors. Re-adding the same chunk id replaces it."""
        ...

    def query(
        self,
        vector: Sequence[float],
        k: int = 5,
        where: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        """Nearest chunks, best first. `where` filters on the provenance metadata."""
        ...

    def count(self) -> int: ...

    def persist(self) -> None:
        """Flush to disk. A store that writes as it goes may do nothing here."""
        ...
