"""Indexing-layer interfaces. `Embedder` lands with RAG-002, `VectorStore` with RAG-006."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    @property
    def label(self) -> str:
        """`provider/model`, recorded with every index and eval number (ADR-005)."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per input text, in input order."""
        ...

    def list_models(self) -> list[str]:
        """Model ids the endpoint serves. Raises `ModelServerError` if it cannot say."""
        ...
