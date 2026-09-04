"""FAISS adapter behind the `VectorStore` protocol (RAG-007).

FAISS is a similarity index and nothing else: it maps vectors to integer ids and has no
concept of a document, a payload, or a metadata filter. Everything Chroma provides around
the search has to be built here, which is the substance of the comparison rather than an
inconvenience:

- **Payloads** live in a parallel list, persisted as JSONL beside the index.
- **Upsert** does not exist. Adding an id that is already present would duplicate it, so
  a rebuild of an existing id is handled by writing a fresh index.
- **Filtering** happens after the search, so a filtered query must over-fetch. On a corpus
  where the filter is selective this is the difference between the two stores.

Two index types. `flat` is exhaustive and exact. `hnsw` is a navigable small-world graph:
sub-linear and approximate, so it can miss a true neighbour, which the benchmark measures
as recall against flat.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from quarterly_rag.chunking.base import Chunk
from quarterly_rag.indexing.base import SearchHit

INDEX_FILE = "index.faiss"
PAYLOAD_FILE = "chunks.jsonl"
INDEX_TYPES = ("flat", "hnsw")
HNSW_NEIGHBOURS = 32
"""Graph degree. Higher is more accurate and slower to build."""
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64
FILTER_OVERFETCH = 20
"""How much wider to search when a filter is present, since FAISS cannot filter itself."""


class FaissStore:
    def __init__(self, path: Path, *, index_type: str = "flat", dimensions: int = 768) -> None:
        if index_type not in INDEX_TYPES:
            raise ValueError(f"unknown FAISS index type {index_type!r}; expected {INDEX_TYPES}")
        self.path = path
        self.index_type = index_type
        self.dimensions = dimensions
        path.mkdir(parents=True, exist_ok=True)
        self._chunks: list[Chunk] = []
        self._index: Any | None = None
        self._load()

    @property
    def name(self) -> str:
        return f"faiss-{self.index_type}"

    # --- building ----------------------------------------------------------------

    def _new_index(self) -> Any:
        if self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(
                self.dimensions, HNSW_NEIGHBOURS, faiss.METRIC_INNER_PRODUCT
            )
            index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
            index.hnsw.efSearch = HNSW_EF_SEARCH
            return index
        # Vectors arrive unit-normalised from the embedding endpoint, so an inner product
        # is the cosine similarity and no separate normalisation step is needed.
        return faiss.IndexFlatIP(self.dimensions)

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.shape[1] != self.dimensions:
            raise ValueError(
                f"expected {self.dimensions}-dimensional vectors, got {matrix.shape[1]}"
            )
        # FAISS has no upsert: an id added twice is stored twice. Re-adding a chunk that is
        # already present rebuilds from scratch rather than silently duplicating it.
        known = {c.chunk_id for c in self._chunks}
        if any(c.chunk_id in known for c in chunks):
            self._rebuild_without(known & {c.chunk_id for c in chunks})
        if self._index is None:
            self._index = self._new_index()
        self._index.add(matrix)
        self._chunks.extend(chunks)

    def _rebuild_without(self, drop: set[str]) -> None:
        """Rebuilding is the only way to remove: FAISS ids are positional."""
        kept = [c for c in self._chunks if c.chunk_id not in drop]
        if len(kept) == len(self._chunks):
            return
        raise NotImplementedError(
            "FAISS cannot replace a vector in place, and this store keeps no copy of the "
            "vectors to rebuild from. Delete the index directory and build again."
        )

    # --- searching ---------------------------------------------------------------

    def query(
        self,
        vector: Sequence[float],
        k: int = 5,
        where: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        if self._index is None or not self._chunks:
            return []
        # FAISS filters nothing, so a filtered query searches wider and discards after.
        wanted = (
            min(k * FILTER_OVERFETCH, len(self._chunks)) if where else min(k, len(self._chunks))
        )
        query = np.asarray([list(vector)], dtype="float32")
        scores, ids = self._index.search(query, wanted)
        hits: list[SearchHit] = []
        for score, index in zip(scores[0], ids[0], strict=True):
            if index < 0:  # HNSW returns -1 when it finds fewer than requested
                continue
            chunk = self._chunks[int(index)]
            if where and not _matches(chunk, where):
                continue
            hits.append(SearchHit(chunk=chunk, score=float(score)))
            if len(hits) >= k:
                break
        return hits

    def count(self) -> int:
        return len(self._chunks)

    # --- persistence -------------------------------------------------------------

    def persist(self) -> None:
        """Two files, written together: the index and the payloads it has no room for."""
        if self._index is None:
            return
        faiss.write_index(self._index, str(self.path / INDEX_FILE))
        body = "\n".join(c.model_dump_json() for c in self._chunks)
        (self.path / PAYLOAD_FILE).write_text(body + "\n" if body else "", encoding="utf-8")

    def _load(self) -> None:
        index_path = self.path / INDEX_FILE
        payload_path = self.path / PAYLOAD_FILE
        if not (index_path.exists() and payload_path.exists()):
            return
        self._index = faiss.read_index(str(index_path))
        self._chunks = [
            Chunk.model_validate_json(line)
            for line in payload_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if self._index.ntotal != len(self._chunks):
            raise ValueError(
                f"{self.path} holds {self._index.ntotal} vectors and {len(self._chunks)} "
                "payloads; the two files were written out of step"
            )


def _matches(chunk: Chunk, where: dict[str, object]) -> bool:
    return all(getattr(chunk, field, None) == value for field, value in where.items())
