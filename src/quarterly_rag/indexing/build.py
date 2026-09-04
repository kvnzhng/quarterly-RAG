"""Embed chunks into a vector store, and record how the index was built (RAG-006).

Every index carries an `index.json` naming the embedding model, the chunking strategy and
the embed variant that produced it. RAG-008's run record quotes it, so a number can always
be traced to the index it came from.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from quarterly_rag.chunking.base import Chunk
from quarterly_rag.chunking.build import iter_chunks
from quarterly_rag.config import Settings
from quarterly_rag.indexing.base import Embedder, VectorStore
from quarterly_rag.indexing.chroma import ChromaStore
from quarterly_rag.indexing.embed_text import RAW, VARIANTS, embed_text

MANIFEST = "index.json"
DEFAULT_BATCH = 32


def index_path(settings: Settings, store: str, strategy: str, variant: str) -> Path:
    return settings.index_dir / store / strategy / variant


def build_store(settings: Settings, store: str, strategy: str, variant: str) -> VectorStore:
    if store == "chroma":
        return ChromaStore(index_path(settings, store, strategy, variant))
    if store == "faiss":
        raise NotImplementedError("the FAISS adapter and the benchmark are RAG-007")
    raise ValueError(f"unknown vector store {store!r}")


@dataclass
class IndexReport:
    store: str
    strategy: str
    variant: str
    path: Path
    embedder: str
    dimensions: int
    tickers: list[str]
    embedded: int
    total: int
    seconds: float

    def as_manifest(self) -> dict[str, object]:
        return {
            "store": self.store,
            "chunk_strategy": self.strategy,
            "embed_variant": self.variant,
            "embedder": self.embedder,
            "dimensions": self.dimensions,
            "tickers": self.tickers,
            "chunks": self.total,
            "built_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        }


def _batched(chunks: Sequence[Chunk], size: int) -> Iterator[Sequence[Chunk]]:
    for start in range(0, len(chunks), size):
        yield chunks[start : start + size]


def build_index(
    settings: Settings,
    tickers: Sequence[str],
    *,
    embedder: Embedder,
    store_name: str = "chroma",
    strategy: str = "fixed",
    variant: str = RAW,
    batch_size: int = DEFAULT_BATCH,
    on_batch: object = None,
) -> IndexReport:
    """Embed every chunk for `tickers` and upsert it into the store.

    Re-running is safe: chunk ids are positional, so an unchanged chunk overwrites itself.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown embed variant {variant!r}; expected one of {VARIANTS}")
    tickers = [t.upper() for t in tickers]
    chunks: list[Chunk] = []
    for ticker in tickers:
        chunks.extend(iter_chunks(settings, ticker, strategy))
    if not chunks:
        raise FileNotFoundError(
            f"no chunks for {', '.join(tickers)} under strategy {strategy!r}; "
            f"run `rag chunk build --ticker {tickers[0]}` first"
        )

    store = build_store(settings, store_name, strategy, variant)
    started = datetime.now(tz=UTC)
    dimensions = 0
    embedded = 0
    for batch in _batched(chunks, batch_size):
        vectors = embedder.embed_documents([embed_text(c, variant) for c in batch])
        store.add(batch, vectors)
        embedded += len(batch)
        dimensions = dimensions or (len(vectors[0]) if vectors else 0)
        if callable(on_batch):
            on_batch(embedded, len(chunks))
    store.persist()

    report = IndexReport(
        store=store_name,
        strategy=strategy,
        variant=variant,
        path=index_path(settings, store_name, strategy, variant),
        embedder=embedder.label,
        dimensions=dimensions,
        tickers=tickers,
        embedded=embedded,
        total=store.count(),
        seconds=(datetime.now(tz=UTC) - started).total_seconds(),
    )
    (report.path / MANIFEST).write_text(json.dumps(report.as_manifest(), indent=2) + "\n")
    return report


def load_manifest(settings: Settings, store: str, strategy: str, variant: str) -> dict | None:
    path = index_path(settings, store, strategy, variant) / MANIFEST
    return json.loads(path.read_text()) if path.exists() else None
