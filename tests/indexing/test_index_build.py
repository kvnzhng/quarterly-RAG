"""Index building end to end with a fake embedder, so no network is touched."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from quarterly_rag.chunking.base import Chunk
from quarterly_rag.chunking.build import chunks_dir
from quarterly_rag.config import Settings
from quarterly_rag.indexing.build import build_index, build_store, index_path, load_manifest
from quarterly_rag.indexing.embed_text import CONTEXT, RAW


class CountingEmbedder:
    """Vectors keyed off the text, so identical text gives identical vectors."""

    label = "fake/embed-2d"

    def __init__(self) -> None:
        self.documents: list[str] = []
        self.batches: list[int] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.documents.extend(texts)
        self.batches.append(len(texts))
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def list_models(self) -> list[str]:
        return ["fake"]


@pytest.fixture
def chunked(settings: Settings, make_chunk) -> Settings:
    path = chunks_dir(settings, "fixed", "AAPL") / "0000320193-26-000020.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [
        make_chunk(
            f"0000320193-26-000020:{i}-{i + 10}", f"passage {i}", char_start=i, char_end=i + 10
        )
        for i in (0, 100, 200)
    ]
    path.write_text("\n".join(c.model_dump_json() for c in chunks) + "\n")
    return settings


def test_builds_an_index_and_records_how(chunked: Settings) -> None:
    embedder = CountingEmbedder()
    report = build_index(chunked, ["aapl"], embedder=embedder, batch_size=2)

    assert report.embedded == 3
    assert report.total == 3
    assert report.dimensions == 2
    assert report.variant == RAW
    assert embedder.batches == [2, 1]  # batched, not one request per chunk
    assert embedder.documents == ["passage 0", "passage 100", "passage 200"]

    manifest = load_manifest(chunked, "chroma", "fixed", RAW)
    assert manifest["embedder"] == "fake/embed-2d"
    assert manifest["chunks"] == 3
    assert manifest["embed_variant"] == RAW
    assert manifest["chunk_strategy"] == "fixed"
    assert manifest["tickers"] == ["AAPL"]
    assert json.loads((report.path / "index.json").read_text()) == manifest


def test_context_variant_embeds_the_header_and_lands_in_its_own_directory(
    chunked: Settings,
) -> None:
    embedder = CountingEmbedder()
    report = build_index(chunked, ["AAPL"], embedder=embedder, variant=CONTEXT)

    assert all(t.startswith("Apple Inc. (AAPL) 10-Q FY2026 Q3") for t in embedder.documents)
    assert report.path == index_path(chunked, "chroma", "fixed", CONTEXT)
    assert report.path != index_path(chunked, "chroma", "fixed", RAW)
    # The two variants are separate indexes, so one does not overwrite the other.
    build_index(chunked, ["AAPL"], embedder=CountingEmbedder(), variant=RAW)
    assert build_store(chunked, "chroma", "fixed", CONTEXT).count() == 3
    assert build_store(chunked, "chroma", "fixed", RAW).count() == 3


def test_rebuilding_replaces_rather_than_duplicates(chunked: Settings) -> None:
    build_index(chunked, ["AAPL"], embedder=CountingEmbedder())
    second = build_index(chunked, ["AAPL"], embedder=CountingEmbedder())
    assert second.total == 3


def test_stored_chunks_keep_their_provenance(chunked: Settings) -> None:
    build_index(chunked, ["AAPL"], embedder=CountingEmbedder())
    store = build_store(chunked, "chroma", "fixed", RAW)
    hits = store.query([1.0, 0.0], k=3)
    assert len(hits) == 3
    for hit in hits:
        assert isinstance(hit.chunk, Chunk)
        assert hit.chunk.ticker == "AAPL"
        assert hit.chunk.period_label == "FY2026 Q3"
        assert hit.chunk.source_url.startswith("https://www.sec.gov/")


def test_missing_chunks_and_bad_arguments_are_clear_errors(settings: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="rag chunk build"):
        build_index(settings, ["AAPL"], embedder=CountingEmbedder())
    with pytest.raises(ValueError, match="unknown embed variant"):
        build_index(settings, ["AAPL"], embedder=CountingEmbedder(), variant="semantic")
    with pytest.raises(NotImplementedError, match="RAG-007"):
        build_store(settings, "faiss", "fixed", RAW)
    with pytest.raises(ValueError, match="unknown vector store"):
        build_store(settings, "pinecone", "fixed", RAW)
