from __future__ import annotations

import pytest

from quarterly_rag.config import Settings
from quarterly_rag.indexing.base import VectorStore
from quarterly_rag.indexing.chroma import ChromaStore


@pytest.fixture
def store(settings: Settings) -> ChromaStore:
    return ChromaStore(settings.index_dir / "chroma" / "fixed" / "raw")


def test_round_trips_a_chunk_with_all_provenance(
    store: ChromaStore, make_chunk, unit_vector
) -> None:
    assert isinstance(store, VectorStore)
    chunk = make_chunk("a:1-2", "Total net sales | 109,417", contains_table=True)
    store.add([chunk], [unit_vector(1, 0)])
    store.persist()

    (hit,) = store.query(unit_vector(1, 0), k=1)
    assert hit.chunk == chunk  # every field survives the trip, not just the text
    assert hit.score == pytest.approx(1.0, abs=1e-6)


def test_annual_filings_keep_a_null_quarter(store: ChromaStore, make_chunk, unit_vector) -> None:
    # Chroma metadata cannot hold None, so the store maps it to 0 and back.
    chunk = make_chunk("k:1-2", "Item 7. MD and A", form="10-K", fiscal_quarter=None)
    store.add([chunk], [unit_vector(1, 0)])
    (hit,) = store.query(unit_vector(1, 0), k=1)
    assert hit.chunk.fiscal_quarter is None


def test_results_are_ranked_by_similarity(store: ChromaStore, make_chunk, unit_vector) -> None:
    store.add(
        [make_chunk("a:1-2", "near"), make_chunk("b:1-2", "far")],
        [unit_vector(1, 0.1), unit_vector(0, 1)],
    )
    hits = store.query(unit_vector(1, 0), k=2)
    assert [h.chunk.chunk_id for h in hits] == ["a:1-2", "b:1-2"]
    assert hits[0].score > hits[1].score
    assert 0.0 <= hits[1].score <= 1.0


def test_metadata_filter_restricts_the_search(store: ChromaStore, make_chunk, unit_vector) -> None:
    store.add(
        [make_chunk("a:1-2", "apple"), make_chunk("n:1-2", "nvidia", ticker="NVDA")],
        [unit_vector(1, 0), unit_vector(1, 0.01)],
    )
    hits = store.query(unit_vector(1, 0), k=5, where={"ticker": "NVDA"})
    assert [h.chunk.ticker for h in hits] == ["NVDA"]

    hits = store.query(unit_vector(1, 0), k=5, where={"fiscal_year": {"$gte": 2026}})
    assert len(hits) == 2


def test_re_adding_the_same_id_replaces_it(store: ChromaStore, make_chunk, unit_vector) -> None:
    store.add([make_chunk("a:1-2", "first")], [unit_vector(1, 0)])
    store.add([make_chunk("a:1-2", "second")], [unit_vector(1, 0)])
    assert store.count() == 1
    assert store.query(unit_vector(1, 0), k=1)[0].chunk.text == "second"


def test_reopening_the_directory_finds_the_chunks(
    settings: Settings, make_chunk, unit_vector
) -> None:
    path = settings.index_dir / "chroma" / "fixed" / "raw"
    ChromaStore(path).add([make_chunk("a:1-2", "persisted")], [unit_vector(1, 0)])
    reopened = ChromaStore(path)
    assert reopened.count() == 1
    assert reopened.query(unit_vector(1, 0), k=1)[0].chunk.text == "persisted"


def test_empty_store_and_mismatched_input(store: ChromaStore, make_chunk, unit_vector) -> None:
    assert store.query(unit_vector(1, 0), k=5) == []
    assert store.count() == 0
    with pytest.raises(ValueError, match="1 chunks but 2 vectors"):
        store.add([make_chunk("a:1-2", "x")], [unit_vector(1, 0), unit_vector(0, 1)])


def test_k_larger_than_the_collection(store: ChromaStore, make_chunk, unit_vector) -> None:
    store.add([make_chunk("a:1-2", "only")], [unit_vector(1, 0)])
    assert len(store.query(unit_vector(1, 0), k=50)) == 1
