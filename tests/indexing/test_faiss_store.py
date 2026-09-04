from __future__ import annotations

import pytest

from quarterly_rag.chunking.base import Chunk
from quarterly_rag.config import Settings
from quarterly_rag.indexing.base import VectorStore
from quarterly_rag.indexing.faiss_store import INDEX_FILE, PAYLOAD_FILE, FaissStore


@pytest.fixture
def store(settings: Settings) -> FaissStore:
    return FaissStore(settings.index_dir / "faiss", index_type="flat", dimensions=2)


def test_round_trips_a_chunk_with_all_provenance(
    store: FaissStore, make_chunk, unit_vector
) -> None:
    assert isinstance(store, VectorStore)
    assert store.name == "faiss-flat"
    chunk = make_chunk("a:1-2", "Total net sales | 109,417", contains_table=True)
    store.add([chunk], [unit_vector(1, 0)])

    (hit,) = store.query(unit_vector(1, 0), k=1)
    assert hit.chunk == chunk  # FAISS stores no payload, so this proves the parallel list
    assert hit.score == pytest.approx(1.0, abs=1e-5)


def test_results_are_ranked_by_similarity(store: FaissStore, make_chunk, unit_vector) -> None:
    store.add(
        [make_chunk("a:1-2", "near"), make_chunk("b:1-2", "far")],
        [unit_vector(1, 0.1), unit_vector(0, 1)],
    )
    hits = store.query(unit_vector(1, 0), k=2)
    assert [h.chunk.chunk_id for h in hits] == ["a:1-2", "b:1-2"]
    assert hits[0].score > hits[1].score


def test_filtering_happens_after_the_search(store: FaissStore, make_chunk, unit_vector) -> None:
    # FAISS has no filter of its own, so the store over-fetches and discards.
    store.add(
        [make_chunk("a:1-2", "apple"), make_chunk("n:1-2", "nvidia", ticker="NVDA")],
        [unit_vector(1, 0), unit_vector(1, 0.01)],
    )
    hits = store.query(unit_vector(1, 0), k=5, where={"ticker": "NVDA"})
    assert [h.chunk.ticker for h in hits] == ["NVDA"]

    hits = store.query(
        unit_vector(1, 0), k=5, where={"ticker": "AAPL", "period_label": "FY2026 Q3"}
    )
    assert [h.chunk.chunk_id for h in hits] == ["a:1-2"]


def test_reopening_the_directory_finds_the_chunks(
    settings: Settings, make_chunk, unit_vector
) -> None:
    path = settings.index_dir / "faiss"
    first = FaissStore(path, dimensions=2)
    first.add([make_chunk("a:1-2", "persisted")], [unit_vector(1, 0)])
    first.persist()
    assert (path / INDEX_FILE).exists() and (path / PAYLOAD_FILE).exists()

    reopened = FaissStore(path, dimensions=2)
    assert reopened.count() == 1
    assert reopened.query(unit_vector(1, 0), k=1)[0].chunk.text == "persisted"


def test_index_and_payloads_written_out_of_step_are_caught(
    settings: Settings, make_chunk, unit_vector
) -> None:
    path = settings.index_dir / "faiss"
    store = FaissStore(path, dimensions=2)
    store.add(
        [make_chunk("a:1-2", "x"), make_chunk("b:1-2", "y")], [unit_vector(1, 0), unit_vector(0, 1)]
    )
    store.persist()
    # The two files are the store's whole state; a mismatch must not read as an empty index.
    (path / PAYLOAD_FILE).write_text("")
    with pytest.raises(ValueError, match="out of step"):
        FaissStore(path, dimensions=2)


def test_an_hnsw_index_answers_the_same_shape_of_query(
    settings: Settings, make_chunk, unit_vector
) -> None:
    store = FaissStore(settings.index_dir / "hnsw", index_type="hnsw", dimensions=2)
    assert store.name == "faiss-hnsw"
    chunks = [make_chunk(f"c{i}:1-2", f"passage {i}") for i in range(20)]
    store.add(chunks, [unit_vector(1, i / 20) for i in range(20)])
    hits = store.query(unit_vector(1, 0), k=3)
    assert len(hits) == 3
    assert all(isinstance(h.chunk, Chunk) for h in hits)


def test_empty_store_and_mismatched_input(store: FaissStore, make_chunk, unit_vector) -> None:
    assert store.query(unit_vector(1, 0), k=5) == []
    assert store.count() == 0
    store.persist()  # nothing to write, and it must not fail
    with pytest.raises(ValueError, match="1 chunks but 2 vectors"):
        store.add([make_chunk("a:1-2", "x")], [unit_vector(1, 0), unit_vector(0, 1)])


def test_the_vector_width_is_checked(store: FaissStore, make_chunk) -> None:
    with pytest.raises(ValueError, match="2-dimensional"):
        store.add([make_chunk("a:1-2", "x")], [[1.0, 0.0, 0.0]])


def test_re_adding_a_chunk_is_refused_rather_than_duplicated(
    store: FaissStore, make_chunk, unit_vector
) -> None:
    # FAISS has no upsert: adding an id twice would store it twice and both would rank.
    store.add([make_chunk("a:1-2", "first")], [unit_vector(1, 0)])
    with pytest.raises(NotImplementedError, match="build again"):
        store.add([make_chunk("a:1-2", "second")], [unit_vector(1, 0)])


def test_an_unknown_index_type_is_rejected(settings: Settings) -> None:
    with pytest.raises(ValueError, match="unknown FAISS index type"):
        FaissStore(settings.index_dir / "x", index_type="ivf")


def test_k_larger_than_the_collection(store: FaissStore, make_chunk, unit_vector) -> None:
    store.add([make_chunk("a:1-2", "only")], [unit_vector(1, 0)])
    assert len(store.query(unit_vector(1, 0), k=50)) == 1
