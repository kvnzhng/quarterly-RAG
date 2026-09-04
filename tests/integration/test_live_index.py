"""Builds both index variants against the live embedding endpoint and measures recall."""

from __future__ import annotations

import pytest

from quarterly_rag.chunking.build import iter_chunks
from quarterly_rag.config import get_settings
from quarterly_rag.evaluation.metrics import recall_at_k
from quarterly_rag.evaluation.retrieval_eval import run_retrieval_eval
from quarterly_rag.indexing.build import build_store, load_manifest
from quarterly_rag.indexing.embed_text import CONTEXT, RAW
from quarterly_rag.indexing.embedder import build_embedder
from quarterly_rag.retrieval.dense import DenseRetriever

pytestmark = pytest.mark.integration


def _index_or_skip(variant: str):
    settings = get_settings()
    store = build_store(settings, "chroma", "fixed", variant)
    if store.count() == 0:
        pytest.skip(f"no {variant} index; run rag index build")
    return settings, store


@pytest.mark.parametrize("variant", [RAW, CONTEXT])
def test_a_query_returns_chunks_with_provenance(variant: str) -> None:
    settings, store = _index_or_skip(variant)
    retriever = DenseRetriever(build_embedder(settings), store)
    results = retriever.retrieve("What were Apple's total net sales?", k=5)

    assert len(results) == 5
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]
    assert results == sorted(results, key=lambda r: -r.score)
    for result in results:
        chunk = result.chunk
        assert chunk.ticker in {"AAPL", "NVDA"}
        assert chunk.accession and chunk.section and chunk.period_label
        assert chunk.source_url.startswith("https://www.sec.gov/")
        assert 0.0 <= result.score <= 1.0
        # The stored text still resolves against the filing it came from.
        filing = (settings.processed_dir / chunk.ticker / f"{chunk.accession}.txt").read_text()
        assert filing[chunk.char_start : chunk.char_end] == chunk.text


def test_the_index_holds_every_chunk_and_says_how_it_was_built() -> None:
    settings, store = _index_or_skip(RAW)
    expected = sum(len(list(iter_chunks(settings, t))) for t in ("AAPL", "NVDA"))
    assert store.count() == expected
    manifest = load_manifest(settings, "chroma", "fixed", RAW)
    assert manifest["chunks"] == expected
    assert manifest["dimensions"] > 0
    assert manifest["embed_variant"] == RAW


def test_a_metadata_filter_restricts_results_to_one_company() -> None:
    settings, store = _index_or_skip(RAW)
    retriever = DenseRetriever(build_embedder(settings), store)
    results = retriever.retrieve("revenue", k=10, where={"ticker": "NVDA"})
    assert results
    assert {r.chunk.ticker for r in results} == {"NVDA"}


def test_context_headers_retrieve_better_than_raw_text() -> None:
    """The reason both variants exist: the choice is measured, not assumed (RAG-006).

    Uses the RAG-008 metrics so there is one definition of recall in the repo.
    """
    settings = get_settings()
    embedder = build_embedder(settings)
    recall = {}
    for variant in (RAW, CONTEXT):
        store = build_store(settings, "chroma", "fixed", variant)
        if store.count() == 0:
            pytest.skip(f"no {variant} index; run rag index build")
        report = run_retrieval_eval(
            settings, DenseRetriever(embedder, store), variant=variant, ks=(5,)
        )
        recall[variant] = recall_at_k(report.results, 5)
    assert recall[CONTEXT] > recall[RAW], recall
