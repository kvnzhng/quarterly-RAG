"""Chunks the whole downloaded corpus and checks the invariants that later tickets rely on."""

from __future__ import annotations

import pytest

from quarterly_rag.chunking.build import build_ticker, iter_chunks
from quarterly_rag.config import get_settings
from quarterly_rag.evaluation.questions import load_questions, questions_path
from quarterly_rag.ingestion.manifest import Manifest
from quarterly_rag.ingestion.parse import TABLE_CLOSE, TABLE_OPEN
from quarterly_rag.ingestion.records import load_records

pytestmark = pytest.mark.integration
TICKERS = ("AAPL", "NVDA")


def _corpus_or_skip():
    settings = get_settings()
    for ticker in TICKERS:
        if Manifest.load(Manifest.path_for(settings.raw_dir, ticker)) is None:
            pytest.skip(f"no corpus for {ticker}; run rag ingest download")
    return settings


def test_every_chunk_resolves_and_stays_in_its_section() -> None:
    settings = _corpus_or_skip()
    for ticker in TICKERS:
        build_ticker(settings, ticker)
        chunks = list(iter_chunks(settings, ticker))
        assert chunks
        texts: dict[str, str] = {}
        sections: dict[str, list] = {}
        for chunk in chunks:
            if chunk.accession not in texts:
                base = settings.processed_dir / ticker / chunk.accession
                texts[chunk.accession] = base.with_suffix(".txt").read_text()
                sections[chunk.accession] = load_records(base.with_suffix(".jsonl"))
            assert texts[chunk.accession][chunk.char_start : chunk.char_end] == chunk.text
            assert chunk.text.count(TABLE_OPEN) == chunk.text.count(TABLE_CLOSE)
            holding = [
                s
                for s in sections[chunk.accession]
                if s.char_start <= chunk.char_start and chunk.char_end <= s.char_end
            ]
            assert len(holding) == 1 and holding[0].section == chunk.section
        assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_every_gold_evidence_span_is_reachable_by_overlap() -> None:
    """RAG-008 scores a chunk relevant when it overlaps a gold span, so no span may be
    stranded between chunks."""
    settings = _corpus_or_skip()
    by_ticker = {t: list(iter_chunks(settings, t)) for t in TICKERS}
    unreachable = []
    for question in load_questions(questions_path(settings)):
        for span in question.evidence:
            overlapping = [
                c
                for c in by_ticker.get(question.ticker, [])
                if c.accession == span.accession
                and c.char_start < span.char_end
                and span.char_start < c.char_end
            ]
            if not overlapping:
                unreachable.append((question.id, span.accession))
    assert not unreachable, unreachable
