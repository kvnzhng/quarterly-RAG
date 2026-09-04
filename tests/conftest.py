from __future__ import annotations

import math

import pytest

from quarterly_rag.chunking.base import Chunk
from quarterly_rag.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointing at a temp data dir, independent of any local .env."""
    return Settings(_env_file=None, data_dir=tmp_path / "data")


@pytest.fixture
def make_chunk():
    """Factory for a fully-populated Chunk; override any field by keyword."""

    def build(chunk_id: str = "0000320193-26-000020:100-140", text: str = "a passage", **overrides):
        fields = {
            "chunk_id": chunk_id,
            "strategy": "fixed",
            "ticker": "AAPL",
            "cik": 320193,
            "company": "Apple Inc.",
            "form": "10-Q",
            "accession": "0000320193-26-000020",
            "filing_date": "2026-07-31",
            "period_of_report": "2026-06-27",
            "fiscal_year": 2026,
            "fiscal_quarter": 3,
            "period_label": "FY2026 Q3",
            "part": 1,
            "item": "1",
            "section": "Part I.Item 1",
            "title": "Financial Statements",
            "source_url": "https://www.sec.gov/Archives/edgar/data/320193/x/aapl.htm",
            "text_path": "processed/AAPL/0000320193-26-000020.txt",
            "char_start": 100,
            "char_end": 100 + len(text),
            "text": text,
            "word_count": len(text.split()),
            "contains_table": False,
        }
        return Chunk(**(fields | overrides))

    return build


@pytest.fixture
def unit_vector():
    """A two-dimensional unit vector, so cosine similarity is easy to reason about."""

    def build(x: float, y: float) -> list[float]:
        length = math.hypot(x, y)
        return [x / length, y / length]

    return build
