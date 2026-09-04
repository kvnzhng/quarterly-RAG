from __future__ import annotations

from datetime import date
from itertools import pairwise
from pathlib import Path

import pytest

from quarterly_rag.chunking.base import Chunk, Chunker, count_words
from quarterly_rag.chunking.fixed import FixedWindowChunker
from quarterly_rag.ingestion.parse import TABLE_CLOSE, TABLE_OPEN
from quarterly_rag.ingestion.records import SectionRecord

FIXTURES = Path(__file__).parent.parent / "ingestion" / "fixtures"


def make_section(
    text: str, char_start: int = 1000, section: str = "Part I.Item 2"
) -> SectionRecord:
    return SectionRecord(
        ticker="AAPL",
        cik=320193,
        company="Apple Inc.",
        form="10-Q",
        accession="0000320193-26-000020",
        filing_date=date(2026, 7, 31),
        period_of_report=date(2026, 6, 27),
        fiscal_year=2026,
        fiscal_quarter=3,
        period_label="FY2026 Q3",
        part=1,
        item="2",
        section=section,
        title="Management's Discussion and Analysis",
        char_start=char_start,
        char_end=char_start + len(text),
        text=text,
        source_url="https://www.sec.gov/Archives/edgar/data/320193/x/aapl.htm",
        text_path="processed/AAPL/0000320193-26-000020.txt",
    )


def paragraph(word: str, n: int) -> str:
    return " ".join(f"{word}{i}" for i in range(n))


def test_satisfies_the_protocol_and_names_itself() -> None:
    chunker = FixedWindowChunker()
    assert isinstance(chunker, Chunker)
    assert chunker.name == "fixed"


def test_short_section_is_one_chunk_with_full_provenance() -> None:
    section = make_section("Item 2. Management's Discussion\nNet sales rose.")
    (chunk,) = FixedWindowChunker(target_words=350).split(section)

    assert chunk.text == section.text
    assert (chunk.char_start, chunk.char_end) == (1000, 1000 + len(section.text))
    assert chunk.chunk_id == f"0000320193-26-000020:1000-{1000 + len(section.text)}"
    assert chunk.strategy == "fixed"
    assert (chunk.ticker, chunk.cik, chunk.company) == ("AAPL", 320193, "Apple Inc.")
    assert (chunk.section, chunk.item, chunk.part) == ("Part I.Item 2", "2", 1)
    assert (chunk.period_label, chunk.fiscal_quarter) == ("FY2026 Q3", 3)
    assert chunk.source_url.endswith("aapl.htm")
    assert chunk.word_count == count_words(section.text)
    assert chunk.contains_table is False


def test_offsets_index_into_the_filing_text() -> None:
    body = "\n".join(paragraph(f"line{i}", 40) for i in range(20))
    section = make_section(body, char_start=500)
    # The filing text is whatever precedes the section plus the section itself.
    filing_text = "x" * 500 + body

    chunks = FixedWindowChunker(target_words=200, overlap_words=40).split(section)
    assert len(chunks) > 1
    for chunk in chunks:
        assert filing_text[chunk.char_start : chunk.char_end] == chunk.text


def test_chunks_advance_and_overlap_by_whole_lines() -> None:
    body = "\n".join(paragraph(f"l{i}", 40) for i in range(20))
    chunks = FixedWindowChunker(target_words=200, overlap_words=45).split(make_section(body))

    starts = [c.char_start for c in chunks]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)  # every chunk advances
    for earlier, later in pairwise(chunks):
        assert later.char_start < earlier.char_end  # they overlap
        assert body[later.char_start - 1000 - 1] == "\n"  # at a line boundary


def test_zero_overlap_produces_a_clean_partition() -> None:
    body = "\n".join(paragraph(f"l{i}", 40) for i in range(12))
    chunks = FixedWindowChunker(target_words=120, overlap_words=0).split(make_section(body))
    assert len(chunks) > 2
    for earlier, later in pairwise(chunks):
        assert later.char_start > earlier.char_end


def test_a_table_is_never_split() -> None:
    rows = "\n".join(f"Row {i} | {i * 111:,} | {i * 222:,}" for i in range(60))
    body = (
        paragraph("intro", 120)
        + f"\n{TABLE_OPEN}\nheader: Three Months Ended | Nine Months Ended\n{rows}\n{TABLE_CLOSE}\n"
        + paragraph("outro", 120)
    )
    chunks = FixedWindowChunker(target_words=150, overlap_words=20).split(make_section(body))

    for chunk in chunks:
        assert chunk.text.count(TABLE_OPEN) == chunk.text.count(TABLE_CLOSE)
    holding = [c for c in chunks if TABLE_OPEN in c.text]
    assert len(holding) == 1
    assert holding[0].contains_table is True
    assert "Row 59 |" in holding[0].text
    assert "header: Three Months Ended" in holding[0].text


def test_a_table_larger_than_the_window_becomes_one_oversized_chunk() -> None:
    rows = "\n".join(f"Row {i} | {i:,}" for i in range(200))
    body = f"{TABLE_OPEN}\nheader: Year Ended\n{rows}\n{TABLE_CLOSE}"
    (chunk,) = FixedWindowChunker(target_words=50, overlap_words=10).split(make_section(body))
    assert chunk.word_count > 50  # atomic beats truncated
    assert chunk.text == body


def test_overlap_never_carries_a_table_forward() -> None:
    table = (
        f"{TABLE_OPEN}\nheader: Year Ended\n"
        + "\n".join(f"Row {i} | {i}" for i in range(20))
        + f"\n{TABLE_CLOSE}"
    )
    body = (
        paragraph("a", 60)
        + "\n"
        + table
        + "\n"
        + "\n".join(paragraph(f"b{i}", 40) for i in range(6))
    )
    chunks = FixedWindowChunker(target_words=120, overlap_words=60).split(make_section(body))
    holding = [i for i, c in enumerate(chunks) if TABLE_OPEN in c.text]
    assert len(holding) == 1


def test_empty_and_blank_sections_produce_nothing() -> None:
    assert FixedWindowChunker().split(make_section("")) == []
    assert FixedWindowChunker().split(make_section("\n\n  \n")) == []


def test_bad_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="target_words"):
        FixedWindowChunker(target_words=0)
    with pytest.raises(ValueError, match="overlap_words"):
        FixedWindowChunker(target_words=100, overlap_words=100)


def test_chunk_ids_are_positional_not_sequential() -> None:
    body = "\n".join(paragraph(f"l{i}", 40) for i in range(8))
    chunks = FixedWindowChunker(target_words=120, overlap_words=0).split(make_section(body))
    assert [c.chunk_id for c in chunks] == [
        Chunk.make_id(c.accession, c.char_start, c.char_end) for c in chunks
    ]
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_word_count_ignores_layout() -> None:
    assert count_words("a  b\nc\t d") == 4
    assert count_words("") == 0
