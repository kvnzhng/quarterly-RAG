from __future__ import annotations

from datetime import date
from itertools import pairwise

import pytest

from quarterly_rag.chunking.recursive import RecursiveChunker
from quarterly_rag.chunking.structural import (
    ParentChildChunker,
    SectionAwareChunker,
    blocks,
    looks_like_heading,
)
from quarterly_rag.ingestion.parse import TABLE_CLOSE, TABLE_OPEN
from quarterly_rag.ingestion.records import SectionRecord

BODY = "\n".join(
    [
        "Item 2. Management's Discussion and Analysis",
        "Segment Operating Performance",
        "Americas net sales increased during the third quarter of 2026.",
        "Europe net sales also increased.",
        "Gross Margin",
        TABLE_OPEN,
        "header: 2026 | 2025",
        "Total gross margin | 54,770 | 43,718",
        TABLE_CLOSE,
        "Products gross margin increased primarily due to a favourable mix.",
        "Operating Expenses",
        "Research and development increased due to headcount.",
    ]
)


def section(text: str = BODY, char_start: int = 1000) -> SectionRecord:
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
        section="Part I.Item 2",
        title="Management's Discussion and Analysis",
        char_start=char_start,
        char_end=char_start + len(text),
        text=text,
        source_url="https://www.sec.gov/Archives/edgar/data/320193/x/aapl.htm",
        text_path="processed/AAPL/0000320193-26-000020.txt",
    )


@pytest.mark.parametrize(
    ("line", "is_heading"),
    [
        ("Gross Margin", True),
        ("Segment Operating Performance", True),
        ("(Loss)/income", True),
        ("Americas net sales increased during the third quarter of 2026.", False),
        ("Total gross margin | 54,770 | 43,718", False),  # a table row
        (TABLE_OPEN, False),
        ("header: 2026 | 2025", False),
        ("Available Information:", False),  # ends in a colon, so it introduces prose
        ("", False),
        ("a much longer line that keeps going well past a title length", False),
    ],
)
def test_a_heading_is_a_short_title_line(line, is_heading) -> None:
    assert looks_like_heading(line) is is_heading


def test_a_section_splits_at_its_own_headings() -> None:
    found = list(blocks(BODY))
    assert [b.heading for b in found] == [
        "Item 2. Management's Discussion and Analysis",
        "Segment Operating Performance",
        "Gross Margin",
        "Operating Expenses",
    ]
    # Every character belongs to exactly one block, in order.
    assert found[0].start == 0
    assert found[-1].end == len(BODY)
    for earlier, later in pairwise(found):
        assert earlier.end == later.start


def test_a_heading_inside_a_table_is_not_a_heading() -> None:
    # A short table row would otherwise start a new block mid-table.
    found = list(blocks(BODY))
    margin = next(b for b in found if b.heading == "Gross Margin")
    body = BODY[margin.start : margin.end]
    assert body.count(TABLE_OPEN) == body.count(TABLE_CLOSE) == 1


def test_section_aware_keeps_each_topic_whole() -> None:
    chunker = SectionAwareChunker(target_words=400)
    chunks = chunker.split(section())
    assert chunker.name == "section-aware"
    assert len(chunks) == 4
    assert all(c.strategy == "section-aware" for c in chunks)
    # The table and the sentence explaining it stay together under one heading.
    margin = next(c for c in chunks if "Gross Margin" in c.text)
    assert "Total gross margin | 54,770" in margin.text
    assert "favourable mix" in margin.text
    assert margin.contains_table


def test_section_aware_falls_back_to_the_window_on_a_long_block() -> None:
    long_block = "Operating Expenses\n" + "\n".join(f"sentence {i} of prose" for i in range(200))
    chunks = SectionAwareChunker(target_words=50, overlap_words=0).split(section(long_block))
    assert len(chunks) > 1
    for chunk in chunks:
        assert long_block[chunk.char_start - 1000 : chunk.char_end - 1000] == chunk.text


def test_offsets_resolve_for_every_structural_strategy() -> None:
    for chunker in (
        SectionAwareChunker(target_words=60, overlap_words=0),
        ParentChildChunker(child_words=20, parent_words=200),
        RecursiveChunker(target_words=40, overlap_words=0),
    ):
        for chunk in chunker.split(section()):
            assert BODY[chunk.char_start - 1000 : chunk.char_end - 1000] == chunk.text


def test_parent_child_embeds_the_child_and_stands_for_the_parent() -> None:
    chunker = ParentChildChunker(child_words=8, parent_words=400)
    chunks = chunker.split(section())
    assert chunker.name == "parent-child"
    assert len(chunks) > len(list(blocks(BODY)))

    child = chunks[0]
    assert child.parent_char_start is not None and child.parent_char_end is not None
    # The child is what is embedded; the parent is what a citation would quote.
    assert child.effective_span == (child.parent_char_start, child.parent_char_end)
    assert child.parent_char_start <= child.char_start
    assert child.char_end <= child.parent_char_end
    assert child.word_count <= 400

    # Several children share one parent.
    parents = {c.effective_span for c in chunks}
    assert len(parents) < len(chunks)


def test_a_chunk_without_a_parent_stands_for_itself() -> None:
    (chunk, *_) = SectionAwareChunker(target_words=400).split(section())
    assert chunk.parent_char_start is None
    assert chunk.effective_span == (chunk.char_start, chunk.char_end)


def test_recursive_prefers_line_boundaries() -> None:
    chunks = RecursiveChunker(target_words=12, overlap_words=0).split(section())
    assert len(chunks) > 1
    joined = "".join(c.text for c in chunks)
    assert joined.replace("\n", "") == BODY.replace("\n", "")


def test_bad_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="target_words"):
        RecursiveChunker(target_words=0)
    with pytest.raises(ValueError, match="overlap_words"):
        RecursiveChunker(target_words=10, overlap_words=10)
