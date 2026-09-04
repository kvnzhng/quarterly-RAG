"""Fixed-size window chunker: the v1 strategy (RAG-005).

Deliberately the simplest thing that respects the corpus. It packs whole lines up to a word
target, never crosses a section boundary (it only ever sees one section), and never splits a
table. On this corpus a table is 84 words at the median and 801 at the largest, so treating
tables as atomic costs a handful of oversized chunks and buys a rule with no exceptions.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from quarterly_rag.chunking.base import Chunk, chunk_from_span, count_words
from quarterly_rag.ingestion.parse import TABLE_CLOSE, TABLE_OPEN
from quarterly_rag.ingestion.records import SectionRecord

DEFAULT_WORDS = 350
DEFAULT_OVERLAP_WORDS = 60


@dataclass(frozen=True)
class _Block:
    """A run of lines that must stay together: one table, or one ordinary line."""

    start: int
    """Offset of the block's first character, relative to the section text."""
    end: int
    words: int
    is_table: bool


def _blocks(text: str) -> Iterator[_Block]:
    """Split a section into indivisible blocks, keeping every table whole."""
    offset = 0
    table_start: int | None = None
    table_words = 0
    for line in text.split("\n"):
        end = offset + len(line)
        stripped = line.strip()
        if stripped == TABLE_OPEN and table_start is None:
            table_start = offset
            table_words = count_words(line)
        elif table_start is not None:
            table_words += count_words(line)
            if stripped == TABLE_CLOSE:
                yield _Block(table_start, end, table_words, is_table=True)
                table_start = None
        elif stripped:
            yield _Block(offset, end, count_words(line), is_table=False)
        offset = end + 1
    if table_start is not None:  # a table the parser never closed
        yield _Block(table_start, len(text), table_words, is_table=True)


class FixedWindowChunker:
    """Windows of about `target_words`, overlapping by whole lines.

    Overlap is measured in words and applied at block boundaries, so a chunk always starts
    where a line starts and `text[char_start:char_end]` stays exact.
    """

    def __init__(
        self,
        target_words: int = DEFAULT_WORDS,
        overlap_words: int = DEFAULT_OVERLAP_WORDS,
    ) -> None:
        if target_words <= 0:
            raise ValueError("target_words must be positive")
        if not 0 <= overlap_words < target_words:
            raise ValueError("overlap_words must be non-negative and below target_words")
        self.target_words = target_words
        self.overlap_words = overlap_words

    @property
    def name(self) -> str:
        return "fixed"

    def split(self, section: SectionRecord) -> Sequence[Chunk]:
        blocks = list(_blocks(section.text))
        if not blocks:
            return []
        chunks: list[Chunk] = []
        window: list[_Block] = []
        words = 0
        for block in blocks:
            # A table that does not fit starts its own chunk rather than being cut.
            if window and words + block.words > self.target_words:
                chunks.append(self._chunk(section, window))
                window = self._carry_over(window)
                words = sum(b.words for b in window)
            window.append(block)
            words += block.words
        if window:
            chunks.append(self._chunk(section, window))
        return chunks

    def _carry_over(self, window: list[_Block]) -> list[_Block]:
        """The tail of the finished chunk that the next one repeats, whole blocks only."""
        if not self.overlap_words:
            return []
        carried: list[_Block] = []
        words = 0
        for block in reversed(window):
            if block.is_table or words + block.words > self.overlap_words:
                break
            carried.insert(0, block)
            words += block.words
        return carried

    def _chunk(self, section: SectionRecord, window: list[_Block]) -> Chunk:
        return chunk_from_span(section, self.name, window[0].start, window[-1].end)
