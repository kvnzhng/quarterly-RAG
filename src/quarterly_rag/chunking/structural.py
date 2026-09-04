"""Section-aware and parent-child chunking (RAG-020).

RAG-009 measured a ceiling that ranking cannot lift: ten of 33 questions have their
evidence nowhere in the top 100 of 1,391 chunks. Two causes were visible in those chunks.
A financial statement is row labels and figures, sharing almost no words with a
natural-language question. And an answering sentence often sits at the end of a chunk
whose opening is about something else, so the chunk's embedding is dominated by the wrong
topic.

Both are boundary problems, so both strategies here cut on the document's own structure:
filings put a short title line above each block of narrative ("Gross Margin", "Segment
Operating Performance", "Available Information"), and those lines are where a topic
actually changes.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from quarterly_rag.chunking.base import Chunk, chunk_from_span, count_words
from quarterly_rag.chunking.fixed import DEFAULT_OVERLAP_WORDS, DEFAULT_WORDS, FixedWindowChunker
from quarterly_rag.ingestion.parse import TABLE_CLOSE, TABLE_OPEN
from quarterly_rag.ingestion.records import SectionRecord

MAX_HEADING_WORDS = 8
MIN_HEADING_WORDS = 1
DEFAULT_CHILD_WORDS = 120
"""Children are small on purpose: a short passage's embedding is dominated by its own
subject rather than by whatever it was packed next to."""


def looks_like_heading(line: str) -> bool:
    """A short title line, the way a filing writes one.

    Not a table row, not a sentence, not a page footer. Deliberately strict: a false
    heading fragments a section, and fragments are what this strategy exists to avoid.
    """
    stripped = line.strip()
    if not stripped or "|" in stripped:
        return False
    if stripped in {TABLE_OPEN, TABLE_CLOSE} or stripped.startswith("["):
        return False
    words = stripped.split()
    if not MIN_HEADING_WORDS <= len(words) <= MAX_HEADING_WORDS:
        return False
    if stripped.endswith((".", ":", ";", ",")):
        return False
    return bool(re.match(r"[A-Z(]", stripped))


@dataclass(frozen=True)
class Block:
    """A titled run of a section: the heading line plus everything up to the next one."""

    start: int
    end: int
    heading: str


def blocks(text: str) -> Iterator[Block]:
    """Split a section at its own sub-headings. The first block may have none."""
    offset = 0
    starts: list[tuple[int, str]] = []
    in_table = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped == TABLE_OPEN:
            in_table = True
        elif stripped == TABLE_CLOSE:
            in_table = False
        elif not in_table and looks_like_heading(line):
            starts.append((offset, stripped))
        offset += len(line) + 1

    if not starts or starts[0][0] > 0:
        starts.insert(0, (0, ""))
    for index, (start, heading) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(text)
        if text[start:end].strip():
            yield Block(start, end, heading)


class SectionAwareChunker:
    """One chunk per sub-heading, sub-split by the fixed window when a block is too long."""

    def __init__(
        self, target_words: int = DEFAULT_WORDS, overlap_words: int = DEFAULT_OVERLAP_WORDS
    ) -> None:
        self.target_words = target_words
        self.overlap_words = overlap_words
        self._fallback = FixedWindowChunker(target_words, overlap_words)

    @property
    def name(self) -> str:
        return "section-aware"

    def split(self, section: SectionRecord) -> Sequence[Chunk]:
        chunks: list[Chunk] = []
        for block in blocks(section.text):
            body = section.text[block.start : block.end]
            if count_words(body) <= self.target_words:
                chunks.append(chunk_from_span(section, self.name, block.start, block.end))
                continue
            # Too long for one chunk, so pack it with the fixed window and shift the
            # offsets back into the section's frame.
            inner = _as_section(section, block)
            for piece in self._fallback.split(inner):
                offset = piece.char_start - inner.char_start + block.start
                length = piece.char_end - piece.char_start
                chunks.append(chunk_from_span(section, self.name, offset, offset + length))
        return chunks


class ParentChildChunker:
    """Small children for retrieval, the titled block around them for generation.

    The child is what gets embedded, so its vector is about its own few sentences. The
    parent is what a citation quotes, so the generator still sees the table's caption and
    the paragraph that introduces it.
    """

    def __init__(
        self,
        child_words: int = DEFAULT_CHILD_WORDS,
        parent_words: int = DEFAULT_WORDS,
        overlap_words: int = 0,
    ) -> None:
        self.child_words = child_words
        self.parent_words = parent_words
        self._splitter = FixedWindowChunker(child_words, overlap_words)

    @property
    def name(self) -> str:
        return "parent-child"

    def split(self, section: SectionRecord) -> Sequence[Chunk]:
        chunks: list[Chunk] = []
        for block in blocks(section.text):
            parent = (block.start, block.end)
            inner = _as_section(section, block)
            children = self._splitter.split(inner)
            if not children:
                continue
            for child in children:
                offset = child.char_start - inner.char_start + block.start
                length = child.char_end - child.char_start
                chunks.append(
                    chunk_from_span(section, self.name, offset, offset + length, parent=parent)
                )
        return chunks


def _as_section(section: SectionRecord, block: Block) -> SectionRecord:
    """A block presented as a section, so a chunker can be reused on it unchanged."""
    return section.model_copy(
        update={
            "text": section.text[block.start : block.end],
            "char_start": section.char_start + block.start,
            "char_end": section.char_start + block.end,
        }
    )
