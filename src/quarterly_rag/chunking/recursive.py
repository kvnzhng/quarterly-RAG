"""Recursive character chunking (RAG-020).

Split on the largest natural boundary that fits, then the next largest: lines, then
sentences, then words. The standard general-purpose strategy, included so the comparison
has a baseline that knows nothing about SEC structure.

It **does** cut tables, unlike every other chunker here, because a table row ends in a
newline and a newline is the first boundary it reaches. That is the strategy rather than a
defect, and it is the weakness the comparison exists to price: on this corpus it leaves
221 chunks holding half a table. Like every chunker it is handed one section at a time, so
it never crosses an Item boundary.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from quarterly_rag.chunking.base import Chunk, count_words
from quarterly_rag.chunking.fixed import DEFAULT_OVERLAP_WORDS, DEFAULT_WORDS, chunk_from_span
from quarterly_rag.ingestion.records import SectionRecord

SEPARATORS: tuple[str, ...] = ("\n", ". ", " ")


class RecursiveChunker:
    def __init__(
        self, target_words: int = DEFAULT_WORDS, overlap_words: int = DEFAULT_OVERLAP_WORDS
    ) -> None:
        if target_words <= 0:
            raise ValueError("target_words must be positive")
        if not 0 <= overlap_words < target_words:
            raise ValueError("overlap_words must be non-negative and below target_words")
        self.target_words = target_words
        self.overlap_words = overlap_words

    @property
    def name(self) -> str:
        return "recursive"

    def split(self, section: SectionRecord) -> Sequence[Chunk]:
        spans = _split_span(section.text, 0, len(section.text), self.target_words, SEPARATORS)
        merged = _merge(section.text, spans, self.target_words, self.overlap_words)
        return [
            chunk_from_span(section, self.name, start, end)
            for start, end in merged
            if section.text[start:end].strip()
        ]


def _split_span(
    text: str, start: int, end: int, target: int, separators: Sequence[str]
) -> list[tuple[int, int]]:
    """Recursively cut [start, end) until each piece fits, preferring larger boundaries."""
    if count_words(text[start:end]) <= target or not separators:
        return [(start, end)]
    separator, rest = separators[0], separators[1:]
    pieces: list[tuple[int, int]] = []
    cursor = start
    for match in re.finditer(re.escape(separator), text[start:end]):
        cut = start + match.end()
        pieces.append((cursor, cut))
        cursor = cut
    if cursor < end:
        pieces.append((cursor, end))
    if len(pieces) <= 1:
        return _split_span(text, start, end, target, rest)
    out: list[tuple[int, int]] = []
    for piece_start, piece_end in pieces:
        out.extend(_split_span(text, piece_start, piece_end, target, rest))
    return out


def _merge(
    text: str, spans: Sequence[tuple[int, int]], target: int, overlap: int
) -> list[tuple[int, int]]:
    """Pack adjacent pieces up to the target.

    A table already cut by the recursive split stays cut: merging cannot rejoin what the
    separators divided, and pretending otherwise would hide the strategy's real cost.
    """
    merged: list[tuple[int, int]] = []
    window: list[tuple[int, int]] = []
    words = 0
    for span in spans:
        piece_words = count_words(text[span[0] : span[1]])
        if window and words + piece_words > target:
            merged.append((window[0][0], window[-1][1]))
            window = _carry(text, window, overlap)
            words = sum(count_words(text[a:b]) for a, b in window)
        window.append(span)
        words += piece_words
    if window:
        merged.append((window[0][0], window[-1][1]))
    return merged


def _carry(text: str, window: Sequence[tuple[int, int]], overlap: int) -> list[tuple[int, int]]:
    if not overlap:
        return []
    carried: list[tuple[int, int]] = []
    words = 0
    for span in reversed(window):
        piece_words = count_words(text[span[0] : span[1]])
        if words + piece_words > overlap:
            break
        carried.insert(0, span)
        words += piece_words
    return carried
