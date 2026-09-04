"""When does a chunk count as finding a question's answer? (RAG-008)

Gold evidence is a character span into the parsed filing (RAG-019) and chunks carry
offsets into the same text (RAG-005), so relevance is a range overlap. That is the whole
point of labelling spans rather than chunk ids: re-chunk and the labels still apply.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from quarterly_rag.chunking.base import Chunk
from quarterly_rag.evaluation.questions import EvalQuestion, EvidenceSpan


@dataclass(frozen=True)
class OverlapRule:
    """How much of a gold span a chunk must cover to count as having found it.

    The default accepts any overlap at all. A stricter rule is what settles an argument
    about whether a chunk clipping three characters of a span really found the evidence.
    """

    min_chars: int = 1
    min_fraction: float = 0.0
    """Fraction of the gold span's length the chunk must cover, 0 to 1."""

    def describe(self) -> str:
        if self.min_fraction:
            return f"min_chars={self.min_chars}, min_fraction={self.min_fraction:g}"
        return f"any overlap (min_chars={self.min_chars})"


DEFAULT_RULE = OverlapRule()


def overlap_chars(chunk: Chunk, span: EvidenceSpan) -> int:
    """Characters shared with a gold span.

    Measured against the chunk's effective span, so a parent-child strategy is judged on
    the passage the generator would receive rather than on the smaller one that was
    embedded to find it.
    """
    if chunk.accession != span.accession:
        return 0
    start, end = chunk.effective_span
    return max(0, min(end, span.char_end) - max(start, span.char_start))


def covers(chunk: Chunk, span: EvidenceSpan, rule: OverlapRule = DEFAULT_RULE) -> bool:
    shared = overlap_chars(chunk, span)
    if shared < rule.min_chars:
        return False
    span_length = span.char_end - span.char_start
    return shared >= rule.min_fraction * span_length


def is_relevant(chunk: Chunk, question: EvalQuestion, rule: OverlapRule = DEFAULT_RULE) -> bool:
    """A chunk is relevant when it covers any one of the question's gold spans."""
    if chunk.ticker != question.ticker:
        return False
    return any(covers(chunk, span, rule) for span in question.evidence)


def relevant_in_corpus(
    chunks: Iterable[Chunk], question: EvalQuestion, rule: OverlapRule = DEFAULT_RULE
) -> int:
    """How many chunks in the whole corpus are relevant.

    nDCG needs this: a question whose evidence straddles two chunks cannot score 1.0 on a
    single hit, and pretending otherwise would flatter every retriever equally.
    """
    return sum(1 for chunk in chunks if is_relevant(chunk, question, rule))
