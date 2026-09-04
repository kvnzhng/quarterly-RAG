"""What text actually goes to the embedding model (RAG-006).

A chunk of a financial table is row labels and figures: the words "Apple" and "third
quarter of fiscal 2026" live in the chunk's provenance, not in its text, so a question
naming either has little to match on. Prepending a one-line header is the fix Anthropic's
contextual retrieval reports gains from, and it is also a variable, because every chunk
from one filing then carries the same boilerplate. Both forms are built so RAG-008 decides
with a number instead of a citation.
"""

from __future__ import annotations

from quarterly_rag.chunking.base import Chunk

RAW = "raw"
CONTEXT = "context"
VARIANTS = (RAW, CONTEXT)


def context_header(chunk: Chunk) -> str:
    """One line naming the company, filing, period and section this chunk came from."""
    return (
        f"{chunk.company} ({chunk.ticker}) {chunk.form} {chunk.period_label} | "
        f"{chunk.section} {chunk.title}"
    )


def embed_text(chunk: Chunk, variant: str = RAW) -> str:
    if variant == RAW:
        return chunk.text
    if variant == CONTEXT:
        return f"{context_header(chunk)}\n\n{chunk.text}"
    raise ValueError(f"unknown embed variant {variant!r}; expected one of {VARIANTS}")
