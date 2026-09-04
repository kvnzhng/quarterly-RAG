"""Chunking interfaces. A chunk is the unit of retrieval and the unit of evidence.

Offsets index into the same `data/processed/<TICKER>/<accession>.txt` that sections and
gold evidence spans use (RAG-004, RAG-019), so `filing_text[char_start:char_end]` is
exactly `chunk.text`. That invariant is what makes RAG-008's relevance test a range
overlap and RAG-010's citation check a substring test, with no special cases.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from quarterly_rag.ingestion.parse import TABLE_OPEN
from quarterly_rag.ingestion.records import SectionRecord


class Chunk(BaseModel):
    """A retrievable passage. Provenance is required, never optional."""

    chunk_id: str = Field(description="Stable across rebuilds: '<accession>:<start>-<end>'")
    strategy: str

    # provenance, carried whole from the section record it came from
    ticker: str
    cik: int
    company: str
    form: str
    accession: str
    filing_date: str
    period_of_report: str
    fiscal_year: int
    fiscal_quarter: int | None
    period_label: str
    part: int
    item: str
    section: str
    title: str
    source_url: str
    text_path: str

    char_start: int
    char_end: int
    text: str
    word_count: int
    contains_table: bool

    # Parent-child chunking retrieves a small passage and hands the generator the larger
    # one around it. The child is what is embedded and what `text` holds; the parent is
    # the span a citation should quote. Equal to the child's own span for every other
    # strategy, so nothing downstream needs a special case.
    parent_char_start: int | None = None
    parent_char_end: int | None = None

    @property
    def effective_span(self) -> tuple[int, int]:
        """The span this chunk stands for: the parent when there is one, else itself."""
        if self.parent_char_start is None or self.parent_char_end is None:
            return self.char_start, self.char_end
        return self.parent_char_start, self.parent_char_end

    @staticmethod
    def make_id(accession: str, char_start: int, char_end: int) -> str:
        """Derived from position, not a counter, so a boundary moving elsewhere in the
        filing does not renumber every chunk after it."""
        return f"{accession}:{char_start}-{char_end}"


@runtime_checkable
class Chunker(Protocol):
    @property
    def name(self) -> str:
        """Strategy name; also the directory chunks are written under."""
        ...

    def split(self, section: SectionRecord) -> Sequence[Chunk]:
        """Chunks for one section, in document order. Never crosses a section boundary,
        because the caller only ever passes one section."""
        ...


def chunk_from_span(
    section: SectionRecord,
    strategy: str,
    start: int,
    end: int,
    *,
    parent: tuple[int, int] | None = None,
) -> Chunk:
    """Build a chunk for `section.text[start:end]`, carrying the section's provenance.

    Offsets given here are relative to the section; the chunk stores them relative to the
    filing, which is where every gold span and citation points.
    """
    text = section.text[start:end]
    absolute_start = section.char_start + start
    absolute_end = section.char_start + end
    return Chunk(
        chunk_id=Chunk.make_id(section.accession, absolute_start, absolute_end),
        strategy=strategy,
        ticker=section.ticker,
        cik=section.cik,
        company=section.company,
        form=section.form,
        accession=section.accession,
        filing_date=section.filing_date.isoformat(),
        period_of_report=section.period_of_report.isoformat(),
        fiscal_year=section.fiscal_year,
        fiscal_quarter=section.fiscal_quarter,
        period_label=section.period_label,
        part=section.part,
        item=section.item,
        section=section.section,
        title=section.title,
        source_url=section.source_url,
        text_path=section.text_path,
        char_start=absolute_start,
        char_end=absolute_end,
        text=text,
        word_count=count_words(text),
        contains_table=TABLE_OPEN in text,
        parent_char_start=section.char_start + parent[0] if parent else None,
        parent_char_end=section.char_start + parent[1] if parent else None,
    )


def count_words(text: str) -> int:
    """Whitespace words, the unit chunk sizes are expressed in.

    Deliberately not called tokens. On this corpus a word averages 6.4 characters, and a
    BPE tokenizer splits figures like `$109,417` into several tokens, so a word count runs
    well under a model's token count. Sizes here are for comparing strategies (RAG-020),
    not for fitting a context window; RAG-006 can use the embedder's own tokenizer if it
    needs a true limit.
    """
    return len(text.split())
