"""Build chunks from parsed sections and report the size distribution (RAG-005).

Chunks are written under `data/chunks/<strategy>/<TICKER>/<accession>.jsonl` so several
strategies can sit side by side; RAG-020 compares them on the same eval labels.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from quarterly_rag.chunking.base import Chunk, Chunker
from quarterly_rag.chunking.fixed import FixedWindowChunker
from quarterly_rag.chunking.recursive import RecursiveChunker
from quarterly_rag.chunking.structural import ParentChildChunker, SectionAwareChunker
from quarterly_rag.config import Settings
from quarterly_rag.ingestion.manifest import Manifest
from quarterly_rag.ingestion.records import load_records

CHUNKS_DIRNAME = "chunks"
SMALL_CHUNK_WORDS = 50
"""Below this a chunk is mostly a heading; worth reporting, not worth preventing."""


def chunks_dir(settings: Settings, strategy: str, ticker: str) -> Path:
    return settings.data_dir / CHUNKS_DIRNAME / strategy / ticker.upper()


STRATEGIES = ("fixed", "recursive", "section-aware", "parent-child")
DEFAULT_STRATEGY = "section-aware"


def build_chunker(strategy: str, settings: Settings) -> Chunker:
    if strategy == "fixed":
        return FixedWindowChunker(settings.chunk_words, settings.chunk_overlap_words)
    if strategy == "recursive":
        return RecursiveChunker(settings.chunk_words, settings.chunk_overlap_words)
    if strategy == "section-aware":
        return SectionAwareChunker(settings.chunk_words, settings.chunk_overlap_words)
    if strategy == "parent-child":
        return ParentChildChunker(settings.child_words, settings.chunk_words)
    raise ValueError(f"unknown chunking strategy {strategy!r}; expected one of {STRATEGIES}")


@dataclass(frozen=True)
class FilingChunks:
    accession: str
    form: str
    period_label: str
    sections: int
    chunks: int
    path: Path
    written: bool


@dataclass
class ChunkStats:
    """Size distribution, plus the two tails that matter when comparing strategies."""

    word_counts: list[int] = field(default_factory=list)
    oversized: int = 0
    """Chunks past the target, which only happens when one table exceeds it."""
    small: int = 0
    with_table: int = 0

    def add(self, chunk: Chunk, target_words: int) -> None:
        self.word_counts.append(chunk.word_count)
        if chunk.word_count > target_words:
            self.oversized += 1
        if chunk.word_count < SMALL_CHUNK_WORDS:
            self.small += 1
        if chunk.contains_table:
            self.with_table += 1

    def _percentile(self, fraction: float) -> int:
        if not self.word_counts:
            return 0
        ordered = sorted(self.word_counts)
        return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]

    @property
    def count(self) -> int:
        return len(self.word_counts)

    @property
    def median(self) -> int:
        return self._percentile(0.5)

    @property
    def p90(self) -> int:
        return self._percentile(0.9)

    @property
    def largest(self) -> int:
        return max(self.word_counts, default=0)

    @property
    def smallest(self) -> int:
        return min(self.word_counts, default=0)


@dataclass
class ChunkReport:
    ticker: str
    strategy: str
    target_words: int
    filings: list[FilingChunks] = field(default_factory=list)
    stats: ChunkStats = field(default_factory=ChunkStats)
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def written(self) -> int:
        return sum(1 for f in self.filings if f.written)


def _write_if_changed(path: Path, text: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return True


def build_ticker(settings: Settings, ticker: str, strategy: str = "fixed") -> ChunkReport:
    ticker = ticker.upper()
    # Config errors first: an unknown strategy is worth reporting whether or not a corpus exists.
    chunker = build_chunker(strategy, settings)
    manifest = Manifest.load(Manifest.path_for(settings.raw_dir, ticker))
    if manifest is None:
        raise FileNotFoundError(
            f"no manifest for {ticker}; run `rag ingest download --ticker {ticker}` first"
        )
    report = ChunkReport(ticker=ticker, strategy=strategy, target_words=settings.chunk_words)
    out_dir = chunks_dir(settings, strategy, ticker)

    for filing in manifest.filings:
        records_path = settings.processed_dir / ticker / f"{filing.accession}.jsonl"
        if not records_path.exists():
            report.errors.append(
                (filing.accession, f"not parsed; run `rag ingest parse --ticker {ticker}`")
            )
            continue
        sections = load_records(records_path)
        chunks: list[Chunk] = []
        for section in sections:
            chunks.extend(chunker.split(section))
        for chunk in chunks:
            report.stats.add(chunk, report.target_words)
        body = "\n".join(chunk.model_dump_json() for chunk in chunks)
        path = out_dir / f"{filing.accession}.jsonl"
        written = _write_if_changed(path, body + "\n" if chunks else "")
        report.filings.append(
            FilingChunks(
                accession=filing.accession,
                form=filing.form,
                period_label=filing.period_label,
                sections=len(sections),
                chunks=len(chunks),
                path=path,
                written=written,
            )
        )
    return report


def load_chunks(path: Path) -> list[Chunk]:
    return [
        Chunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def iter_chunks(settings: Settings, ticker: str, strategy: str = "fixed") -> Iterable[Chunk]:
    for path in sorted(chunks_dir(settings, strategy, ticker).glob("*.jsonl")):
        yield from load_chunks(path)
