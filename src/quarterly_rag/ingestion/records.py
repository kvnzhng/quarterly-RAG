"""Parsed sections as JSONL records with full provenance (RAG-004).

One record per SEC Item. Offsets index into `<accession>.txt`, written beside the
records, so a section, a chunk (RAG-005), a gold evidence span (RAG-019) and a citation
(RAG-010) all address the same string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field

from quarterly_rag.config import Settings
from quarterly_rag.ingestion.manifest import Filing, Manifest
from quarterly_rag.ingestion.parse import Coverage, parse_filing


class SectionRecord(BaseModel):
    """Provenance fields are required, never optional (project principle)."""

    ticker: str
    cik: int
    company: str
    form: str
    accession: str
    filing_date: date
    period_of_report: date
    fiscal_year: int
    fiscal_quarter: int | None
    period_label: str
    part: int
    item: str
    section: str = Field(description="Stable key, e.g. 'Part I.Item 2' or 'Item 7'")
    title: str
    char_start: int
    char_end: int
    text: str
    source_url: str
    text_path: str = Field(description="Normalized filing text, relative to data_dir")


@dataclass(frozen=True)
class ParseResult:
    accession: str
    form: str
    period_label: str
    sections: int
    chars: int
    coverage: Coverage
    records_path: Path
    written: bool

    @property
    def ok(self) -> bool:
        return self.coverage.ok


@dataclass
class ParseReport:
    ticker: str
    results: list[ParseResult] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    """(accession, message) for filings that could not be parsed at all."""

    @property
    def failures(self) -> int:
        return len(self.errors) + sum(1 for r in self.results if not r.ok)


def _write_if_changed(path: Path, text: str) -> bool:
    """Keeps re-parsing a no-op on disk, the way the downloader keeps re-downloading one."""
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return True


def parse_one(settings: Settings, filing: Filing) -> ParseResult:
    html = (settings.data_dir / filing.path).read_text(encoding="utf-8", errors="replace")
    parsed = parse_filing(html)
    out_dir = settings.processed_dir / filing.ticker
    text_relative = Path("processed") / filing.ticker / f"{filing.accession}.txt"
    records_path = out_dir / f"{filing.accession}.jsonl"

    lines = []
    for section in parsed.sections:
        record = SectionRecord(
            ticker=filing.ticker,
            cik=filing.cik,
            company=filing.company,
            form=filing.form,
            accession=filing.accession,
            filing_date=filing.filing_date,
            period_of_report=filing.period_of_report,
            fiscal_year=filing.fiscal_year,
            fiscal_quarter=filing.fiscal_quarter,
            period_label=filing.period_label,
            part=section.part,
            item=section.item,
            section=section.key,
            title=section.title,
            char_start=section.char_start,
            char_end=section.char_end,
            text=section.text,
            source_url=filing.source_url,
            text_path=text_relative.as_posix(),
        )
        lines.append(record.model_dump_json())

    written = _write_if_changed(settings.data_dir / text_relative, parsed.text)
    written |= _write_if_changed(records_path, "\n".join(lines) + "\n" if lines else "")
    return ParseResult(
        accession=filing.accession,
        form=filing.form,
        period_label=filing.period_label,
        sections=len(parsed.sections),
        chars=len(parsed.text),
        coverage=parsed.coverage(filing.form),
        records_path=records_path,
        written=written,
    )


def parse_ticker(settings: Settings, ticker: str) -> ParseReport:
    """Parse every filing in a ticker's manifest into `data/processed/<TICKER>/`."""
    ticker = ticker.upper()
    manifest = Manifest.load(Manifest.path_for(settings.raw_dir, ticker))
    if manifest is None:
        raise FileNotFoundError(
            f"no manifest for {ticker}; run `rag ingest download --ticker {ticker}` first"
        )
    report = ParseReport(ticker=ticker)
    for filing in manifest.filings:
        try:
            report.results.append(parse_one(settings, filing))
        except (OSError, ValueError) as exc:
            report.errors.append((filing.accession, f"{exc.__class__.__name__}: {exc}"))
    return report


def load_records(path: Path) -> list[SectionRecord]:
    return [
        SectionRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
