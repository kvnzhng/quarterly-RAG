"""Per-ticker manifest: the filings on disk, so the corpus is reproducible (ADR-004)."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class Filing(BaseModel):
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
    primary_document: str
    source_url: str
    path: str = Field(description="Relative to data_dir, e.g. raw/AAPL/<accession>/<document>")
    size_bytes: int
    downloaded_at: datetime


class Manifest(BaseModel):
    ticker: str
    cik: int
    company: str
    fiscal_year_end: str = Field(description="EDGAR's nominal MMDD fiscal year end")
    filings: list[Filing] = Field(default_factory=list)

    @staticmethod
    def path_for(raw_dir: Path, ticker: str) -> Path:
        return raw_dir / ticker.upper() / "manifest.json"

    @classmethod
    def load(cls, path: Path) -> Manifest | None:
        if not path.exists():
            return None
        return cls.model_validate_json(path.read_text())

    def by_accession(self) -> dict[str, Filing]:
        return {f.accession: f for f in self.filings}

    def upsert(self, filing: Filing) -> None:
        self.filings = [f for f in self.filings if f.accession != filing.accession] + [filing]
        self.filings.sort(key=lambda f: (f.period_of_report, f.accession))

    def save(self, path: Path) -> bool:
        """Write only when the content changed, so a no-op run leaves the file untouched."""
        text = self.model_dump_json(indent=2) + "\n"
        if path.exists() and path.read_text() == text:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text)
        tmp.replace(path)
        return True
