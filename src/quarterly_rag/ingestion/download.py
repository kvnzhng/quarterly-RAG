"""Idempotent `rag ingest download`: EDGAR -> data/raw/<TICKER>/<accession>/<doc> + manifest."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from quarterly_rag.config import Settings
from quarterly_rag.ingestion.edgar import EdgarClient, EdgarError, FilingRef
from quarterly_rag.ingestion.fiscal import fiscal_period
from quarterly_rag.ingestion.manifest import Filing, Manifest

DEFAULT_FORMS: tuple[str, ...] = ("10-Q", "10-K")
Status = Literal["new", "cached", "failed"]


@dataclass(frozen=True)
class DownloadItem:
    form: str
    accession: str
    filing_date: date
    period_label: str
    status: Status
    size_bytes: int = 0
    error: str = ""


@dataclass
class DownloadReport:
    ticker: str
    company: str
    manifest_path: Path
    manifest_written: bool = False
    items: list[DownloadItem] = field(default_factory=list)

    def count(self, status: Status) -> int:
        return sum(1 for item in self.items if item.status == status)


def download_filings(
    settings: Settings,
    client: EdgarClient,
    ticker: str,
    *,
    forms: Sequence[str] = DEFAULT_FORMS,
    since: date,
) -> DownloadReport:
    """Download every matching filing not already on disk and update the ticker's manifest.

    A filing already present at its manifest path is left alone (no request). One failed
    document does not stop the others; it is reported and the manifest keeps the successes.
    """
    ticker = ticker.upper()
    cik, _ = client.lookup(ticker)
    submissions = client.submissions(cik)
    company = client.company(submissions)
    refs = client.filings(submissions, forms=forms, since=since)

    manifest_path = Manifest.path_for(settings.raw_dir, ticker)
    manifest = Manifest.load(manifest_path) or Manifest(
        ticker=ticker, cik=cik, company=company.name, fiscal_year_end=company.fiscal_year_end
    )
    known = manifest.by_accession()
    report = DownloadReport(ticker=ticker, company=company.name, manifest_path=manifest_path)

    for ref in refs:
        period = fiscal_period(ref.report_date, company.fiscal_year_end, ref.form)
        relative = Path("raw") / ticker / ref.accession / ref.primary_document
        dest = settings.data_dir / relative
        if ref.accession in known and dest.exists():
            report.items.append(_item(ref, period.label, "cached", known[ref.accession].size_bytes))
            continue
        if dest.exists():  # file present but manifest lost or stale: record without refetching
            size = dest.stat().st_size
            downloaded_at = datetime.fromtimestamp(dest.stat().st_mtime, tz=UTC)
            status: Status = "cached"
        else:
            try:
                size = client.download(ref, dest)
            except EdgarError as exc:
                report.items.append(_item(ref, period.label, "failed", error=str(exc)))
                continue
            downloaded_at = datetime.now(tz=UTC)
            status = "new"
        manifest.upsert(
            Filing(
                ticker=ticker,
                cik=cik,
                company=company.name,
                form=ref.form,
                accession=ref.accession,
                filing_date=ref.filing_date,
                period_of_report=ref.report_date,
                fiscal_year=period.year,
                fiscal_quarter=period.quarter,
                period_label=period.label,
                primary_document=ref.primary_document,
                source_url=ref.source_url,
                path=relative.as_posix(),
                size_bytes=size,
                downloaded_at=downloaded_at,
            )
        )
        report.items.append(_item(ref, period.label, status, size))

    report.manifest_written = manifest.save(manifest_path)
    return report


def _item(
    ref: FilingRef, period_label: str, status: Status, size_bytes: int = 0, error: str = ""
) -> DownloadItem:
    return DownloadItem(
        form=ref.form,
        accession=ref.accession,
        filing_date=ref.filing_date,
        period_label=period_label,
        status=status,
        size_bytes=size_bytes,
        error=error,
    )
