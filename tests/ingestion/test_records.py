from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quarterly_rag.cli import app
from quarterly_rag.config import Settings
from quarterly_rag.ingestion.manifest import Filing, Manifest
from quarterly_rag.ingestion.records import load_records, parse_ticker

FIXTURES = Path(__file__).parent / "fixtures"


def seed(settings: Settings, form: str = "10-Q", fixture: str = "tenq.htm") -> Filing:
    accession = "0000320193-26-000020" if form == "10-Q" else "0000320193-25-000079"
    relative = Path("raw") / "AAPL" / accession / "aapl.htm"
    target = settings.data_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text((FIXTURES / fixture).read_text())
    filing = Filing(
        ticker="AAPL",
        cik=320193,
        company="Apple Inc.",
        form=form,
        accession=accession,
        filing_date=date(2026, 7, 31),
        period_of_report=date(2026, 6, 27),
        fiscal_year=2026,
        fiscal_quarter=3 if form == "10-Q" else None,
        period_label="FY2026 Q3" if form == "10-Q" else "FY2026",
        primary_document="aapl.htm",
        source_url="https://www.sec.gov/Archives/edgar/data/320193/x/aapl.htm",
        path=relative.as_posix(),
        size_bytes=target.stat().st_size,
        downloaded_at=datetime.now(tz=UTC),
    )
    manifest = Manifest(
        ticker="AAPL", cik=320193, company="Apple Inc.", fiscal_year_end="0926", filings=[filing]
    )
    manifest.save(Manifest.path_for(settings.raw_dir, "AAPL"))
    return filing


def test_records_carry_provenance_and_resolve_against_the_text(settings: Settings) -> None:
    seed(settings)
    report = parse_ticker(settings, "aapl")
    assert report.failures == 0
    result = report.results[0]
    assert result.written and result.sections == 4

    records = load_records(result.records_path)
    first = records[0]
    assert (first.ticker, first.cik, first.company) == ("AAPL", 320193, "Apple Inc.")
    assert (first.form, first.period_label, first.fiscal_quarter) == ("10-Q", "FY2026 Q3", 3)
    assert first.section == "Part I.Item 1"
    assert first.source_url.endswith("aapl.htm")
    assert first.text_path == "processed/AAPL/0000320193-26-000020.txt"

    full_text = (settings.data_dir / first.text_path).read_text()
    for record in records:
        assert full_text[record.char_start : record.char_end] == record.text


def test_reparsing_writes_nothing(settings: Settings) -> None:
    seed(settings)
    first = parse_ticker(settings, "AAPL").results[0]
    before = first.records_path.read_bytes()
    second = parse_ticker(settings, "AAPL").results[0]
    assert not second.written
    assert second.records_path.read_bytes() == before


def test_missing_manifest_is_a_clear_error(settings: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="rag ingest download"):
        parse_ticker(settings, "AAPL")


def test_unreadable_filing_is_recorded_not_raised(settings: Settings) -> None:
    filing = seed(settings)
    (settings.data_dir / filing.path).unlink()
    report = parse_ticker(settings, "AAPL")
    assert report.results == []
    assert report.failures == 1
    assert "FileNotFoundError" in report.errors[0][1]


def test_cli_ingest_parse(monkeypatch, settings: Settings) -> None:
    seed(settings)
    monkeypatch.setattr("quarterly_rag.cli.get_settings", lambda: settings)
    result = CliRunner().invoke(app, ["ingest", "parse", "-t", "AAPL"])
    assert result.exit_code == 0, result.stdout
    assert "1 filings parsed, 1 written, 0 failed" in result.stdout
    assert "absent" in result.stdout  # the truncated fixture omits non-critical items

    result = CliRunner().invoke(app, ["ingest", "parse", "-t", "NVDA"])
    assert result.exit_code == 1
    assert "rag ingest download" in result.stdout


def test_jsonl_is_one_record_per_line(settings: Settings) -> None:
    seed(settings, form="10-K", fixture="tenk.htm")
    result = parse_ticker(settings, "AAPL").results[0]
    lines = result.records_path.read_text().strip().split("\n")
    assert len(lines) == 5
    assert {json.loads(line)["section"] for line in lines} == {
        "Part I.Item 1",
        "Part I.Item 1A",
        "Part II.Item 7",
        "Part II.Item 7A",
        "Part II.Item 8",
    }
