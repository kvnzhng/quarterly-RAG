"""End-to-end: a real filing fixture, parsed and chunked through the actual pipeline."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quarterly_rag.chunking.build import build_ticker, chunks_dir, iter_chunks, load_chunks
from quarterly_rag.cli import app
from quarterly_rag.config import Settings
from quarterly_rag.ingestion.manifest import Filing, Manifest
from quarterly_rag.ingestion.parse import TABLE_CLOSE, TABLE_OPEN
from quarterly_rag.ingestion.records import load_records, parse_ticker

FIXTURES = Path(__file__).parent.parent / "ingestion" / "fixtures"
ACCESSION = "0000320193-26-000020"


@pytest.fixture
def parsed(settings: Settings) -> Settings:
    """A parsed one-filing corpus, built by the real download-then-parse path."""
    relative = Path("raw") / "AAPL" / ACCESSION / "aapl.htm"
    target = settings.data_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text((FIXTURES / "tenq.htm").read_text())
    Manifest(
        ticker="AAPL",
        cik=320193,
        company="Apple Inc.",
        fiscal_year_end="0926",
        filings=[
            Filing(
                ticker="AAPL",
                cik=320193,
                company="Apple Inc.",
                form="10-Q",
                accession=ACCESSION,
                filing_date=date(2026, 7, 31),
                period_of_report=date(2026, 6, 27),
                fiscal_year=2026,
                fiscal_quarter=3,
                period_label="FY2026 Q3",
                primary_document="aapl.htm",
                source_url="https://www.sec.gov/Archives/edgar/data/320193/x/aapl.htm",
                path=relative.as_posix(),
                size_bytes=target.stat().st_size,
                downloaded_at=datetime.now(tz=UTC),
            )
        ],
    ).save(Manifest.path_for(settings.raw_dir, "AAPL"))
    parse_ticker(settings, "AAPL")
    return settings


def test_chunks_stay_inside_their_section_and_resolve_to_the_filing_text(parsed: Settings) -> None:
    report = build_ticker(parsed, "aapl")
    assert report.errors == []
    assert report.written == 1

    sections = load_records(parsed.processed_dir / "AAPL" / f"{ACCESSION}.jsonl")
    filing_text = (parsed.processed_dir / "AAPL" / f"{ACCESSION}.txt").read_text()
    by_key = {s.section: s for s in sections}
    chunks = list(iter_chunks(parsed, "AAPL"))
    assert chunks

    for chunk in chunks:
        assert filing_text[chunk.char_start : chunk.char_end] == chunk.text
        section = by_key[chunk.section]
        assert section.char_start <= chunk.char_start
        assert chunk.char_end <= section.char_end
        assert chunk.text.count(TABLE_OPEN) == chunk.text.count(TABLE_CLOSE)

    assert {c.section for c in chunks} == {s.section for s in sections}
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_a_small_section_becomes_exactly_one_chunk(parsed: Settings) -> None:
    build_ticker(parsed, "AAPL")
    chunks = [c for c in iter_chunks(parsed, "AAPL") if c.section == "Part II.Item 6"]
    assert len(chunks) == 1
    assert "Exhibit 31.1" in chunks[0].text


def test_stats_describe_the_distribution(parsed: Settings) -> None:
    report = build_ticker(parsed, "AAPL")
    stats = report.stats
    assert stats.count == sum(f.chunks for f in report.filings)
    assert 0 < stats.smallest <= stats.median <= stats.p90 <= stats.largest
    assert stats.with_table >= 1


def test_rebuilding_writes_nothing(parsed: Settings) -> None:
    first = build_ticker(parsed, "AAPL")
    path = first.filings[0].path
    before = path.read_bytes()
    second = build_ticker(parsed, "AAPL")
    assert second.written == 0
    assert path.read_bytes() == before


def test_chunk_ids_survive_a_rebuild(parsed: Settings) -> None:
    first = (
        [c.chunk_id for c in iter_chunks(parsed, "AAPL")] if build_ticker(parsed, "AAPL") else []
    )
    build_ticker(parsed, "AAPL")
    assert [c.chunk_id for c in iter_chunks(parsed, "AAPL")] == first


def test_chunks_are_written_under_the_strategy_directory(parsed: Settings) -> None:
    build_ticker(parsed, "AAPL")
    path = chunks_dir(parsed, "fixed", "AAPL") / f"{ACCESSION}.jsonl"
    assert path.exists()
    assert path.parent.parent.name == "fixed"
    assert all(c.strategy == "fixed" for c in load_chunks(path))


def test_unparsed_filing_is_reported_not_raised(parsed: Settings) -> None:
    (parsed.processed_dir / "AAPL" / f"{ACCESSION}.jsonl").unlink()
    report = build_ticker(parsed, "AAPL")
    assert report.filings == []
    assert "rag ingest parse" in report.errors[0][1]


def test_missing_manifest_and_unknown_strategy_are_clear_errors(settings: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="rag ingest download"):
        build_ticker(settings, "AAPL")
    with pytest.raises(ValueError, match="expected one of"):
        build_ticker(settings, "AAPL", strategy="semantic")


def test_cli_chunk_build(monkeypatch, parsed: Settings) -> None:
    monkeypatch.setattr("quarterly_rag.cli.get_settings", lambda: parsed)
    result = CliRunner().invoke(app, ["chunk", "build", "-t", "AAPL"])
    assert result.exit_code == 0, result.stdout
    assert "1 of 1 files written" in result.stdout
    assert "size distribution" in result.stdout

    result = CliRunner().invoke(app, ["chunk", "build", "-t", "AAPL"])
    assert "0 of 1 files written" in result.stdout

    result = CliRunner().invoke(app, ["chunk", "build", "-t", "NVDA"])
    assert result.exit_code == 1
    assert "rag ingest download" in result.stdout
