"""Parses every downloaded filing. Needs `rag ingest download` to have run first."""

from __future__ import annotations

import pytest

from quarterly_rag.config import get_settings
from quarterly_rag.ingestion.manifest import Manifest
from quarterly_rag.ingestion.records import load_records, parse_ticker

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("ticker", ["AAPL", "NVDA"])
def test_every_downloaded_filing_parses_with_critical_items(ticker: str) -> None:
    settings = get_settings()
    if Manifest.load(Manifest.path_for(settings.raw_dir, ticker)) is None:
        pytest.skip(f"no corpus for {ticker}; run rag ingest download")
    report = parse_ticker(settings, ticker)
    assert report.errors == []
    assert report.results
    for result in report.results:
        assert result.ok, f"{result.accession}: missing {result.coverage.missing_critical}"
        assert result.coverage.unexpected == []
        expected = 23 if result.form == "10-K" else 9
        assert result.sections >= expected if result.form == "10-K" else result.sections >= 9
        records = load_records(result.records_path)
        text = (settings.data_dir / records[0].text_path).read_text()
        for record in records:
            assert text[record.char_start : record.char_end] == record.text
