"""Downloads one real Apple 10-K from EDGAR. Deselected by default (`make test-all`)."""

from __future__ import annotations

from datetime import date

import pytest

from quarterly_rag.config import Settings
from quarterly_rag.ingestion.download import download_filings
from quarterly_rag.ingestion.edgar import EdgarClient

pytestmark = pytest.mark.integration


def test_live_download_one_apple_10k(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data")  # user agent from .env, data dir isolated
    client = EdgarClient(settings.edgar_user_agent)
    report = download_filings(settings, client, "AAPL", forms=("10-K",), since=date(2024, 10, 1))
    assert report.count("failed") == 0
    assert report.count("new") >= 1
    newest = report.items[-1]
    assert newest.form == "10-K"
    assert newest.size_bytes > 100_000
    assert newest.period_label.startswith("FY")
