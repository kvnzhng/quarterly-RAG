from __future__ import annotations

import json
from datetime import date

from typer.testing import CliRunner

from quarterly_rag.cli import app
from quarterly_rag.config import Settings
from quarterly_rag.ingestion.download import download_filings
from quarterly_rag.ingestion.edgar import EdgarClient
from quarterly_rag.ingestion.manifest import Manifest

from .edgar_fixtures import UA, FakeEdgar

SINCE = date(2024, 1, 1)


def make_client(fake: FakeEdgar) -> EdgarClient:
    return EdgarClient(UA, transport=fake.transport(), sleep=lambda s: None)


def test_downloads_matching_filings_and_writes_manifest(settings: Settings) -> None:
    fake = FakeEdgar()
    report = download_filings(settings, make_client(fake), "aapl", since=SINCE)

    assert (report.ticker, report.company) == ("AAPL", "Apple Inc.")
    assert [(i.form, i.period_label, i.status) for i in report.items] == [
        ("10-Q", "FY2024 Q2", "new"),
        ("10-K", "FY2025", "new"),
        ("10-Q", "FY2026 Q3", "new"),
    ]
    assert report.manifest_written
    assert report.manifest_path == settings.data_dir / "raw" / "AAPL" / "manifest.json"

    manifest = Manifest.load(report.manifest_path)
    assert manifest is not None
    assert (manifest.cik, manifest.fiscal_year_end) == (320193, "0926")
    tenk = manifest.by_accession()["0000320193-25-000079"]
    assert tenk.path == "raw/AAPL/0000320193-25-000079/aapl-20250927.htm"
    assert (
        (settings.data_dir / tenk.path).read_bytes().startswith(b"<html>aapl-20250927.htm</html>")
    )
    assert tenk.size_bytes == (settings.data_dir / tenk.path).stat().st_size
    assert (tenk.fiscal_year, tenk.fiscal_quarter, tenk.period_of_report) == (
        2025,
        None,
        date(2025, 9, 27),
    )
    assert tenk.source_url.endswith("/320193/000032019325000079/aapl-20250927.htm")
    assert len(fake.urls("/Archives/")) == 3


def test_second_run_is_a_no_op(settings: Settings) -> None:
    fake = FakeEdgar()
    first = download_filings(settings, make_client(fake), "AAPL", since=SINCE)
    manifest_bytes = first.manifest_path.read_bytes()
    archive_requests = len(fake.urls("/Archives/"))

    second = download_filings(settings, make_client(fake), "AAPL", since=SINCE)

    assert [i.status for i in second.items] == ["cached", "cached", "cached"]
    assert second.count("new") == 0
    assert not second.manifest_written
    assert first.manifest_path.read_bytes() == manifest_bytes
    assert len(fake.urls("/Archives/")) == archive_requests


def test_manifest_merges_across_runs_with_different_windows(settings: Settings) -> None:
    fake = FakeEdgar()
    download_filings(settings, make_client(fake), "AAPL", since=date(2025, 1, 1))
    report = download_filings(settings, make_client(fake), "AAPL", since=SINCE)
    assert [i.status for i in report.items] == ["new", "cached", "cached"]
    manifest = Manifest.load(report.manifest_path)
    assert manifest is not None
    assert [f.period_label for f in manifest.filings] == ["FY2024 Q2", "FY2025", "FY2026 Q3"]


def test_file_on_disk_without_manifest_is_recorded_not_refetched(settings: Settings) -> None:
    fake = FakeEdgar()
    report = download_filings(settings, make_client(fake), "AAPL", since=SINCE)
    report.manifest_path.unlink()
    fake.requests.clear()

    report = download_filings(settings, make_client(fake), "AAPL", since=SINCE)
    assert [i.status for i in report.items] == ["cached", "cached", "cached"]
    assert fake.urls("/Archives/") == []
    assert report.manifest_written
    assert len(Manifest.load(report.manifest_path).filings) == 3


def test_one_failed_document_does_not_stop_the_others(settings: Settings) -> None:
    fake = FakeEdgar(failing_docs={"aapl-20250927.htm"})
    report = download_filings(settings, make_client(fake), "AAPL", since=SINCE)
    assert [i.status for i in report.items] == ["new", "failed", "new"]
    assert "HTTP 500" in report.items[1].error
    assert report.count("failed") == 1
    manifest = Manifest.load(report.manifest_path)
    assert [f.form for f in manifest.filings] == ["10-Q", "10-Q"]


def test_cli_ingest_download(monkeypatch, settings: Settings) -> None:
    fake = FakeEdgar()
    monkeypatch.setattr("quarterly_rag.cli.get_settings", lambda: settings)
    monkeypatch.setattr("quarterly_rag.cli.EdgarClient", lambda ua, **kw: make_client(fake))

    result = CliRunner().invoke(app, ["ingest", "download", "-t", "AAPL", "--since", "2024-01-01"])
    assert result.exit_code == 0, result.stdout
    assert "FY2025" in result.stdout
    assert "3 new, 0 cached, 0 failed; manifest written" in result.stdout

    result = CliRunner().invoke(app, ["ingest", "download", "-t", "AAPL", "--since", "2024-01-01"])
    assert result.exit_code == 0
    assert "0 new, 3 cached, 0 failed; manifest unchanged" in result.stdout

    fake.failing_docs.add("aapl-20221231.htm")
    result = CliRunner().invoke(app, ["ingest", "download", "-t", "AAPL", "--since", "2023-01-01"])
    assert result.exit_code == 1
    assert "failed: " in result.stdout


def test_cli_rejects_placeholder_user_agent(monkeypatch, settings: Settings) -> None:
    monkeypatch.setattr("quarterly_rag.cli.get_settings", lambda: settings)  # default UA
    result = CliRunner().invoke(app, ["ingest", "download", "-t", "AAPL"])
    assert result.exit_code == 2
    assert "EDGAR_USER_AGENT" in result.stdout


def test_manifest_round_trips_through_json(settings: Settings) -> None:
    fake = FakeEdgar()
    report = download_filings(settings, make_client(fake), "AAPL", since=SINCE)
    raw = json.loads(report.manifest_path.read_text())
    assert raw["ticker"] == "AAPL"
    assert {f["period_label"] for f in raw["filings"]} == {"FY2024 Q2", "FY2025", "FY2026 Q3"}
