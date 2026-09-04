from __future__ import annotations

from datetime import date

import pytest

from quarterly_rag.config import Settings
from quarterly_rag.ingestion.edgar import EdgarClient, EdgarError, RateLimiter, validate_user_agent

from .edgar_fixtures import AAPL_CIK, AAPL_OLDER, UA, FakeEdgar, page, submissions


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def make_client(fake: FakeEdgar, **kwargs) -> EdgarClient:
    return EdgarClient(UA, transport=fake.transport(), sleep=lambda s: None, **kwargs)


def test_default_settings_user_agent_is_rejected_as_placeholder() -> None:
    with pytest.raises(EdgarError, match="EDGAR_USER_AGENT"):
        validate_user_agent(Settings(_env_file=None).edgar_user_agent)
    with pytest.raises(EdgarError):
        validate_user_agent("quarterly-RAG Jane Doe")  # no email at all
    validate_user_agent("quarterly-RAG Jane Doe jane@her-domain.com")


def test_user_agent_header_is_sent_on_every_request() -> None:
    fake = FakeEdgar()
    client = make_client(fake)
    cik, name = client.lookup("aapl")
    assert (cik, name) == (AAPL_CIK, "Apple Inc.")
    client.submissions(cik)
    assert len(fake.requests) == 2
    assert all(r.headers["user-agent"] == UA for r in fake.requests)


def test_ticker_list_is_fetched_once_and_unknown_ticker_raises() -> None:
    fake = FakeEdgar()
    client = make_client(fake)
    client.lookup("AAPL")
    client.lookup("NVDA")
    assert len(fake.urls("company_tickers")) == 1
    with pytest.raises(EdgarError, match="not in EDGAR"):
        client.lookup("ZZZZ")


def test_filings_filters_forms_since_and_missing_periods() -> None:
    fake = FakeEdgar()
    client = make_client(fake)
    refs = client.filings(
        fake.submissions_by_cik[AAPL_CIK], forms=("10-Q", "10-K"), since=date(2024, 1, 1)
    )
    assert [(r.form, r.accession) for r in refs] == [
        ("10-Q", "0000320193-24-000069"),  # oldest first
        ("10-K", "0000320193-25-000079"),  # the 10-K/A amendment is not a 10-K
        ("10-Q", "0000320193-26-000020"),
    ]
    assert refs[1].source_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
    )
    assert fake.urls("submissions-001") == []  # no need to page: since is inside the recent window


def test_filings_pages_into_older_files_only_when_since_requires_it() -> None:
    fake = FakeEdgar()
    fake.submissions_by_cik[AAPL_CIK] = submissions(
        files=[
            {
                "name": "CIK0000320193-submissions-001.json",
                "filingFrom": "2015-01-01",
                "filingTo": "2022-12-31",
            }
        ]
    )
    fake.extra_pages["CIK0000320193-submissions-001.json"] = page(AAPL_OLDER)
    client = make_client(fake)
    sub = fake.submissions_by_cik[AAPL_CIK]

    refs = client.filings(sub, forms=("10-K",), since=date(2019, 1, 1))
    assert [r.accession for r in refs] == ["0000320193-19-000119", "0000320193-25-000079"]
    assert len(fake.urls("submissions-001")) == 1

    refs = client.filings(sub, forms=("10-K",), since=date(2023, 6, 1))
    assert [r.accession for r in refs] == ["0000320193-25-000079"]
    assert len(fake.urls("submissions-001")) == 1  # not fetched again: filingTo predates since


def test_403_points_at_the_user_agent_setting() -> None:
    fake = FakeEdgar(status_override=403)
    with pytest.raises(EdgarError, match="EDGAR_USER_AGENT"):
        make_client(fake).lookup("AAPL")


def test_rate_limiter_spaces_calls() -> None:
    clock = FakeClock()
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(round(seconds, 6))
        clock.t += seconds

    limiter = RateLimiter(5, clock=clock, sleep=sleep)
    for _ in range(3):
        limiter.wait()
    assert sleeps == [0.2, 0.2]
    clock.t += 10
    limiter.wait()
    assert sleeps == [0.2, 0.2]  # a long idle gap needs no sleep


def test_every_request_kind_goes_through_the_same_limiter(tmp_path) -> None:
    clock = FakeClock()
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(round(seconds, 6))
        clock.t += seconds

    fake = FakeEdgar()
    client = EdgarClient(
        UA, transport=fake.transport(), requests_per_second=4, clock=clock, sleep=sleep
    )
    cik, _ = client.lookup("AAPL")  # www.sec.gov
    sub = client.submissions(cik)  # data.sec.gov
    ref = client.filings(sub, forms=("10-K",), since=date(2025, 1, 1))[0]
    size = client.download(ref, tmp_path / ref.primary_document)  # www.sec.gov archives
    assert size > 0
    assert sleeps == [0.25, 0.25]
    assert (tmp_path / ref.primary_document).exists()
    assert not (tmp_path / (ref.primary_document + ".part")).exists()


def test_requests_per_second_is_capped_at_the_sec_limit() -> None:
    clock = FakeClock()
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(round(seconds, 6))
        clock.t += seconds

    fake = FakeEdgar()
    client = EdgarClient(
        UA, transport=fake.transport(), requests_per_second=1000, clock=clock, sleep=sleep
    )
    client.lookup("AAPL")
    client.submissions(AAPL_CIK)
    assert sleeps == [0.1]
