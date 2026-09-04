"""SEC EDGAR client: ticker -> CIK, submissions, primary documents.

The fair-access rules (https://www.sec.gov/os/accessing-edgar-data) are built in: a declared
User-Agent with a contact, and one rate limiter shared by every request to either host,
because the SEC applies its cap per client, not per hostname.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import httpx

WWW = "https://www.sec.gov"
DATA = "https://data.sec.gov"
TICKERS_URL = f"{WWW}/files/company_tickers.json"
SEC_MAX_REQUESTS_PER_SECOND = 10
DEFAULT_REQUESTS_PER_SECOND = 5.0  # comfortably under the SEC cap
PLACEHOLDER_MARKERS = ("example.com", "unknown@", "your-name", "your-email")


class EdgarError(RuntimeError):
    pass


def validate_user_agent(user_agent: str) -> None:
    lowered = user_agent.strip().lower()
    if "@" not in lowered or any(marker in lowered for marker in PLACEHOLDER_MARKERS):
        raise EdgarError(
            "EDGAR_USER_AGENT must name you and a real contact email "
            "(SEC format: '<app> <name> <email>'); the SEC rejects anonymous clients. "
            "Set it in .env."
        )


class RateLimiter:
    """Spaces calls at least `1 / per_second` apart. Clock and sleep are injectable for tests."""

    def __init__(
        self,
        per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = 1.0 / per_second
        self._clock = clock
        self._sleep = sleep
        self._next_allowed = 0.0

    def wait(self) -> None:
        now = self._clock()
        if now < self._next_allowed:
            self._sleep(self._next_allowed - now)
            now = self._next_allowed
        self._next_allowed = now + self._interval


@dataclass(frozen=True)
class Company:
    cik: int
    name: str
    fiscal_year_end: str


@dataclass(frozen=True)
class FilingRef:
    cik: int
    form: str
    accession: str
    filing_date: date
    report_date: date
    primary_document: str

    @property
    def source_url(self) -> str:
        return (
            f"{WWW}/Archives/edgar/data/{self.cik}/"
            f"{self.accession.replace('-', '')}/{self.primary_document}"
        )


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        *,
        timeout_s: float = 60.0,
        requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        validate_user_agent(user_agent)
        self._limiter = RateLimiter(
            min(requests_per_second, SEC_MAX_REQUESTS_PER_SECOND), clock=clock, sleep=sleep
        )
        self._http = httpx.Client(
            headers={"User-Agent": user_agent.strip(), "Accept-Encoding": "gzip, deflate"},
            timeout=timeout_s,
            transport=transport,
            follow_redirects=True,
        )
        self._tickers: dict[str, tuple[int, str]] | None = None

    # --- lookups -----------------------------------------------------------------

    def lookup(self, ticker: str) -> tuple[int, str]:
        """Ticker -> (CIK, company name) from EDGAR's company list; fetched once per client."""
        if self._tickers is None:
            rows = self._get_json(TICKERS_URL)
            self._tickers = {
                str(row["ticker"]).upper(): (int(row["cik_str"]), str(row["title"]))
                for row in rows.values()
            }
        try:
            return self._tickers[ticker.upper()]
        except KeyError:
            raise EdgarError(f"ticker {ticker!r} is not in EDGAR's company list") from None

    def submissions(self, cik: int) -> dict[str, Any]:
        return self._get_json(f"{DATA}/submissions/CIK{cik:010d}.json")

    @staticmethod
    def company(submissions: dict[str, Any]) -> Company:
        return Company(
            cik=int(submissions["cik"]),
            name=str(submissions["name"]),
            fiscal_year_end=str(submissions.get("fiscalYearEnd") or ""),
        )

    def filings(
        self, submissions: dict[str, Any], *, forms: Sequence[str], since: date
    ) -> list[FilingRef]:
        """Filings of the given forms filed on or after `since`, oldest first.

        `recent` holds the latest 1000 filings; older ones sit in extra pages that are only
        fetched when `since` reaches past the recent window.
        """
        cik = int(submissions["cik"])
        pages = [submissions["filings"]["recent"]]
        recent_dates = [date.fromisoformat(d) for d in pages[0]["filingDate"]]
        if recent_dates and since < min(recent_dates):
            for extra in submissions["filings"].get("files", []):
                if date.fromisoformat(extra["filingTo"]) >= since:
                    pages.append(self._get_json(f"{DATA}/submissions/{extra['name']}"))
        wanted = set(forms)
        refs: list[FilingRef] = []
        for page in pages:
            for i, form in enumerate(page["form"]):
                if form not in wanted:
                    continue
                filed = date.fromisoformat(page["filingDate"][i])
                if filed < since or not page["reportDate"][i]:
                    continue
                refs.append(
                    FilingRef(
                        cik=cik,
                        form=form,
                        accession=page["accessionNumber"][i],
                        filing_date=filed,
                        report_date=date.fromisoformat(page["reportDate"][i]),
                        primary_document=page["primaryDocument"][i],
                    )
                )
        refs.sort(key=lambda r: (r.filing_date, r.accession))
        return refs

    # --- downloads ---------------------------------------------------------------

    def download(self, ref: FilingRef, dest: Path) -> int:
        """Fetch the primary document to `dest` atomically; returns the byte count."""
        body = self._get_bytes(ref.source_url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(body)
        tmp.replace(dest)
        return len(body)

    # --- transport ---------------------------------------------------------------

    def _get(self, url: str) -> httpx.Response:
        self._limiter.wait()
        try:
            response = self._http.get(url)
        except httpx.HTTPError as exc:
            raise EdgarError(f"GET {url}: {exc.__class__.__name__}: {exc}") from exc
        if response.status_code == 403:
            raise EdgarError(
                f"GET {url} -> HTTP 403. The SEC refused the request; this is almost always "
                "the User-Agent. Check EDGAR_USER_AGENT in .env."
            )
        if response.is_error:
            raise EdgarError(f"GET {url} -> HTTP {response.status_code}")
        return response

    def _get_json(self, url: str) -> Any:
        try:
            return self._get(url).json()
        except ValueError as exc:
            raise EdgarError(f"GET {url}: response is not JSON") from exc

    def _get_bytes(self, url: str) -> bytes:
        return self._get(url).content
