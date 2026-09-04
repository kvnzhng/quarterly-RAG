"""A fake EDGAR behind httpx.MockTransport, shared by the ingestion tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

AAPL_CIK = 320193
NVDA_CIK = 1045810
UA = "quarterly-RAG Test Person test@quarterly-rag.test"

# (accession, filingDate, reportDate, form, primaryDocument)
AAPL_RECENT = [
    ("0000320193-26-000020", "2026-07-31", "2026-06-27", "10-Q", "aapl-20260627.htm"),
    ("0000320193-25-000080", "2025-11-05", "2025-09-27", "10-K/A", "aapl-20250927a.htm"),
    ("0000320193-25-000079", "2025-10-31", "2025-09-27", "10-K", "aapl-20250927.htm"),
    ("0000320193-25-000050", "2025-04-01", "", "8-K", "aapl-8k.htm"),
    ("0000320193-24-000070", "2024-05-04", "", "10-Q", "aapl-noperiod.htm"),
    ("0000320193-24-000069", "2024-05-03", "2024-03-30", "10-Q", "aapl-20240330.htm"),
    ("0000320193-23-000006", "2023-02-03", "2022-12-31", "10-Q", "aapl-20221231.htm"),
]
AAPL_OLDER = [
    ("0000320193-19-000119", "2019-10-31", "2019-09-28", "10-K", "aapl-20190928.htm"),
]

KEYS = ["accessionNumber", "filingDate", "reportDate", "form", "primaryDocument"]


def page(rows: list[tuple[str, ...]]) -> dict[str, list[str]]:
    return {key: [row[i] for row in rows] for i, key in enumerate(KEYS)}


def submissions(
    cik: int = AAPL_CIK, name: str = "Apple Inc.", fye: str = "0926", rows=None, files=()
):
    return {
        "cik": str(cik),
        "name": name,
        "fiscalYearEnd": fye,
        "tickers": ["AAPL"],
        "filings": {
            "recent": page(rows if rows is not None else AAPL_RECENT),
            "files": list(files),
        },
    }


@dataclass
class FakeEdgar:
    """Routes EDGAR URLs to canned responses and records every request."""

    submissions_by_cik: dict[int, dict] = field(default_factory=dict)
    extra_pages: dict[str, dict] = field(default_factory=dict)
    failing_docs: set[str] = field(default_factory=set)
    status_override: int | None = None
    requests: list[httpx.Request] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.submissions_by_cik.setdefault(AAPL_CIK, submissions())

    def urls(self, contains: str) -> list[str]:
        return [str(r.url) for r in self.requests if contains in str(r.url)]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if self.status_override is not None:
            return httpx.Response(self.status_override, text="blocked")
        if url == "https://www.sec.gov/files/company_tickers.json":
            return httpx.Response(
                200,
                json={
                    "0": {"cik_str": AAPL_CIK, "ticker": "AAPL", "title": "Apple Inc."},
                    "1": {"cik_str": NVDA_CIK, "ticker": "NVDA", "title": "NVIDIA CORP"},
                },
            )
        if url.startswith("https://data.sec.gov/submissions/CIK"):
            cik = int(url.rsplit("CIK", 1)[1].split("-")[0].removesuffix(".json"))
            if url.endswith("-submissions-001.json"):
                name = url.rsplit("/", 1)[1]
                return httpx.Response(200, json=self.extra_pages[name])
            return httpx.Response(200, json=self.submissions_by_cik[cik])
        if url.startswith("https://www.sec.gov/Archives/edgar/data/"):
            doc = url.rsplit("/", 1)[1]
            if doc in self.failing_docs:
                return httpx.Response(500, text="server error")
            return httpx.Response(200, content=f"<html>{doc}</html>".encode() * 100)
        return httpx.Response(404, text="not found")

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)
