"""Fiscal year and quarter labels from a report date and the company's fiscal year end.

Apple's year ends on the last Saturday of September and Nvidia's on the last Sunday of
January, so period dates drift a few days around the nominal `fiscalYearEnd` (MMDD) that
EDGAR reports; a 53-week year can run six days past it. Hence the tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

FYE_TOLERANCE = timedelta(days=10)
QUARTER_DAYS = 365.25 / 4


@dataclass(frozen=True)
class FiscalPeriod:
    year: int
    quarter: int | None
    """1 to 4 for a quarterly report, None for an annual one."""

    @property
    def label(self) -> str:
        return f"FY{self.year}" if self.quarter is None else f"FY{self.year} Q{self.quarter}"


def _nominal_end(year: int, fiscal_year_end: str) -> date:
    month, day = int(fiscal_year_end[:2]), int(fiscal_year_end[2:])
    try:
        return date(year, month, day)
    except ValueError:  # 0229 in a non-leap year
        return date(year, month, 28)


def fiscal_period(report_date: date, fiscal_year_end: str, form: str) -> FiscalPeriod:
    """`fiscal_year_end` is EDGAR's MMDD string; `form` decides annual vs quarterly."""
    if not (len(fiscal_year_end) == 4 and fiscal_year_end.isdigit()):
        raise ValueError(f"fiscal_year_end must be MMDD, got {fiscal_year_end!r}")
    end_this_year = _nominal_end(report_date.year, fiscal_year_end)
    year = (
        report_date.year if report_date <= end_this_year + FYE_TOLERANCE else report_date.year + 1
    )
    if form.startswith("10-K"):
        return FiscalPeriod(year, None)
    fy_start = _nominal_end(year - 1, fiscal_year_end)
    days_into_year = (report_date - fy_start).days
    quarter = min(max(round(days_into_year / QUARTER_DAYS), 1), 4)
    return FiscalPeriod(year, quarter)
