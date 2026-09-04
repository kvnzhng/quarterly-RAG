from __future__ import annotations

from datetime import date

import pytest

from quarterly_rag.ingestion.fiscal import fiscal_period


@pytest.mark.parametrize(
    ("fye", "report_date", "form", "label"),
    [
        # Dates pulled live from EDGAR on 2026-09-04; 52/53-week years overshoot the nominal end.
        ("0926", date(2025, 9, 27), "10-K", "FY2025"),
        ("0926", date(2024, 9, 28), "10-K", "FY2024"),
        ("0926", date(2025, 12, 27), "10-Q", "FY2026 Q1"),
        ("0926", date(2024, 3, 30), "10-Q", "FY2024 Q2"),
        ("0926", date(2026, 6, 27), "10-Q", "FY2026 Q3"),
        ("0131", date(2026, 1, 25), "10-K", "FY2026"),
        ("0131", date(2026, 4, 26), "10-Q", "FY2027 Q1"),
        ("0131", date(2026, 7, 26), "10-Q", "FY2027 Q2"),
        ("0131", date(2025, 10, 26), "10-Q", "FY2026 Q3"),
        # Calendar-year company.
        ("1231", date(2025, 12, 31), "10-K", "FY2025"),
        ("1231", date(2025, 3, 31), "10-Q", "FY2025 Q1"),
        ("1231", date(2025, 9, 30), "10-Q", "FY2025 Q3"),
    ],
)
def test_labels_match_the_companies_own_naming(fye, report_date, form, label) -> None:
    period = fiscal_period(report_date, fye, form)
    assert period.label == label


def test_annual_report_has_no_quarter() -> None:
    assert fiscal_period(date(2025, 9, 27), "0926", "10-K").quarter is None
    assert fiscal_period(date(2025, 9, 27), "0926", "10-K/A").quarter is None


def test_bad_fiscal_year_end_rejected() -> None:
    with pytest.raises(ValueError, match="MMDD"):
        fiscal_period(date(2025, 1, 1), "", "10-K")
