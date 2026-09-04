from __future__ import annotations

from pathlib import Path

import pytest

from quarterly_rag.ingestion.parse import (
    TABLE_CLOSE,
    TABLE_OPEN,
    parse_filing,
    render_table,
    to_text,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def tenq():
    return parse_filing((FIXTURES / "tenq.htm").read_text())


@pytest.fixture(scope="module")
def tenk():
    return parse_filing((FIXTURES / "tenk.htm").read_text())


def test_tenq_items_are_keyed_by_part(tenq) -> None:
    # Item 1 means Financial Statements in Part I and Legal Proceedings in Part II, so the
    # part is what makes the key unique.
    assert [s.key for s in tenq.sections] == [
        "Part I.Item 1",
        "Part I.Item 2",
        "Part II.Item 1A",
        "Part II.Item 6",
    ]
    assert tenq.sections[0].title == "Financial Statements"
    assert tenq.sections[1].part == 1
    assert tenq.sections[2].part == 2


def test_table_of_contents_rows_are_not_headings(tenq) -> None:
    # The contents table renders as "Item 1. | Financial Statements | 1", which looks like
    # a heading until you notice it is inside a table.
    assert len(tenq.sections) == 4
    assert "Item 1. | Financial Statements" in tenq.text  # the row is still in the text


def test_cross_references_are_not_headings(tenq) -> None:
    mda = next(s for s in tenq.sections if s.key == "Part I.Item 2")
    assert "See Item 1A of this Form 10-Q" in mda.text  # kept in the body, not split on
    risks = next(s for s in tenq.sections if s.key == "Part II.Item 1A")
    # This sentence starts its own line with "Part I, ...". A bare part-prefix match would
    # set the part back to I and misattribute every heading after it.
    assert risks.text.count("Part I, Item 1A of the 2025 Form 10-K") == 1
    assert tenq.sections[-1].part == 2


def test_offsets_index_into_the_normalized_text(tenq, tenk) -> None:
    for parsed in (tenq, tenk):
        for section in parsed.sections:
            assert parsed.text[section.char_start : section.char_end] == section.text
        starts = [s.char_start for s in parsed.sections]
        assert starts == sorted(starts)


def test_last_section_stops_at_the_signature_block(tenq) -> None:
    exhibits = tenq.sections[-1]
    assert "Exhibit 31.1" in exhibits.text
    assert "SIGNATURES" not in exhibits.text
    assert "Securities Exchange Act" not in exhibits.text


def test_tables_are_pipe_delimited_with_markers_and_tidy_numbers(tenq) -> None:
    financials = tenq.sections[0]
    assert TABLE_OPEN in financials.text and TABLE_CLOSE in financials.text
    assert "header: Three Months Ended" in financials.text
    assert "June 27, 2026 | June 28, 2025" in financials.text
    assert "Net sales | $109,417 | $94,036" in financials.text
    assert "Cost of sales | (54,647) | (50,318)" in financials.text


def test_tenk_covers_its_items_across_parts(tenk) -> None:
    assert [s.key for s in tenk.sections] == [
        "Part I.Item 1",
        "Part I.Item 1A",
        "Part II.Item 7",
        "Part II.Item 7A",
        "Part II.Item 8",
    ]
    coverage = tenk.coverage("10-K")
    assert coverage.ok
    assert coverage.missing_critical == []
    assert (1, "1") in coverage.found and (2, "8") in coverage.found
    assert (3, "10") in coverage.missing  # the fixture is truncated
    assert coverage.unexpected == []


def test_section_keys_use_real_roman_numerals() -> None:
    # Part IV is where Nvidia files its consolidated financial statements, so "Part IIII"
    # would be wrong on exactly the records that matter most.
    html = (
        "<html><body>"
        "<div>Part I</div><div>Item 1. Business</div><div>a</div>"
        "<div>Part II</div><div>Item 7. MD and A</div><div>b</div>"
        "<div>Part III</div><div>Item 10. Directors</div><div>c</div>"
        "<div>Part IV</div><div>Item 15. Exhibits and Schedules</div><div>d</div>"
        "</body></html>"
    )
    assert [s.key for s in parse_filing(html).sections] == [
        "Part I.Item 1",
        "Part II.Item 7",
        "Part III.Item 10",
        "Part IV.Item 15",
    ]


def test_coverage_flags_a_missing_critical_item() -> None:
    parsed = parse_filing("<html><body><div>Item 9B. Other Information</div></body></html>")
    coverage = parsed.coverage("10-K")
    assert not coverage.ok
    assert (1, "1") in coverage.missing_critical


def test_nvidia_style_10q_without_defaults_and_mine_safety_is_still_ok(tenq) -> None:
    coverage = tenq.coverage("10-Q")
    assert coverage.ok  # critical items present
    assert (2, "3") in coverage.missing  # absent, but not a failure
    assert coverage.unexpected == []


def test_empty_table_and_plain_html() -> None:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup("<table><tr><td></td><td> </td></tr></table>", "html.parser")
    assert render_table(soup.find("table")) == []
    assert to_text("<html><body><div>One</div><div>Two</div></body></html>") == "One\nTwo"


def test_paragraph_starting_with_item_is_not_a_heading() -> None:
    long_line = "Item 5. " + " ".join(f"word{i}" for i in range(40))
    parsed = parse_filing(f"<html><body><div>{long_line}</div></body></html>")
    assert parsed.sections == []
