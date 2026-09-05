from __future__ import annotations

import pytest

from quarterly_rag.generation.numbers import (
    caption_scale,
    figure_supported,
    parse_figures,
    unsupported_figures,
)

TABLE = (
    "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS\n"
    "(In millions, except per-share amounts)\n"
    "[TABLE]\nheader: Three Months Ended\n"
    "Total net sales | 109,417 | 94,036\n"
    "Gross margin | 54,770 | 43,718\n[/TABLE]"
)
PERCENTS = "Total gross margin percentage | 46.9% | 46.2% | 44.1%"


def test_a_figure_carries_its_unit_word() -> None:
    (figure,) = parse_figures("$109,417 million")
    assert figure.value == 109_417
    assert figure.absolute == 109_417_000_000
    assert not figure.is_percent

    (percent,) = parse_figures("46.9%")
    assert percent.is_percent
    assert percent.absolute == 46.9


def test_bare_years_are_not_figures() -> None:
    assert parse_figures("during 2025 compared to 2024") == []
    # A year with a currency or a unit is an amount, not a date.
    assert [f.raw for f in parse_figures("$2,025 million")] == ["$2,025 million"]


def test_the_day_in_a_date_is_not_an_amount() -> None:
    # "June 27, 2026" would otherwise contribute 27 and be flagged as an unverified figure.
    assert parse_figures("as of June 27, 2026") == []
    assert parse_figures("Jun 27, 2026 | Jun 28, 2025") == []
    assert [f.raw for f in parse_figures("the dividend was $0.27 per share")] == ["$0.27"]
    assert [f.raw for f in parse_figures("27 million shares")] == ["27 million"]
    assert [f.raw for f in parse_figures("September 27, 2025, about 166,000 employees")] == [
        "166,000"
    ]


def test_caption_declares_the_table_unit() -> None:
    assert caption_scale(TABLE) == 1e6
    assert caption_scale("(dollars in millions)") == 1e6
    assert caption_scale("(In thousands)") == 1e3
    assert caption_scale("no unit here") is None


@pytest.mark.parametrize(
    ("written", "supported"),
    [
        ("$109,417 million", True),  # same unit as the caption
        ("109,417", True),  # the digits exactly as printed
        ("$109.4 billion", True),  # scaled, within tolerance
        ("$54,770 million", True),  # a different row
        ("$15,381 million", False),  # a difference the table never states
        ("$110,000 million", False),  # close but outside tolerance
        ("46.9%", False),  # no percentages in this table
    ],
)
def test_a_figure_is_supported_when_the_passage_states_it(written: str, supported: bool) -> None:
    (figure,) = parse_figures(written)
    assert figure_supported(figure, TABLE) is supported


def test_percentages_match_percentages_only() -> None:
    (percent,) = parse_figures("46.9%")
    assert figure_supported(percent, PERCENTS)
    (other,) = parse_figures("50.1%")
    assert not figure_supported(other, PERCENTS)
    # 46.9 as a plain number is not the same claim as 46.9 percent.
    (plain,) = parse_figures("46.9")
    assert not figure_supported(plain, PERCENTS)


def test_a_passage_without_a_caption_tries_the_usual_scales() -> None:
    passage = "Revenue | 96,221 | 46,743"
    assert figure_supported(parse_figures("$96,221 million")[0], passage)
    assert figure_supported(parse_figures("96,221")[0], passage)
    assert not figure_supported(parse_figures("$500 million")[0], passage)


def test_unsupported_figures_checks_every_cited_passage() -> None:
    sentence = "Net sales were $109,417 million and the margin was 46.9%."
    assert [f.raw for f in unsupported_figures(sentence, [TABLE])] == ["46.9%"]
    assert unsupported_figures(sentence, [TABLE, PERCENTS]) == []


def test_a_unit_word_on_the_next_line_does_not_belong_to_this_number() -> None:
    """Apple's operating expenses table put `Percentage` on the row below `$29,915`.

    Reading the unit across the line break made that figure 29,915 percent, so a correct
    answer quoting it was reported as a figure the passage does not contain. Found while
    measuring RAG-021.
    """
    table = (
        "Research and development | $34,550 | 10% | $31,370 | 5% | $29,915\n"
        "Percentage of total net sales | 8% | 8% | 8%"
    )
    (last,) = (f for f in parse_figures(table) if f.value == 29915)
    assert not last.is_percent
    (quoted,) = parse_figures("$29,915 million")
    assert figure_supported(quoted, table)


def test_percentage_is_not_read_as_the_unit_percent() -> None:
    (figure,) = parse_figures("8 percentage points")
    assert not figure.is_percent
    (percent,) = parse_figures("8 percent")
    assert percent.is_percent


def test_a_narrow_no_break_space_still_attaches_the_unit() -> None:
    """`gpt-oss:20b` writes U+202F between a figure and its unit, on every figure it writes.

    Restricting the gap to a space or a tab dropped the unit from all of them, so the answer
    read as 109,417 rather than 109,417 million.
    """
    (figure,) = parse_figures("$109,417\u202fmillion")
    assert figure.absolute == 109_417_000_000.0
    assert figure_supported(figure, "(In millions)\nTotal net sales | 109,417")
    (non_breaking,) = parse_figures("$109,417\u00a0million")
    assert non_breaking.absolute == 109_417_000_000.0


def test_a_unitless_figure_matches_a_percentage_only_where_it_is_asked_to() -> None:
    """Prose and a calculation operand are held to different standards, on purpose.

    "The rate was 46.9" is not the claim the passage makes; `24.1` inside a `CALC:` line is
    a cell of the table quoted sloppily, and the arithmetic still has to come out (RAG-029).
    """
    (plain,) = parse_figures("46.9")
    assert not figure_supported(plain, PERCENTS)
    assert figure_supported(plain, PERCENTS, unitless_matches_percent=True)
    (dollars,) = parse_figures("$46.9 million")
    assert not figure_supported(dollars, PERCENTS, unitless_matches_percent=True)
