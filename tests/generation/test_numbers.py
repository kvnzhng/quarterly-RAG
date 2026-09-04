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
