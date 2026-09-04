"""The calculation verifier (RAG-021).

Three checks, tested apart from each other: operands are cited, operands are printed in the
passage they cite, and the arithmetic gives the stated result. Every input here is a shape a
local model actually produced or is likely to.
"""

from __future__ import annotations

import pytest

from quarterly_rag.generation.calculations import (
    ARITHMETIC_MISMATCH,
    OPERAND_NOT_IN_PASSAGE,
    UNCITED_OPERAND,
    UNKNOWN_TAG,
    UNPARSED,
    Calculation,
    matching_calculation,
    parse_calculation,
    split_calculations,
    verify_calculation,
)
from quarterly_rag.generation.numbers import parse_figures

REVENUE = "Revenue (In millions)\nRevenue | 96,221 | 46,743\nGross margin | 54,770 | 43,000"
RATES = "Effective tax rate | 15.6% | 24.1%"


def check(line: str, passages: dict[str, str] | None = None) -> Calculation:
    return verify_calculation(parse_calculation(line), passages or {"c1": REVENUE})


def test_a_growth_rate_recomputed_from_two_cited_operands_verifies() -> None:
    calculation = check("CALC: (96,221 [c1] - 46,743 [c1]) / 46,743 [c1] * 100 = 105.8%")
    assert calculation.verified
    assert calculation.reason == ""
    assert [o.text for o in calculation.operands] == ["96,221", "46,743", "46,743", "100"]
    assert all(o.in_passage for o in calculation.operands if o.tag)
    assert calculation.tags == ["c1"]


def test_a_ratio_left_unmultiplied_still_matches_a_percentage() -> None:
    """`a / b = 50.1%` is the same claim as `a / b * 100 = 50.1%`, written shorter."""
    assert check("CALC: 54,770 [c1] / 109,417 [c1] = 50.1%", {"c1": "54,770 and 109,417"}).verified


def test_wrong_arithmetic_over_real_operands_fails() -> None:
    calculation = check("CALC: 96,221 [c1] - 46,743 [c1] = 50,000")
    assert not calculation.verified
    assert calculation.reason == ARITHMETIC_MISMATCH
    assert calculation.computed == 49478.0


def test_an_operand_the_passage_does_not_state_fails() -> None:
    calculation = check("CALC: 96,221 [c1] - 40,000 [c1] = 56,221")
    assert calculation.reason == OPERAND_NOT_IN_PASSAGE
    assert [o.in_passage for o in calculation.operands] == [True, False]


def test_an_uncited_operand_fails_even_when_the_arithmetic_is_right() -> None:
    """The hole this closes: a hallucinated figure smuggled in as a bare number."""
    calculation = check("CALC: 96,221 [c1] - 46,743 = 49,478")
    assert calculation.reason == UNCITED_OPERAND
    assert calculation.computed == 49478.0


@pytest.mark.parametrize("constant", ["100", "1000", "1,000,000"])
def test_scale_constants_need_no_citation(constant: str) -> None:
    calculation = parse_calculation(f"CALC: 96,221 [c1] / {constant} = 1")
    assert [o.tag for o in calculation.operands] == ["c1", ""]
    assert verify_calculation(calculation, {"c1": REVENUE}).reason != UNCITED_OPERAND


def test_a_calculation_citing_a_passage_that_was_not_provided_fails() -> None:
    assert check("CALC: 96,221 [c9] - 46,743 [c1] = 49,478").reason == UNKNOWN_TAG


def test_prose_on_the_calculation_line_is_unparsed_not_wrong() -> None:
    """An expression nobody can recompute is a different failure from a wrong one."""
    assert check("CALC: revenue roughly doubled year over year").reason == UNPARSED
    assert check("CALC: 96,221 [c1] minus 46,743 [c1] = 49,478").reason == UNPARSED


def test_division_by_zero_computes_nothing_rather_than_raising() -> None:
    calculation = parse_calculation("CALC: 96,221 [c1] / 0 [c1] = 0")
    assert calculation.computed is None
    assert not verify_calculation(calculation, {"c1": REVENUE}).verified


@pytest.mark.parametrize(
    "line",
    [
        "CALC: 96,221 [c1] \u00d7 100 = 9,622,100",  # multiplication sign
        "CALC: 96,221 [c1] x 100 = 9,622,100",
        "**CALC:** 96,221 [c1] * 100 = 9,622,100",
        "- CALC: 96,221 [c1] * 100 = 9,622,100",
        "CALC: 96,221 【c1】 * 100 = 9,622,100",
    ],
)
def test_the_shapes_models_actually_write_are_all_read(line: str) -> None:
    assert check(line).verified


def test_a_minus_sign_that_is_not_a_hyphen_is_still_subtraction() -> None:
    assert check("CALC: 96,221 [c1] \u2212 46,743 [c1] = 49,478").verified  # minus sign
    assert check("CALC: 96,221 [c1] \u2013 46,743 [c1] = 49,478").verified  # en dash


def test_a_result_in_billions_matches_operands_printed_in_millions() -> None:
    """The table says 96,221; the sentence says $49.5 billion. One figure, two units."""
    assert check("CALC: 96,221 [c1] - 46,743 [c1] = $49.5 billion").verified


def test_a_rounded_result_is_confirmed_by_the_unrounded_computation() -> None:
    """ "about 106%" claims a value between 105.5 and 106.5, and 105.85 is one."""
    assert check("CALC: (96,221 [c1] - 46,743 [c1]) / 46,743 [c1] * 100 = about 106%").verified
    assert check("CALC: (96,221 [c1] - 46,743 [c1]) / 46,743 [c1] * 100 = ≈ 106%").verified
    # The slack is the wider of half a unit of the last digit and 0.5% of the value, so a
    # point and a half out is a mismatch however the answer rounded.
    assert not check("CALC: (96,221 [c1] - 46,743 [c1]) / 46,743 [c1] * 100 = 107.4%").verified


def test_percentage_operands_subtract_as_written() -> None:
    calculation = check("CALC: 15.6% [c1] - 24.1% [c1] = -8.5%", {"c1": RATES})
    assert calculation.verified
    assert calculation.result_value == -8.5


def test_a_parenthesised_result_is_negative_as_a_filing_writes_it() -> None:
    assert check("CALC: 15.6% [c1] - 24.1% [c1] = (8.5)%", {"c1": RATES}).verified
    assert not check("CALC: 24.1% [c1] - 15.6% [c1] = (8.5)%", {"c1": RATES}).verified


def test_operands_carrying_their_own_unit_are_scaled_before_the_arithmetic() -> None:
    passage = "Net sales were $96.2 billion, up from $46.7 billion."
    assert check("CALC: $96.2 billion [c1] - $46.7 billion [c1] = $49.5 billion", {"c1": passage})


def test_prose_and_calculations_are_separated() -> None:
    prose, lines = split_calculations(
        "Revenue grew about 106% [c1].\nCALC: 96,221 [c1] / 46,743 [c1] = 2.06\nIt was strong [c1]."
    )
    assert prose == "Revenue grew about 106% [c1].\nIt was strong [c1]."
    assert lines == ["CALC: 96,221 [c1] / 46,743 [c1] = 2.06"]


def test_a_calculation_written_after_the_sentence_is_still_lifted_out() -> None:
    """Asked for a line of its own, a model sometimes appends it to the sentence instead."""
    prose, lines = split_calculations("Revenue grew [c1]. CALC: 96,221 [c1] - 46,743 [c1] = 49,478")
    assert prose == "Revenue grew [c1]."
    assert lines == ["CALC: 96,221 [c1] - 46,743 [c1] = 49,478"]


def test_several_calculations_are_read_independently() -> None:
    prose, lines = split_calculations(
        "Two numbers [c1].\nCALC: 96,221 [c1] - 46,743 [c1] = 49,478\nCALC: 96,221 [c1] * 2 = 1"
    )
    assert prose == "Two numbers [c1]."
    assert len(lines) == 2


def test_a_calculation_is_matched_to_the_figure_it_explains() -> None:
    calculations = [parse_calculation("CALC: 416,161 [c1] - 391,035 [c1] = 25,126")]
    (billions,) = parse_figures("up about $25.1 billion")
    assert matching_calculation(calculations, billions) is calculations[0]
    (unrelated,) = parse_figures("$1,234 million")
    assert matching_calculation(calculations, unrelated) is None


def test_only_a_verified_calculation_is_matched_when_asked_for_one() -> None:
    calculations = [verify_calculation(parse_calculation("CALC: 1 [c9] + 1 [c9] = 2"), {})]
    (two,) = parse_figures("2 units")
    assert matching_calculation(calculations, two) is calculations[0]
    assert matching_calculation(calculations, two, verified_only=True) is None
