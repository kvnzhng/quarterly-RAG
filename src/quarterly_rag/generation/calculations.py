"""Recomputing a derived number from the operands it cites (RAG-021).

A presence check asks whether a figure appears in the passage it cites, which a growth
rate never does: it is computed from two figures that do. `numbers.py` therefore labels
every such figure `derived, unverified` and stops. This module is the next step.

The generator is asked to show its arithmetic, one line per computed number:

    CALC: (96,221 [c1] - 46,743 [c1]) / 46,743 [c1] * 100 = 105.8%

and this module checks three things independently:

1. **every operand is cited**, apart from the scale constants (100, 1000, a million);
2. **every operand is printed in the passage it cites**, through the same unit-aware
   presence check answers are held to;
3. **the arithmetic recomputes the stated result**, within the rounding the answer wrote.

Only then is the number `verified`. A growth rate whose operands are invented fails (2);
one whose operands are real but whose division is wrong fails (3).

**What this does not catch**: the wrong two *real* figures. If a passage prints four
quarters and the answer divides the wrong pair, both operands are present and the
arithmetic is sound, so this reports `verified` for a number answering a different
question. Choosing the right operands is what the judge's correctness score measures
(RAG-012), and the two layers are reported separately for that reason.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from quarterly_rag.generation.citations import CLOSE, OPEN, parse_tags
from quarterly_rag.generation.numbers import (
    SCALES,
    Figure,
    figure_supported,
    parse_figures,
    values_close,
)

_EMPHASIS = r"(?:\*\*|__)?"
CALC_PREFIX = re.compile(
    rf"^\s*(?:[-*•]\s*)?{_EMPHASIS}\s*CALC\s*{_EMPHASIS}\s*:\s*{_EMPHASIS}\s*", re.I
)
"""A calculation line. Models decorate it as `- CALC:` or `**CALC:**`, so both are read."""

CALC_START = re.compile(rf"(?:^|(?<=[\s.;:)\]]))\s*{_EMPHASIS}\s*CALC\s*{_EMPHASIS}\s*:", re.I)
"""Where a calculation begins, anywhere in a line. A model asked for a line of its own
sometimes writes the calculation after the sentence instead, and a calculation left inside
the prose would be judged as a claim."""

UNCITED_CONSTANTS = (1.0, 100.0, 1_000.0, 1_000_000.0, 1_000_000_000.0)
"""Scale constants an operand may be without a citation. Any other bare number is a figure
the answer never sourced, which is the hole this rule closes."""

RESULT_SCALES = (1.0, 1e3, 1e6, 1e9)
"""A table printing millions gives operands in millions, so a result written in billions is
the same number. The stated result is compared against the computation under each."""

TOLERANCE = 0.005
"""Relative slack on the recomputation, matched to the presence check's."""

VERIFIED = ""
UNPARSED = "unparsed"
UNCHECKED = "not checked against any passage"
UNKNOWN_TAG = "cites a passage that was not provided"
UNCITED_OPERAND = "an operand with no citation"
OPERAND_NOT_IN_PASSAGE = "an operand is not in the passage it cites"
ARITHMETIC_MISMATCH = "the arithmetic does not give the stated result"

# What models write instead of ASCII arithmetic. Escaped rather than typed so the source
# has no characters that read as their ASCII lookalikes.
_OPERATOR_ALIASES = {
    "\u00d7": "*",  # multiplication sign
    "\u00f7": "/",  # division sign
    "\u2212": "-",  # minus sign
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2044": "/",  # fraction slash
}
_UNIT = r"%|(?:percent|thousands?|millions?|billions?|trillion)\b"
_TOKENS = re.compile(
    rf"(?P<space>\s+)"
    rf"|(?P<tag>[{OPEN}][^{CLOSE}]*[{CLOSE}])"
    rf"|(?P<number>[$€£]?\s?\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|[$€£]?\s?\d+(?:\.\d+)?)"
    rf"(?:[^\S\r\n]*(?P<unit>{_UNIT}))?"
    rf"|(?P<times>(?<=[\d\s)])x(?=[\s(\d$]))"
    rf"|(?P<op>[-+*/()])",
    re.I,
)
_APPROXIMATELY = re.compile(r"^\s*(?:about|approx(?:imately)?|roughly|around|~|≈)\s*", re.I)


class Operand(BaseModel):
    """One figure inside a calculation, with the passage it was taken from."""

    text: str
    """The operand as the answer wrote it, e.g. `46,743`."""
    value: float
    """Scaled by any unit word the operand itself carries, so `$46.7 billion` is 4.67e10."""
    tag: str = ""
    """The passage it cites; empty for an uncited number."""
    is_percent: bool = False
    in_passage: bool | None = None
    """Whether the cited passage states it. None when there was no passage to check."""


class Calculation(BaseModel):
    """One `CALC:` line, and what checking it found."""

    raw: str
    expression: str
    result_text: str
    operands: list[Operand] = Field(default_factory=list)
    result_value: float | None = None
    computed: float | None = None
    verified: bool = False
    reason: str = UNPARSED
    """Empty when verified, otherwise which of the three checks failed."""

    @property
    def tags(self) -> list[str]:
        seen: list[str] = []
        for operand in self.operands:
            if operand.tag and operand.tag not in seen:
                seen.append(operand.tag)
        return seen

    def rendered(self) -> str:
        mark = "verified" if self.verified else f"unverified: {self.reason}"
        if not self.verified and self.reason == ARITHMETIC_MISMATCH and self.computed is not None:
            mark = f"{mark} ({_format(self.computed)})"
        return f"{self.raw} [{mark}]"


def is_calculation(line: str) -> bool:
    return bool(CALC_PREFIX.match(line))


def split_calculations(text: str) -> tuple[str, list[str]]:
    """The prose, and the `CALC:` lines lifted out of it.

    Every consumer that reasons about sentences uses the prose: a calculation line is not a
    claim, and judging it as one both distorts faithfulness and pollutes the calibration
    the judge is measured by.
    """
    prose: list[str] = []
    calculations: list[str] = []
    for line in text.splitlines():
        head, found = _segment(line)
        if head.strip():
            prose.append(head.strip())
        calculations.extend(c.strip() for c in found if c.strip())
    return "\n".join(prose).strip(), calculations


def _segment(line: str) -> tuple[str, list[str]]:
    """The prose at the head of a line, then each calculation written on it."""
    starts = [m.start() for m in CALC_START.finditer(line)]
    if not starts:
        return line, []
    bounds = [*starts, len(line)]
    return line[: starts[0]], [line[bounds[i] : bounds[i + 1]] for i in range(len(starts))]


def parse_calculation(line: str) -> Calculation:
    """Read one `CALC:` line into operands, an expression and a stated result."""
    body = CALC_PREFIX.sub("", line).strip()
    left, _, right = body.rpartition("=")
    if not left.strip() or not right.strip():
        return Calculation(raw=line.strip(), expression=body, result_text="", reason=UNPARSED)

    calculation = Calculation(
        raw=line.strip(), expression=left.strip(), result_text=right.strip(), reason=UNPARSED
    )
    parsed = _tokenize(left)
    if parsed is None:
        return calculation
    expression, operands = parsed
    calculation.operands = operands
    calculation.result_value = _result_value(right)
    if calculation.result_value is None:
        return calculation
    calculation.computed = _evaluate(expression, operands)
    calculation.reason = UNCHECKED
    return calculation


def verify_calculation(
    calculation: Calculation,
    passages: Mapping[str, str],
    earlier: Sequence[Calculation] = (),
) -> Calculation:
    """Check the operands against their passages, then the arithmetic against the result.

    `earlier` is the calculations already verified in the same answer. An operand may be one
    of their results: working out a difference and then expressing it as a share of something
    is ordinary arithmetic, and refusing it forced a model to either restate a figure no
    passage prints or not show the second step at all. The chain is only as good as its first
    link, which is checked the same way (RAG-029).
    """
    if calculation.reason == UNPARSED:
        return calculation

    derived = [c.result_value for c in earlier if c.verified and c.result_value is not None]
    checked: list[Operand] = []
    reason = VERIFIED
    for operand in calculation.operands:
        if _is_earlier_result(operand, derived):
            checked.append(operand.model_copy(update={"in_passage": True}))
            continue
        if not operand.tag:
            if not _is_constant(operand.value):
                reason = reason or UNCITED_OPERAND
            checked.append(operand)
            continue
        if operand.tag not in passages:
            reason = reason or UNKNOWN_TAG
            checked.append(operand.model_copy(update={"in_passage": False}))
            continue
        figure = Figure(
            raw=operand.text,
            value=operand.value,
            is_percent=operand.is_percent,
            scale=1.0,
        )
        found = figure_supported(figure, passages[operand.tag], unitless_matches_percent=True)
        if not found:
            reason = reason or OPERAND_NOT_IN_PASSAGE
        checked.append(operand.model_copy(update={"in_passage": found}))

    verified = calculation.model_copy(update={"operands": checked})
    if reason:
        verified.reason = reason
        verified.verified = False
        return verified
    if not _result_matches(verified):
        verified.reason = ARITHMETIC_MISMATCH
        verified.verified = False
        return verified
    verified.reason = VERIFIED
    verified.verified = True
    return verified


def matching_calculation(
    calculations: list[Calculation], figure: Figure, *, verified_only: bool = False
) -> Calculation | None:
    """The calculation whose stated result is this figure, if one of them is.

    Scale is tried because a calculation over a table of millions yields `25,126` for a
    sentence that says `$25.1 billion`.
    """
    for calculation in calculations:
        if verified_only and not calculation.verified:
            continue
        if calculation.result_value is not None and _same_figure(calculation, figure):
            return calculation
    return None


def _is_earlier_result(operand: Operand, derived: Sequence[float]) -> bool:
    """Whether this operand is a figure one of the answer's own earlier lines produced."""
    return any(
        values_close(operand.value, value, tolerance=TOLERANCE) for value in derived if value
    )


def _same_figure(calculation: Calculation, figure: Figure) -> bool:
    stated = calculation.result_value
    if stated is None:
        return False
    slack = _rounding(calculation.result_text)
    if figure.is_percent or _is_percent(calculation.result_text):
        return values_close(stated, figure.value, tolerance=TOLERANCE, absolute=slack)
    return any(
        values_close(stated * scale, figure.absolute, tolerance=TOLERANCE, absolute=slack * scale)
        for scale in RESULT_SCALES
    )


def _tokenize(text: str) -> tuple[str, list[Operand]] | None:
    """Turn `(96,221 [c1] - 46,743 [c1])` into `(n0 - n1)` and its operands.

    Returns None when the line holds anything that is not a number, a citation, an
    arithmetic operator or a bracket: an expression nobody can recompute is unparsed, not
    wrong, and the two are reported separately.
    """
    normalised = text
    for symbol, replacement in _OPERATOR_ALIASES.items():
        normalised = normalised.replace(symbol, replacement)

    expression: list[str] = []
    operands: list[Operand] = []
    position = 0
    while position < len(normalised):
        match = _TOKENS.match(normalised, position)
        if match is None:
            return None
        position = match.end()
        if match.group("space"):
            continue
        if match.group("times"):
            expression.append("*")
            continue
        if match.group("op"):
            expression.append(match.group("op"))
            continue
        if match.group("tag"):
            tags = parse_tags(match.group("tag"))
            if not tags or not operands:
                return None
            if not operands[-1].tag:
                operands[-1] = operands[-1].model_copy(update={"tag": tags[0]})
            continue
        operand = _operand(match)
        if operand is None:
            return None
        expression.append(f"n{len(operands)}")
        operands.append(operand)
    if not operands:
        return None
    return " ".join(expression), operands


def _operand(match: re.Match[str]) -> Operand | None:
    digits = match.group("number")
    unit = (match.group("unit") or "").lower()
    try:
        value = float(re.sub(r"[^0-9.]", "", digits))
    except ValueError:  # pragma: no cover - the pattern cannot produce this
        return None
    is_percent = unit in {"%", "percent"}
    scale = 1.0 if is_percent else SCALES.get(unit, 1.0)
    return Operand(text=match.group(0).strip(), value=value * scale, is_percent=is_percent, tag="")


def _evaluate(expression: str, operands: list[Operand]) -> float | None:
    """Evaluate the normalised expression over its operands.

    The tree is walked rather than handed to `eval`: the string came from a language model,
    and arithmetic over the operands is the only thing it is allowed to express.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    names = {f"n{i}": operand.value for i, operand in enumerate(operands)}
    try:
        return _value_of(tree.body, names)
    except (ZeroDivisionError, OverflowError):
        return None


def _value_of(node: ast.AST, names: dict[str, float]) -> float | None:
    """One arithmetic node, or None for anything outside `+ - * /` over the operands."""
    if isinstance(node, ast.Constant):
        return float(node.value) if isinstance(node.value, int | float) else None
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.UnaryOp):
        inner = _value_of(node.operand, names)
        if inner is None:
            return None
        if isinstance(node.op, ast.USub):
            return -inner
        return inner if isinstance(node.op, ast.UAdd) else None
    if isinstance(node, ast.BinOp):
        left = _value_of(node.left, names)
        right = _value_of(node.right, names)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    return None


def _result_value(text: str) -> float | None:
    """The stated result, signed. `(8.5)%` is negative, as a filing writes it."""
    cleaned = _APPROXIMATELY.sub("", text.strip())
    figures = parse_figures(cleaned)
    if not figures:
        return None
    figure = figures[0]
    value = figure.value if figure.is_percent else figure.absolute
    negative = bool(re.match(r"^\s*[-\u2212]", cleaned)) or bool(
        re.match(r"^\s*\(\s*[$€£]?\s?[\d.,]+", cleaned)
    )
    return -value if negative else value


def _result_matches(calculation: Calculation) -> bool:
    computed, stated = calculation.computed, calculation.result_value
    if computed is None or stated is None:
        return False
    slack = _rounding(calculation.result_text)
    if _is_percent(calculation.result_text):
        # A ratio written as a percentage may or may not have been multiplied out.
        return any(
            values_close(candidate, stated, tolerance=TOLERANCE, absolute=slack)
            for candidate in (computed, computed * 100)
        )
    return any(
        values_close(computed * scale, stated, tolerance=TOLERANCE, absolute=slack)
        for scale in RESULT_SCALES
    )


def _rounding(result_text: str) -> float:
    """Half a unit of the last digit the answer wrote, scaled by any unit word.

    "about 106%" is a claim about a value between 105.5 and 106.5, so recomputing 105.85
    confirms it. Reading it as exactly 106 would fail a correct answer.
    """
    cleaned = _APPROXIMATELY.sub("", result_text.strip())
    match = re.search(r"\d[\d,]*(?:\.(?P<decimals>\d+))?", cleaned)
    if not match:
        return 0.0
    decimals = len(match.group("decimals") or "")
    unit = re.search(_UNIT, cleaned, re.I)
    word = unit.group(0).lower() if unit else ""
    scale = 1.0 if word in {"%", "percent"} else SCALES.get(word, 1.0)
    return 0.5 * (10.0**-decimals) * scale


def _is_percent(text: str) -> bool:
    return bool(re.search(r"%|\bpercent", text, re.I))


def _is_constant(value: float) -> bool:
    return any(values_close(value, constant, tolerance=0.0) for constant in UNCITED_CONSTANTS)


def _format(value: float) -> str:
    rounded = round(value, 4)
    return f"{rounded:,.4f}".rstrip("0").rstrip(".")
