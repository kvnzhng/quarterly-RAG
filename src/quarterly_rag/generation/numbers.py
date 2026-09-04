"""Deciding whether a figure in an answer really appears in the passage it cites (RAG-010).

Filings print `109,417` under a caption reading `(In millions)`, and an answer may say
`$109,417 million` or `$109.4 billion`. All three are the same figure, so a verifier that
compares strings calls a correct answer unsupported. This module compares values after
scaling, and reports anything it cannot match as derived rather than as a hallucination:
the model may have computed it correctly, and RAG-021 is where that gets checked.

Known limit: this asks whether a figure is *present* in the cited passage, not whether the
claim about it is true. A model that writes "a $15,381 million increase" passes when the
passage happens to contain 15,381 anywhere, including as an unrelated line item. Checking
the relationship rather than the presence needs the operands and the operation, which is
exactly what RAG-021 adds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Relative tolerance for a scaled comparison. 0.5% covers "$109.4 billion" against a
# table's "109,417" in millions; looser starts matching the neighbouring column.
TOLERANCE = 0.005

SCALES: dict[str, float] = {
    "thousand": 1e3,
    "thousands": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "billion": 1e9,
    "billions": 1e9,
    "trillion": 1e12,
}
# Scales to try when a passage names no unit at all.
FALLBACK_SCALES = (1.0, 1e3, 1e6)

_NUMBER = re.compile(
    r"(?P<currency>[$€£])?\s?(?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(?P<suffix>%|percent|thousand[s]?|million[s]?|billion[s]?|trillion)?",
    re.I,
)
_CAPTION_UNIT = re.compile(r"\(\s*(?:dollars |\$ )?in (thousand|million|billion)s?", re.I)
# Years and the day part of a date are prose, not amounts. Without this, "June 27, 2026"
# contributes the figure 27 and gets flagged as an unverified number.
_YEAR = re.compile(r"^(19|20)\d{2}$")
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_DATE_DAY_BEFORE = re.compile(rf"(?:{_MONTHS})\.?\s*$", re.I)
_DATE_DAY_AFTER = re.compile(r"^\s*(?:st|nd|rd|th)?\s*,?\s*(?:19|20)\d{2}\b")


@dataclass(frozen=True)
class Figure:
    raw: str
    value: float
    is_percent: bool
    scale: float
    """Multiplier implied by an adjacent unit word; 1.0 when none was written."""

    @property
    def absolute(self) -> float:
        return self.value * self.scale


def caption_scale(text: str) -> float | None:
    """The unit a table caption declares, e.g. `(In millions)`."""
    match = _CAPTION_UNIT.search(text)
    return SCALES[match.group(1).lower()] if match else None


def parse_figures(text: str) -> list[Figure]:
    """Figures worth verifying. Bare years are skipped; they are dates, not amounts."""
    figures: list[Figure] = []
    for match in _NUMBER.finditer(text):
        digits = match.group("digits")
        suffix = (match.group("suffix") or "").lower()
        if _YEAR.match(digits) and not suffix and not match.group("currency"):
            continue
        if not suffix and not match.group("currency") and _is_date_day(text, match):
            continue
        try:
            value = float(digits.replace(",", ""))
        except ValueError:  # pragma: no cover - the pattern cannot produce this
            continue
        is_percent = suffix in {"%", "percent"}
        figures.append(
            Figure(
                raw=match.group(0).strip(),
                value=value,
                is_percent=is_percent,
                scale=1.0 if is_percent else SCALES.get(suffix, 1.0),
            )
        )
    return figures


def _is_date_day(text: str, match: re.Match[str]) -> bool:
    """True for the 27 in `June 27, 2026`, which is a date and not an amount."""
    before = text[: match.start()]
    after = text[match.end() :]
    return bool(_DATE_DAY_BEFORE.search(before) or _DATE_DAY_AFTER.match(after))


def _close(left: float, right: float) -> bool:
    if left == right:
        return True
    largest = max(abs(left), abs(right))
    return largest > 0 and abs(left - right) / largest <= TOLERANCE


def figure_supported(figure: Figure, passage: str) -> bool:
    """True when the passage states this figure, allowing for how the unit is written."""
    passage_figures = parse_figures(passage)
    if figure.is_percent:
        # A percentage is written the same way everywhere; no scaling to reconcile.
        return any(p.is_percent and _close(p.value, figure.value) for p in passage_figures)

    declared = caption_scale(passage)
    candidates = [declared] if declared else list(FALLBACK_SCALES)
    for passage_figure in passage_figures:
        if passage_figure.is_percent:
            continue
        # Quoting the digits as the passage prints them is what the prompt asks for, so
        # `109,417` against a table of millions is a match, not a unit error.
        if _close(passage_figure.value, figure.value):
            return True
        # The passage's own unit word wins over the caption when it wrote one.
        scales = [passage_figure.scale] if passage_figure.scale != 1.0 else candidates
        if any(_close(passage_figure.value * scale, figure.absolute) for scale in scales):
            return True
    return False


def unsupported_figures(sentence: str, passages: list[str]) -> list[Figure]:
    """Figures in the sentence that none of its cited passages state."""
    return [f for f in parse_figures(sentence) if not any(figure_supported(f, p) for p in passages)]
