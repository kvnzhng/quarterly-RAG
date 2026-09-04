"""Reading a question for what it says about the corpus (RAG-009).

Two consumers, one parser. Retrieval turns a named company or fiscal period into a
metadata filter and into extra lexical terms; the refusal gate turns a company or period
the corpus does not hold into an `out_of_scope` reason (RAG-011). They must agree about
what a question means, so they share this.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

QUARTER_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4}
_QUARTER = re.compile(
    r"\b(first|second|third|fourth)\s+quarter\s+(?:of\s+)?(?:fiscal\s+)?(?:year\s+)?(\d{4})\b",
    re.I,
)
_SHORT_QUARTER = re.compile(
    r"\bq([1-4])\s*(?:of\s+)?(?:fy|fiscal\s*(?:year\s*)?)?\s*(\d{4})\b", re.I
)
_FISCAL_YEAR = re.compile(r"\bfiscal\s+(?:year\s+)?(\d{4})\b", re.I)
_FY = re.compile(r"\bfy\s?(\d{4})\b", re.I)
# Company words that identify a ticker. Kept explicit rather than derived, because a
# filter that guesses wrong silently removes the answer from consideration.
COMPANY_TICKERS: dict[str, str] = {
    "apple": "AAPL",
    "aapl": "AAPL",
    "iphone": "AAPL",
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "geforce": "NVDA",
}


@dataclass(frozen=True)
class QueryFacets:
    """What a question states about which filing it wants."""

    tickers: tuple[str, ...] = ()
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None

    @property
    def period_label(self) -> str | None:
        if self.fiscal_year is None:
            return None
        if self.fiscal_quarter is None:
            return f"FY{self.fiscal_year}"
        return f"FY{self.fiscal_year} Q{self.fiscal_quarter}"

    def as_filter(self) -> dict[str, object] | None:
        """A store filter, but only when it is safe.

        Only the ticker is used, and only when exactly one company is named. A period
        filter would exclude a filing that quotes the asked-for year for comparison, and a
        question naming two companies is answered from both.
        """
        if len(self.tickers) != 1:
            return None
        return {"ticker": self.tickers[0]}

    def lexical_terms(self) -> list[str]:
        """Extra tokens for a keyword index, spelled the way chunk metadata spells them.

        A question says "the third quarter of fiscal 2026" and the corpus says
        "FY2026 Q3"; without this the two never share a token.
        """
        terms = list(self.tickers)
        if label := self.period_label:
            terms.extend(label.split())
            if self.fiscal_quarter is None:
                terms.append(label)
        return terms


def parse_facets(question: str) -> QueryFacets:
    tickers = tuple(
        sorted(
            {t for word, t in COMPANY_TICKERS.items() if re.search(rf"\b{word}\b", question, re.I)}
        )
    )
    year = quarter = None
    for pattern, quarter_first in ((_QUARTER, True), (_SHORT_QUARTER, False)):
        if match := pattern.search(question):
            word, digits = match.groups()
            quarter = QUARTER_WORDS[word.lower()] if quarter_first else int(word)
            year = int(digits)
            break
    if year is None:
        for pattern in (_FISCAL_YEAR, _FY):
            if match := pattern.search(question):
                year = int(match.group(1))
                break
    return QueryFacets(tickers=tickers, fiscal_year=year, fiscal_quarter=quarter)


def expand(question: str) -> str:
    """The question plus the corpus's own spelling of any period or company it names."""
    terms = parse_facets(question).lexical_terms()
    return f"{question} {' '.join(terms)}" if terms else question
