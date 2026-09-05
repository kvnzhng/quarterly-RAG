"""The refusal gate: when the system should decline, and why (RAG-011).

Refusing is a feature, not a failure mode. A system that answers everything is wrong on
the questions it cannot answer, and a system that refuses everything is useless. The gate
sits in two places and names its reason, because the fix for each reason is different:
index more data, tune a threshold, or improve the prompt.

- `out_of_scope`      the corpus cannot contain the answer: a company or a period that
                      was never indexed, or a question filings do not answer at all
- `low_confidence`    retrieval found nothing that scores above the threshold
- `insufficient_evidence`  the generator read the passages and said they do not answer
- `verification_failed`    the answer arrived, and not one sentence survived verification
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from quarterly_rag.generation.answer import Answer
from quarterly_rag.retrieval.base import RetrievedChunk

RefusalReason = Literal[
    "out_of_scope", "low_confidence", "insufficient_evidence", "verification_failed"
]

DEFAULT_MIN_SCORE = 0.0
"""Off by default. The threshold is chosen from the sweep in `docs/learning/refusal.md`,
not guessed, and 0 means every retrieval result is allowed through to the generator."""

# Questions filings do not answer whatever the company: live market data, advice, and
# documents ADR-004 put out of scope.
_OUT_OF_SCOPE_TOPICS: tuple[tuple[str, str], ...] = (
    (r"\b(should i|would you recommend|is it a good|good investment|worth buying|buy or sell)\b",
     "asks for investment advice, which filings do not give"),
    (r"\b(share price|stock price|market cap|market capitali[sz]ation|trading at)\b",
     "asks for live market data, which a point-in-time filing does not carry"),
    (r"\b(earnings call|analyst call|press release|investor day|transcript)\b",
     "asks about a document outside the corpus (ADR-004: 10-Q and 10-K only)"),
    (r"\b(weather|temperature|forecast for today)\b",
     "is not a question about the filings"),
)  # fmt: skip


class Refusal(BaseModel):
    reason: RefusalReason
    detail: str
    best_chunks: list[RetrievedChunk] = Field(default_factory=list)
    """Returned even when refusing, so the reader can look for themselves."""


@dataclass(frozen=True)
class CorpusScope:
    """What the index actually contains, used to reject a question it cannot answer."""

    tickers: frozenset[str] = frozenset()
    company_words: frozenset[str] = frozenset()
    """Lowercased words from company names, e.g. `apple`, `nvidia`."""
    fiscal_years: frozenset[int] = frozenset()

    @classmethod
    def from_chunks(cls, chunks: Iterable) -> CorpusScope:
        tickers, words, years = set(), set(), set()
        for chunk in chunks:
            tickers.add(chunk.ticker.upper())
            years.add(chunk.fiscal_year)
            for word in re.findall(r"[A-Za-z]{3,}", chunk.company):
                if word.lower() not in {"inc", "corp", "corporation", "company", "the"}:
                    words.add(word.lower())
        return cls(frozenset(tickers), frozenset(words), frozenset(years))

    @property
    def year_range(self) -> tuple[int, int] | None:
        return (min(self.fiscal_years), max(self.fiscal_years)) if self.fiscal_years else None


# Companies a reader might plausibly ask about that are not in this corpus.
_KNOWN_OUTSIDERS = {
    "tesla": "TSLA", "microsoft": "MSFT", "amazon": "AMZN", "google": "GOOGL",
    "alphabet": "GOOGL", "meta": "META", "facebook": "META", "intel": "INTC",
    "amd": "AMD", "qualcomm": "QCOM", "broadcom": "AVGO", "samsung": "005930",
    "netflix": "NFLX", "oracle": "ORCL", "ibm": "IBM",
}  # fmt: skip
_YEAR_IN_QUESTION = re.compile(r"\b(?:fiscal\s+(?:year\s+)?)?((?:19|20)\d{2})\b", re.I)


@dataclass
class GateSettings:
    min_retrieval_score: float = DEFAULT_MIN_SCORE
    check_scope: bool = True
    check_topics: bool = True


def check_scope(question: str, scope: CorpusScope, settings: GateSettings) -> Refusal | None:
    """Reject before retrieval when the corpus cannot hold the answer.

    Deliberately conservative: it only fires on a company that is definitely absent or a
    fiscal year outside the indexed range, because a false refusal costs an answerable
    question and there is no recovering from it.
    """
    lowered = question.lower()
    if settings.check_topics:
        for pattern, why in _OUT_OF_SCOPE_TOPICS:
            if re.search(pattern, lowered):
                return Refusal(reason="out_of_scope", detail=f"The question {why}.")
    if not settings.check_scope:
        return None

    named_inside = any(word in lowered for word in scope.company_words)
    outsiders = [
        name for name, ticker in _KNOWN_OUTSIDERS.items()
        if re.search(rf"\b{re.escape(name)}\b", lowered) and ticker not in scope.tickers
    ]  # fmt: skip
    if outsiders and not named_inside:
        listed = ", ".join(sorted(scope.tickers))
        return Refusal(
            reason="out_of_scope",
            detail=f"{outsiders[0].title()} is not in the corpus, which holds {listed}.",
        )

    span = scope.year_range
    if span and named_inside:
        asked = [int(y) for y in _YEAR_IN_QUESTION.findall(question)]
        # A filing quotes prior years for comparison, so only a year below the earliest
        # indexed fiscal year is certainly absent.
        if asked and max(asked) < span[0]:
            return Refusal(
                reason="out_of_scope",
                detail=(
                    f"The corpus covers fiscal {span[0]} to {span[1]}; {max(asked)} is before it."
                ),
            )
    return None


def check_retrieval(results: Sequence[RetrievedChunk], settings: GateSettings) -> Refusal | None:
    if not results:
        return Refusal(reason="low_confidence", detail="Retrieval returned nothing.")
    best = max(r.score for r in results)
    if best < settings.min_retrieval_score:
        return Refusal(
            reason="low_confidence",
            detail=(
                f"The closest passage scored {best:.3f}, below the "
                f"{settings.min_retrieval_score:.3f} threshold."
            ),
            best_chunks=list(results[:3]),
        )
    return None


def check_answer(answer: Answer, results: Sequence[RetrievedChunk]) -> Refusal | None:
    """After generation: the model declined, or nothing it wrote survived verification."""
    if answer.insufficient_evidence:
        return Refusal(
            reason="insufficient_evidence",
            detail="The model read the passages and reported that they do not answer it.",
            best_chunks=list(results[:3]),
        )
    if not answer.cited_sentences:
        return Refusal(
            reason="verification_failed",
            detail=(
                "No sentence carried a citation that resolves to a passage the model was given."
            ),
            best_chunks=list(results[:3]),
        )
    return None


@dataclass
class GateOutcome:
    """What the pipeline decided, and everything needed to explain it."""

    refusal: Refusal | None = None
    answer: Answer | None = None
    results: list[RetrievedChunk] = field(default_factory=list)
    trace_id: str = ""
    """The Langfuse trace this question produced, empty when tracing is off (RAG-013).
    Carried so an eval can hang its score on the trace the answer came from."""

    @property
    def refused(self) -> bool:
        return self.refusal is not None

    @property
    def reason(self) -> str | None:
        return self.refusal.reason if self.refusal else None
