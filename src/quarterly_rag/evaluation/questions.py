"""The evaluation set: questions with human-verified evidence spans (RAG-019).

Gold evidence is a **span into the parsed filing text**, never a chunk id. Chunk ids
change with every chunking strategy; character offsets into `data/processed/<TICKER>/
<accession>.txt` do not. A chunk counts as relevant when it overlaps a span, so one
labelled set scores every chunker, store, and retriever that follows.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from quarterly_rag.config import Settings

QuestionType = Literal["lookup", "derived", "cross_period", "unanswerable"]
RefusalReason = Literal["out_of_scope", "insufficient_evidence"]

QUESTIONS_FILE = "questions.jsonl"


class EvidenceSpan(BaseModel):
    accession: str
    section: str = Field(description="Section key from the parser, e.g. 'Part I.Item 2'")
    char_start: int
    char_end: int
    quote: str = Field(description="The exact text at those offsets, so drift is detectable")

    @model_validator(mode="after")
    def _ordered(self) -> EvidenceSpan:
        if self.char_start >= self.char_end:
            raise ValueError(
                f"char_start must precede char_end ({self.char_start}, {self.char_end})"
            )
        return self


class EvalQuestion(BaseModel):
    id: str
    question: str
    ticker: str
    type: QuestionType
    gold_answer: str
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    refusal_reason: RefusalReason | None = None
    note: str = ""
    """Why this question is interesting, for the reader of the eval set."""

    @model_validator(mode="after")
    def _evidence_matches_type(self) -> EvalQuestion:
        if self.type == "unanswerable":
            if self.evidence:
                raise ValueError(f"{self.id}: an unanswerable question carries no evidence")
            if self.refusal_reason is None:
                raise ValueError(f"{self.id}: an unanswerable question needs a refusal_reason")
        elif not self.evidence:
            raise ValueError(f"{self.id}: an answerable question needs at least one evidence span")
        return self


@dataclass(frozen=True)
class SpanCheck:
    question_id: str
    span: EvidenceSpan
    ok: bool
    detail: str


def questions_path(settings: Settings) -> Path:
    return settings.eval_dir / QUESTIONS_FILE


def load_questions(path: Path) -> list[EvalQuestion]:
    return [
        EvalQuestion.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_questions(path: Path, questions: Iterable[EvalQuestion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(q.model_dump_json(exclude_defaults=False) for q in questions)
    path.write_text(body + "\n", encoding="utf-8")


def filing_text(settings: Settings, ticker: str, accession: str) -> str:
    return (settings.processed_dir / ticker.upper() / f"{accession}.txt").read_text(
        encoding="utf-8"
    )


def check_spans(settings: Settings, questions: Iterable[EvalQuestion]) -> list[SpanCheck]:
    """Every span must resolve to its recorded quote in the parsed text.

    This is what catches a re-parse that shifted offsets: the quote no longer matches, so
    the label is stale rather than silently wrong.
    """
    checks: list[SpanCheck] = []
    cache: dict[tuple[str, str], str | None] = {}
    for question in questions:
        for span in question.evidence:
            key = (question.ticker.upper(), span.accession)
            if key not in cache:
                try:
                    cache[key] = filing_text(settings, *key)
                except OSError as exc:
                    cache[key] = None
                    checks.append(
                        SpanCheck(question.id, span, False, f"{exc.__class__.__name__}: {exc}")
                    )
                    continue
            text = cache[key]
            if text is None:
                checks.append(SpanCheck(question.id, span, False, "filing text unavailable"))
                continue
            if span.char_end > len(text):
                checks.append(
                    SpanCheck(
                        question.id, span, False, f"span ends past the text ({len(text)} chars)"
                    )
                )
                continue
            found = text[span.char_start : span.char_end]
            if found != span.quote:
                checks.append(
                    SpanCheck(question.id, span, False, f"quote drifted; text holds {found[:60]!r}")
                )
                continue
            checks.append(SpanCheck(question.id, span, True, span.quote[:60]))
    return checks


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[A-Za-z][A-Za-z&/'-]{3,}")
# Words a gold answer may add for readability without being in the filing.
_CONNECTIVES = frozenset(
    {
        "about",
        "approximately",
        "billion",
        "million",
        "thousand",
        "from",
        "than",
        "that",
        "this",
        "these",
        "those",
        "with",
        "were",
        "have",
        "into",
        "over",
        "under",
        "down",
        "less",
        "more",
        "each",
        "both",
        "only",
        "also",
        "which",
        "their",
        "there",
        "notes",
        "filings",
        "corpus",
        "holds",
        "point-in-time",
        "document",
        "carries",
        "market",
        "reports",
        "statement",
        "reference",
        "guidance",
        "part",
        "volumes",
        "dollars",
        "stopped",
        "disclosing",
        "executive",
        "compensation",
        "proxy",
        "incorporates",
        "forward-looking",
        "annual",
        "report",
        "starts",
        "live",
        "data",
        "kind",
    }
)


def check_gold_answers(questions: Iterable[EvalQuestion]) -> list[tuple[str, str]]:
    """A `lookup` answer must be grounded in that question's own evidence.

    Numbers must match exactly: the filing's table says `Total net sales | 109,417` and the
    answer says "$109,417 million" because the unit lives in the caption, so the string as a
    whole cannot be required, but every figure in it can be. This is the same rule the
    citation verifier applies in RAG-010.

    For an answer with no figures, every content word must appear in the evidence. That
    allows a list assembled from a passage ("Compute & Networking and Graphics") while still
    rejecting an answer invented outside the source.

    Only `lookup`: a derived or cross-period answer is computed from its evidence, and an
    unanswerable one has none.
    """
    problems: list[tuple[str, str]] = []
    for question in questions:
        if question.type != "lookup":
            continue
        evidence = " ".join(span.quote for span in question.evidence)
        numbers = set(_NUMBER.findall(evidence))
        missing = [n for n in _NUMBER.findall(question.gold_answer) if n not in numbers]
        if missing:
            problems.append(
                (question.id, f"numbers {missing} in the gold answer are not in the evidence")
            )
            continue
        if _NUMBER.search(question.gold_answer):
            continue
        lowered = evidence.lower()
        unseen = [
            word
            for word in _WORD.findall(question.gold_answer)
            if word.lower() not in _CONNECTIVES and word.lower() not in lowered
        ]
        if unseen:
            problems.append(
                (question.id, f"words {unseen} in the gold answer are not in the evidence")
            )
    return problems


def counts_by_type(questions: Iterable[EvalQuestion]) -> dict[str, int]:
    tally: dict[str, int] = {}
    for question in questions:
        tally[question.type] = tally.get(question.type, 0) + 1
    return tally


def iter_answerable(questions: Iterable[EvalQuestion]) -> Iterator[EvalQuestion]:
    return (q for q in questions if q.type != "unanswerable")
