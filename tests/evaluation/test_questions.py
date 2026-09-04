from __future__ import annotations

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from quarterly_rag.cli import app
from quarterly_rag.config import Settings
from quarterly_rag.evaluation.questions import (
    EvalQuestion,
    EvidenceSpan,
    check_gold_answers,
    check_spans,
    counts_by_type,
    load_questions,
    questions_path,
    save_questions,
)

TEXT = (
    "Item 1. Financial Statements\n"
    "(In millions)\n"
    "[TABLE]\n"
    "header: Three Months Ended\n"
    "Jun 27, 2026 | Jun 28, 2025\n"
    "Total net sales | 109,417 | 94,036\n"
    "[/TABLE]\n"
)
ACCESSION = "0000320193-26-000020"


def seed_text(settings: Settings, ticker: str = "AAPL", text: str = TEXT) -> tuple[int, int]:
    path = settings.processed_dir / ticker / f"{ACCESSION}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    start = text.index("Total net sales")
    return start, start + len("Total net sales | 109,417 | 94,036")


def make_question(start: int, end: int, quote: str, **overrides) -> EvalQuestion:
    fields = {
        "id": "q001",
        "question": "What were Apple's total net sales in the third quarter of fiscal 2026?",
        "ticker": "AAPL",
        "type": "lookup",
        "gold_answer": "$109,417 million",
        "evidence": [
            EvidenceSpan(
                accession=ACCESSION,
                section="Part I.Item 1",
                char_start=start,
                char_end=end,
                quote=quote,
            )
        ],
    }
    return EvalQuestion(**(fields | overrides))


def test_spans_resolve_against_the_parsed_text(settings: Settings) -> None:
    start, end = seed_text(settings)
    question = make_question(start, end, TEXT[start:end])
    checks = check_spans(settings, [question])
    assert [c.ok for c in checks] == [True]
    assert check_gold_answers([question]) == []


def test_a_reparse_that_shifts_offsets_is_caught(settings: Settings) -> None:
    start, end = seed_text(settings)
    question = make_question(start, end, TEXT[start:end])
    seed_text(settings, text="A new leading line.\n" + TEXT)  # every offset moves
    checks = check_spans(settings, [question])
    assert not checks[0].ok
    assert "quote drifted" in checks[0].detail


def test_span_past_the_end_of_the_text_is_caught(settings: Settings) -> None:
    start, _ = seed_text(settings)
    question = make_question(start, 10_000, TEXT[start:])
    checks = check_spans(settings, [question])
    assert not checks[0].ok
    assert "past the text" in checks[0].detail


def test_missing_filing_text_is_reported_not_raised(settings: Settings) -> None:
    question = make_question(0, 5, "hello")
    checks = check_spans(settings, [question])
    assert not checks[0].ok
    assert "FileNotFoundError" in checks[0].detail


def test_gold_answer_numbers_must_be_in_the_evidence(settings: Settings) -> None:
    start, end = seed_text(settings)
    quote = TEXT[start:end]
    # The unit word lives in the table caption, so the whole string is not required verbatim.
    assert check_gold_answers([make_question(start, end, quote)]) == []
    wrong = make_question(start, end, quote, gold_answer="$999,999 million")
    problems = check_gold_answers([wrong])
    assert problems and "999,999" in problems[0][1]


def test_derived_answers_are_not_required_to_appear(settings: Settings) -> None:
    start, end = seed_text(settings)
    derived = make_question(start, end, TEXT[start:end], type="derived", gold_answer="about 16.4%")
    assert check_gold_answers([derived]) == []


def test_unanswerable_questions_carry_a_reason_and_no_evidence() -> None:
    question = EvalQuestion(
        id="q010",
        question="What was Tesla's total revenue in fiscal 2025?",
        ticker="TSLA",
        type="unanswerable",
        gold_answer="Not in the filings.",
        refusal_reason="out_of_scope",
    )
    assert question.evidence == []
    with pytest.raises(ValidationError, match="needs a refusal_reason"):
        EvalQuestion(id="q011", question="q", ticker="TSLA", type="unanswerable", gold_answer="no")
    with pytest.raises(ValidationError, match="carries no evidence"):
        EvalQuestion(
            id="q012",
            question="q",
            ticker="TSLA",
            type="unanswerable",
            gold_answer="no",
            refusal_reason="out_of_scope",
            evidence=[
                EvidenceSpan(accession=ACCESSION, section="x", char_start=0, char_end=1, quote="a")
            ],
        )


def test_answerable_question_needs_evidence() -> None:
    with pytest.raises(ValidationError, match="needs at least one evidence span"):
        EvalQuestion(id="q1", question="q", ticker="AAPL", type="lookup", gold_answer="1")


def test_backwards_span_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must precede"):
        EvidenceSpan(accession=ACCESSION, section="x", char_start=50, char_end=50, quote="")


def test_round_trip_through_jsonl(settings: Settings) -> None:
    start, end = seed_text(settings)
    questions = [make_question(start, end, TEXT[start:end])]
    path = questions_path(settings)
    save_questions(path, questions)
    assert load_questions(path) == questions
    assert counts_by_type(questions) == {"lookup": 1}


def test_cli_eval_check(monkeypatch, settings: Settings) -> None:
    monkeypatch.setattr("quarterly_rag.cli.get_settings", lambda: settings)
    result = CliRunner().invoke(app, ["eval", "check"])
    assert result.exit_code == 2
    assert "no eval set" in result.stdout

    start, end = seed_text(settings)
    save_questions(questions_path(settings), [make_question(start, end, TEXT[start:end])])
    result = CliRunner().invoke(app, ["eval", "check"])
    assert result.exit_code == 0, result.stdout
    assert "1/1 spans resolve" in result.stdout

    seed_text(settings, text="shifted\n" + TEXT)
    result = CliRunner().invoke(app, ["eval", "check"])
    assert result.exit_code == 1
    assert "0/1 spans resolve" in result.stdout
