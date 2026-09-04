"""Checks the committed eval set against the parsed corpus. Deselected by default."""

from __future__ import annotations

import pytest

from quarterly_rag.config import get_settings
from quarterly_rag.evaluation.questions import (
    check_gold_answers,
    check_spans,
    load_questions,
    questions_path,
)

pytestmark = pytest.mark.integration


def test_every_committed_span_resolves() -> None:
    settings = get_settings()
    path = questions_path(settings)
    if not path.exists():
        pytest.skip("no eval set yet")
    questions = load_questions(path)
    assert questions
    failures = [c for c in check_spans(settings, questions) if not c.ok]
    assert not failures, "\n".join(f"{c.question_id}: {c.detail}" for c in failures)
    assert check_gold_answers(questions) == []
