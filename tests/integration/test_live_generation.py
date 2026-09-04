"""One grounded answer against the live model. Deselected by default."""

from __future__ import annotations

import pytest

from quarterly_rag.chunking.build import iter_chunks
from quarterly_rag.config import get_settings
from quarterly_rag.evaluation.questions import load_questions, questions_path
from quarterly_rag.evaluation.relevance import is_relevant
from quarterly_rag.generation.answer import answer_question
from quarterly_rag.generation.llm import build_llm

pytestmark = pytest.mark.integration


def test_a_grounded_answer_cites_only_passages_it_was_given() -> None:
    settings = get_settings()
    questions = {q.id: q for q in load_questions(questions_path(settings))}
    corpus = [c for t in ("AAPL", "NVDA") for c in iter_chunks(settings, t)]
    if not corpus:
        pytest.skip("no chunks; run rag chunk build")

    question = questions["q002"]  # a prose lookup: headcount, stated in one sentence
    chunks = [c for c in corpus if is_relevant(c, question)][:3]
    assert chunks

    answer = answer_question(
        build_llm(settings), question.question, chunks, max_tokens=settings.answer_max_tokens
    )
    if answer.insufficient_evidence:
        pytest.skip("the model declined; grounding is checked by rag eval generation")

    # Whatever it said, every label it used must be one it was handed.
    assert answer.invalid_tags == []
    provided = {f"c{i}" for i in range(1, len(chunks) + 1)}
    assert {c.tag for c in answer.citations} <= provided
    for citation in answer.citations:
        assert citation.chunk_id in {c.chunk_id for c in chunks}
        assert citation.source_url.startswith("https://www.sec.gov/")
    assert answer.prompt_version == "1"
