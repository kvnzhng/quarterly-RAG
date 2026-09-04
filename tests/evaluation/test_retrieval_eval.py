"""The eval runner end to end with a scripted retriever, so no model is called."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from quarterly_rag.chunking.build import chunks_dir
from quarterly_rag.cli import app
from quarterly_rag.config import Settings
from quarterly_rag.evaluation.questions import (
    EvalQuestion,
    EvidenceSpan,
    questions_path,
    save_questions,
)
from quarterly_rag.evaluation.relevance import OverlapRule
from quarterly_rag.evaluation.retrieval_eval import run_retrieval_eval
from quarterly_rag.retrieval.base import RetrievedChunk

ACCESSION = "0000320193-26-000020"


class ScriptedRetriever:
    """Returns a fixed ranking of chunks, so metrics are checked against known answers."""

    def __init__(self, chunks) -> None:
        self.chunks = chunks
        self.asked: list[tuple[str, int]] = []

    @property
    def name(self) -> str:
        return "scripted"

    def retrieve(self, question: str, k: int = 5, where=None) -> list[RetrievedChunk]:
        self.asked.append((question, k))
        return [
            RetrievedChunk(chunk=c, score=1.0 - i * 0.05, retriever=self.name, rank=i + 1)
            for i, c in enumerate(self.chunks[:k])
        ]


@pytest.fixture
def corpus(settings: Settings, make_chunk):
    """Four chunks over one filing; only the third holds the gold span."""
    chunks = [
        make_chunk(
            f"{ACCESSION}:{start}-{start + 100}",
            f"passage {start}",
            char_start=start,
            char_end=start + 100,
        )
        for start in (0, 100, 200, 300)
    ]
    path = chunks_dir(settings, "fixed", "AAPL") / f"{ACCESSION}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(c.model_dump_json() for c in chunks) + "\n")

    save_questions(
        questions_path(settings),
        [
            EvalQuestion(
                id="q001",
                question="What were Apple's total net sales?",
                ticker="AAPL",
                type="lookup",
                gold_answer="$109,417 million",
                evidence=[
                    EvidenceSpan(
                        accession=ACCESSION,
                        section="Part I.Item 1",
                        char_start=150,
                        char_end=250,
                        quote="x" * 100,
                    )
                ],
            ),
            EvalQuestion(
                id="q002",
                question="What was Tesla's revenue?",
                ticker="TSLA",
                type="unanswerable",
                gold_answer="Not in the filings.",
                refusal_reason="out_of_scope",
            ),
        ],
    )
    return chunks


def test_scores_the_answerable_questions_and_excludes_the_rest(settings, corpus) -> None:
    # The gold span straddles chunks 2 and 3, so two chunks are relevant.
    retriever = ScriptedRetriever(corpus)
    report = run_retrieval_eval(settings, retriever, ks=(1, 3, 5))

    assert report.run.question_count == 1
    assert report.skipped_unanswerable == 1
    (scored,) = report.results
    assert scored.question_id == "q001"
    assert scored.first_relevant_rank == 2
    assert scored.relevant_retrieved == 2
    assert scored.relevant_in_corpus == 2
    assert scored.form == "10-Q"

    payload = report.as_dict()
    assert payload["overall"]["recall"] == {"@1": 0.0, "@3": 1.0, "@5": 1.0}
    # Both relevant chunks were found, but at ranks 2 and 3 rather than 1 and 2, so nDCG
    # is below 1: it grades the ordering, not just the presence.
    assert 0 < payload["overall"]["ndcg"]["@5"] < 1.0
    assert payload["overall"]["ndcg"]["@5"] == pytest.approx(0.6934, abs=1e-4)
    assert payload["overall"]["mrr"] == pytest.approx(0.5)  # first relevant at rank 2
    assert payload["excluded_unanswerable"] == 1
    assert payload["near_miss"]["@5"] == {"filing": 1.0, "section": 1.0, "chunk": 1.0}
    assert payload["near_miss"]["@1"]["chunk"] == 0.0


def test_a_stricter_overlap_rule_changes_the_verdict(settings, corpus) -> None:
    retriever = ScriptedRetriever(corpus)
    loose = run_retrieval_eval(settings, retriever, ks=(5,))
    strict = run_retrieval_eval(
        settings, ScriptedRetriever(corpus), ks=(5,), rule=OverlapRule(min_fraction=1.0)
    )
    assert loose.results[0].first_relevant_rank == 2
    # The gold span runs 150-250 and no single 100-character chunk contains all of it.
    assert strict.results[0].first_relevant_rank is None
    assert strict.run.overlap_rule.startswith("min_chars=1, min_fraction=1")


def test_the_retriever_is_asked_for_the_largest_k(settings, corpus) -> None:
    retriever = ScriptedRetriever(corpus)
    run_retrieval_eval(settings, retriever, ks=(1, 3, 10))
    assert retriever.asked == [("What were Apple's total net sales?", 10)]


def test_the_run_record_says_how_the_number_was_made(settings, corpus) -> None:
    report = run_retrieval_eval(settings, ScriptedRetriever(corpus), ks=(5,), variant="context")
    run = report.run
    assert run.retriever == "scripted"
    assert run.embed_variant == "context"
    assert run.chunk_words == settings.chunk_words
    assert run.parser_version
    assert len(run.corpus_hash) == 16  # no manifest here, so it hashes nothing, and says so
    assert len(run.eval_set_hash) == 16
    assert run.question_count == 1
    assert run.prompt_version is None
    assert isinstance(run.git_dirty, bool)


def test_the_report_is_written_as_json(settings, corpus) -> None:
    report = run_retrieval_eval(settings, ScriptedRetriever(corpus), ks=(5,))
    path = report.write(settings)
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["run_record"]["retriever"] == "scripted"
    assert payload["per_question"][0]["question_id"] == "q001"
    assert "retrieval-raw-" in path.name


def test_missing_chunks_is_a_clear_error(settings) -> None:
    save_questions(questions_path(settings), [])
    with pytest.raises(FileNotFoundError, match="rag chunk build"):
        run_retrieval_eval(settings, ScriptedRetriever([]))


def test_cli_reports_an_empty_index(monkeypatch, settings, corpus) -> None:
    monkeypatch.setattr("quarterly_rag.cli.get_settings", lambda: settings)
    result = CliRunner().invoke(app, ["eval", "retrieval", "-k", "5"])
    assert result.exit_code == 1
    assert "rag index build" in result.stdout
