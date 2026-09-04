from __future__ import annotations

from collections.abc import Sequence

import pytest

from quarterly_rag.generation.answer import (
    DEFAULT_PROMPT_VERSION,
    INSUFFICIENT,
    Answer,
    answer_question,
    load_prompt,
    parse_tags,
    render_passages,
    split_sentences,
    verify,
)
from quarterly_rag.generation.base import ChatMessage, ChatResponse

TABLE = "(In millions)\nTotal net sales | 109,417 | 94,036"
PROSE = "As of September 27, 2025, the Company had approximately 166,000 full-time employees."


class FakeLLM:
    label = "fake/llm"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[ChatMessage] = []

    def chat(self, messages: Sequence[ChatMessage], *, temperature=0.0, max_tokens=1024):
        self.messages = list(messages)
        return ChatResponse(text=self.reply, model="fake", stop_reason="stop")

    def list_models(self) -> list[str]:
        return ["fake"]


@pytest.fixture
def passages(make_chunk):
    return [
        make_chunk("a:1-2", TABLE),
        make_chunk("b:1-2", PROSE, section="Part I.Item 1", period_label="FY2025", form="10-K"),
    ]


def test_tags_are_parsed_in_every_shape_a_model_writes() -> None:
    assert parse_tags("Net sales rose [c1].") == ["c1"]
    assert parse_tags("Both [c1][c3].") == ["c1", "c3"]
    assert parse_tags("Both [c1, c3].") == ["c1", "c3"]
    assert parse_tags("Both [c2; c4].") == ["c2", "c4"]
    assert parse_tags("Repeated [c1][c1].") == ["c1"]
    assert parse_tags("None here.") == []


def test_sentences_split_on_terminators() -> None:
    assert split_sentences("One [c1]. Two [c2].") == ["One [c1].", "Two [c2]."]
    assert split_sentences("Only one.") == ["Only one."]
    assert split_sentences("   ") == []


def test_a_citation_after_the_full_stop_stays_with_its_sentence() -> None:
    # Models place the label on either side of the stop; both mean the same thing.
    assert split_sentences("One. [c1] Two. [c2]") == ["One. [c1]", "Two. [c2]"]
    assert split_sentences("One. [c1]") == ["One. [c1]"]
    assert split_sentences("One. [c1] Two [c2]. Three. [c3]") == [
        "One. [c1]",
        "Two [c2].",
        "Three. [c3]",
    ]


def test_full_width_brackets_count_as_citations(passages) -> None:
    # gpt-oss writes \u3010c1\u3011; scoring that as uncited would blame the model for the parser.
    assert parse_tags("Sales were $109,417 million \u3010c1\u3011.") == ["c1"]
    answer = verify("Sales were $109,417 million \u3010c1\u3011.", passages)
    assert answer.fully_grounded
    assert [c.tag for c in answer.citations] == ["c1"]


def test_a_cited_verified_sentence_passes_clean(passages) -> None:
    answer = verify("Total net sales were $109,417 million [c1].", passages, model="m")
    assert answer.fully_grounded
    assert answer.text == "Total net sales were $109,417 million [c1]."
    assert [c.tag for c in answer.citations] == ["c1"]
    assert answer.citations[0].chunk_id == "a:1-2"
    assert answer.citations[0].period_label == "FY2026 Q3"
    assert answer.model == "m"
    assert answer.prompt_version == DEFAULT_PROMPT_VERSION


def test_an_uncited_sentence_is_unsupported(passages) -> None:
    answer = verify("Net sales were $109,417 million.", passages)
    assert answer.unsupported_sentences == ["Net sales were $109,417 million."]
    assert "[unsupported: uncited]" in answer.text
    assert answer.citations == []
    assert not answer.fully_grounded


def test_a_citation_to_a_passage_never_provided_is_caught(passages) -> None:
    answer = verify("Net sales were $109,417 million [c9].", passages)
    assert answer.invalid_tags == ["c9"]
    assert answer.unsupported_sentences
    assert "was not provided" in answer.text
    assert answer.citations == []


def test_a_figure_absent_from_the_cited_passage_is_derived_not_a_lie(passages) -> None:
    answer = verify("Sales rose by $15,381 million [c1].", passages)
    assert [d.text for d in answer.derived_numbers] == ["$15,381 million"]
    assert answer.derived_numbers[0].cited_tags == ["c1"]
    assert "[derived, unverified: $15,381 million]" in answer.text
    assert answer.unsupported_sentences == []  # the sentence is kept, just labelled
    assert not answer.fully_grounded


def test_the_citation_label_is_not_read_as_a_figure(passages) -> None:
    # "[c1]" would otherwise contribute the number 1 and be flagged as unverified.
    answer = verify("Employment grew last year [c2].", passages)
    assert answer.derived_numbers == []
    assert answer.fully_grounded


def test_a_sentence_citing_two_passages_checks_both(passages) -> None:
    answer = verify("Sales were $109,417 million and headcount was 166,000 [c1][c2].", passages)
    assert answer.derived_numbers == []
    assert [c.tag for c in answer.citations] == ["c1", "c2"]


def test_the_insufficient_evidence_signal_is_preserved(passages) -> None:
    answer = verify(INSUFFICIENT, passages)
    assert answer.insufficient_evidence
    assert answer.citations == []
    assert answer.unsupported_sentences == []
    # Lowercase and trailing text still count as the signal.
    assert verify(f"{INSUFFICIENT}\n", passages).insufficient_evidence


def test_no_passages_means_no_model_call(passages) -> None:
    llm = FakeLLM("should never be used")
    answer = answer_question(llm, "What were net sales?", [])
    assert answer.insufficient_evidence
    assert llm.messages == []


def test_the_prompt_is_sent_with_tagged_passages(passages) -> None:
    llm = FakeLLM("Net sales were $109,417 million [c1].")
    answer = answer_question(llm, "What were net sales?", passages)

    system, user = llm.messages
    assert system.content == load_prompt()
    assert "[c1] (In millions)" in user.content
    assert "[c2] As of September 27" in user.content
    assert user.content.endswith("Question: What were net sales?")
    assert answer.fully_grounded
    assert answer.model == "fake/llm"


def test_passages_are_tagged_from_one(passages) -> None:
    rendered = render_passages(passages)
    assert rendered.startswith("[c1] ")
    assert "\n\n[c2] " in rendered


def test_answer_round_trips_through_json(passages) -> None:
    answer = verify("Sales rose by $15,381 million [c9]. Net sales were 109,417 [c1].", passages)
    assert Answer.model_validate_json(answer.model_dump_json()) == answer


# --- calculation provenance (RAG-021) ----------------------------------------------


def test_a_derived_number_backed_by_a_calculation_is_verified(passages) -> None:
    """The presence check cannot confirm 15,381: no passage prints it. The arithmetic can."""
    answer = verify(
        "Net sales rose $15,381 million [c1].\nCALC: 109,417 [c1] - 94,036 [c1] = 15,381",
        passages,
    )
    assert answer.fully_grounded
    assert [d.text for d in answer.verified_derived] == ["$15,381 million"]
    assert answer.unverified_derived == []
    assert "[derived, verified: $15,381 million]" in answer.text
    assert answer.calculations[0].verified


def test_a_derived_number_with_no_calculation_stays_unverified(passages) -> None:
    answer = verify("Net sales rose $15,381 million [c1].", passages)
    assert not answer.fully_grounded
    assert [d.text for d in answer.unverified_derived] == ["$15,381 million"]
    assert "[derived, unverified: $15,381 million]" in answer.text


def test_a_wrong_calculation_leaves_its_figure_unverified(passages) -> None:
    """Two real operands, the wrong subtraction: exactly what a presence check waves through."""
    answer = verify(
        "Net sales rose $20,000 million [c1].\nCALC: 109,417 [c1] - 94,036 [c1] = 20,000",
        passages,
    )
    assert not answer.fully_grounded
    assert [d.text for d in answer.unverified_derived] == ["$20,000 million"]
    assert answer.derived_numbers[0].calculation is not None
    assert not answer.derived_numbers[0].calculation.verified


def test_calculation_lines_are_not_sentences(passages) -> None:
    """A CALC line is working, not a claim; judging it as one would distort faithfulness."""
    answer = verify(
        "Net sales rose $15,381 million [c1].\nCALC: 109,417 [c1] - 94,036 [c1] = 15,381",
        passages,
    )
    assert answer.prose == "Net sales rose $15,381 million [c1]."
    assert answer.unsupported_sentences == []


def test_a_calculation_citing_a_missing_passage_is_a_resolution_failure(passages) -> None:
    answer = verify(
        "Net sales rose $15,381 million [c1].\nCALC: 109,417 [c1] - 94,036 [c9] = 15,381",
        passages,
    )
    assert answer.invalid_tags == ["c9"]
    assert not answer.fully_grounded


def test_a_calculation_adds_the_passages_it_cites_to_the_citations(passages) -> None:
    answer = verify(
        "Employees grew [c2].\nCALC: 109,417 [c1] - 94,036 [c1] = 15,381",
        passages,
    )
    assert [c.tag for c in answer.citations] == ["c2", "c1"]


def test_the_prompt_version_is_chosen_and_recorded(passages) -> None:
    llm = FakeLLM("Net sales were $109,417 million [c1].")
    answer = answer_question(llm, "What were net sales?", passages, prompt_version="1")
    system, _ = llm.messages
    assert system.content == load_prompt("1")
    assert "CALC:" not in system.content
    assert "CALC:" in load_prompt("2")
    assert answer.prompt_version == "1"


def test_an_unknown_prompt_version_fails_loudly() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("99")


def test_a_truncated_answer_is_flagged_rather_than_blamed(passages) -> None:
    """A calculation cut off by the budget parses as unparsed; the budget did that."""
    answer = verify("Net sales were $109,417 million [c1].", passages, stop_reason="length")
    assert answer.truncated
    assert not verify("Net sales were $109,417 million [c1].", passages).truncated
