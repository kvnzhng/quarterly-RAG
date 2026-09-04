from __future__ import annotations

from collections.abc import Sequence

import pytest

from quarterly_rag.errors import ModelServerError
from quarterly_rag.evaluation.calibration import calibrate
from quarterly_rag.evaluation.judge import (
    ClaimJudgement,
    FaithfulnessResult,
    Judge,
    mean_faithfulness,
    parse_correctness,
    parse_verdict,
)
from quarterly_rag.generation.answer import verify
from quarterly_rag.generation.base import ChatResponse


class ScriptedLLM:
    label = "fake/judge"

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def chat(self, messages, *, temperature=0.0, max_tokens=1024):
        self.prompts.append(messages[-1].content)
        reply = self.replies.pop(0) if self.replies else "SUPPORTED"
        return ChatResponse(text=reply, model="fake", stop_reason="stop")

    def list_models(self):
        return ["fake"]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("SUPPORTED", "supported"),
        ("supported", "supported"),
        ("Supported.", "supported"),
        ("NOT_SUPPORTED", "not_supported"),
        ("not supported", "not_supported"),
        ("NOT-SUPPORTED", "not_supported"),
        # A thinking model restates the options before choosing, so the last one wins.
        ("The choice is SUPPORTED or NOT_SUPPORTED. Answer: NOT_SUPPORTED", "not_supported"),
        ("maybe", "unparsed"),
        ("", "unparsed"),
    ],
)
def test_a_verdict_is_read_from_the_last_word_that_matters(reply, expected) -> None:
    assert parse_verdict(reply) == expected


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("CORRECT", "correct"),
        ("partial", "partial"),
        ("INCORRECT", "incorrect"),
        ("hmm", "unparsed"),
    ],
)
def test_correctness_is_parsed_the_same_way(reply, expected) -> None:
    assert parse_correctness(reply) == expected


def test_faithfulness_judges_each_sentence_against_only_its_own_citation(make_chunk) -> None:
    chunks = [
        make_chunk("a:1-2", "(In millions)\nTotal net sales | 109,417"),
        make_chunk("b:1-2", "The Company had approximately 166,000 employees."),
    ]
    answer = verify("Net sales were $109,417 million [c1]. Headcount was 166,000 [c2].", chunks)
    llm = ScriptedLLM(["SUPPORTED", "NOT_SUPPORTED"])
    result = Judge(llm).faithfulness(answer, {"c1": chunks[0].text, "c2": chunks[1].text})

    assert [c.verdict for c in result.claims] == ["supported", "not_supported"]
    assert result.score == pytest.approx(0.5)
    # The first prompt sees only c1: judging against the whole context is the thing this
    # metric deliberately does not do.
    assert "109,417" in llm.prompts[0]
    assert "166,000" not in llm.prompts[0]


def test_an_uncited_sentence_is_not_judged(make_chunk) -> None:
    chunk = make_chunk("a:1-2", "(In millions)\nTotal net sales | 109,417")
    answer = verify("Net sales were $109,417 million. Something else [c1].", [chunk])
    llm = ScriptedLLM(["SUPPORTED"])
    result = Judge(llm).faithfulness(answer, {"c1": chunk.text})
    # The verifier already labels an uncited sentence; asking a model would add noise.
    assert len(result.claims) == 1
    assert len(llm.prompts) == 1


def test_an_unparsed_verdict_counts_against_the_score() -> None:
    result = FaithfulnessResult(
        [
            ClaimJudgement("a", ["c1"], "supported"),
            ClaimJudgement("b", ["c1"], "unparsed"),
        ]
    )
    assert result.unparsed == 1
    assert result.score == pytest.approx(0.5)


def test_an_endpoint_failure_is_an_unparsed_verdict_not_a_crash(make_chunk) -> None:
    class Broken(ScriptedLLM):
        def chat(self, messages, *, temperature=0.0, max_tokens=1024):
            raise ModelServerError("endpoint down")

    chunk = make_chunk("a:1-2", "Total net sales | 109,417")
    answer = verify("Net sales were 109,417 [c1].", [chunk])
    result = Judge(Broken([])).faithfulness(answer, {"c1": chunk.text})
    assert [c.verdict for c in result.claims] == ["unparsed"]


def test_an_answer_with_no_claims_scores_one() -> None:
    assert FaithfulnessResult([]).score == 1.0
    assert mean_faithfulness([FaithfulnessResult([])]) == 0.0  # nothing to average


def test_mean_faithfulness_ignores_answers_with_no_claims() -> None:
    scored = FaithfulnessResult([ClaimJudgement("a", ["c1"], "supported")])
    unscored = FaithfulnessResult([])
    assert mean_faithfulness([scored, unscored]) == pytest.approx(1.0)


def test_correctness_asks_with_the_reference_answer() -> None:
    llm = ScriptedLLM(["CORRECT"])
    verdict = Judge(llm).correctness(
        "What were net sales?", "$109,417 million", "About $109.4 billion."
    )
    assert verdict == "correct"
    assert "$109,417 million" in llm.prompts[0]
    assert "$109.4 billion" in llm.prompts[0]


def test_calibration_separates_the_safe_and_dangerous_disagreements() -> None:
    claims = [
        ClaimJudgement("verified and supported", ["c1"], "supported"),
        ClaimJudgement("verified but objected to", ["c1"], "not_supported"),
        ClaimJudgement("unverified and waved through", ["c1"], "supported"),
        ClaimJudgement("unverified and caught", ["c1"], "not_supported"),
    ]
    cal = calibrate(claims, {"verified and supported", "verified but objected to"})

    assert cal.agrees_supported == 1
    assert cal.agrees_unsupported == 1
    assert cal.judge_stricter == 1  # safe: it read beyond the figures
    assert cal.judge_looser == 1  # dangerous: it passed an unverified figure
    assert cal.agreement == pytest.approx(0.5)
    assert cal.looser_rate == pytest.approx(0.5)
    assert cal.as_dict()["sentences"] == 4


def test_calibration_of_nothing_does_not_divide_by_zero() -> None:
    cal = calibrate([], set())
    assert cal.agreement == 0.0
    assert cal.looser_rate == 0.0
