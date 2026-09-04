"""LLM-as-judge: faithfulness and answer correctness (RAG-012).

Two judgements, deliberately separate.

**Faithfulness** asks whether each sentence is supported by *the passage it cited*, not by
the retrieved context as a whole. That is the distinction from RAGAS and from most
published faithfulness metrics: an answer that states a true fact while citing the wrong
passage is unfaithful here and faithful there, and citing the wrong passage is exactly the
failure a reader cannot detect for themselves.

**Correctness** asks whether the answer matches the labelled one. It replaces the
`states the gold figure` proxy from RAG-010, which required the same figure written the
same way and so failed a correct answer phrased at a different scale.

A judge is a model and models are wrong, so `calibration.py` scores this one against the
deterministic number verifier before any of its numbers are believed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from quarterly_rag.errors import ModelServerError
from quarterly_rag.generation.answer import Answer, split_sentences
from quarterly_rag.generation.base import LLM, ChatMessage

JUDGE_PROMPT_VERSION = "1"
Verdict = Literal["supported", "not_supported", "unparsed"]
Correctness = Literal["correct", "partial", "incorrect", "unparsed"]

FAITHFULNESS_SYSTEM = (
    "You check whether a claim is supported by a passage from an SEC filing.\n"
    "Reply with exactly one word: SUPPORTED or NOT_SUPPORTED.\n"
    "SUPPORTED means the passage states the claim, or states figures the claim reports "
    "faithfully.\n"
    "NOT_SUPPORTED means the passage does not state it, contradicts it, or the claim adds "
    "something the passage does not say. A claim that is true in general but absent from "
    "the passage is NOT_SUPPORTED."
)
CORRECTNESS_SYSTEM = (
    "You compare an answer with a reference answer to a question about SEC filings.\n"
    "Reply with exactly one word: CORRECT, PARTIAL or INCORRECT.\n"
    "CORRECT means the answer states the same fact as the reference, even if it is worded "
    "differently or uses a different unit for the same amount.\n"
    "PARTIAL means it states part of the fact, or is right but incomplete.\n"
    "INCORRECT means it states something different, or does not answer."
)
MAX_TOKENS = 2048
"""Room for a thinking-mode model to reason before it writes the one word."""

_SUPPORTED = re.compile(r"\bNOT[_\s-]?SUPPORTED\b|\bSUPPORTED\b", re.I)
_CORRECTNESS = re.compile(r"\b(CORRECT|PARTIAL|INCORRECT)\b", re.I)


def parse_verdict(reply: str) -> Verdict:
    """Last match wins: a thinking model restates the options before choosing."""
    matches = _SUPPORTED.findall(reply or "")
    if not matches:
        return "unparsed"
    return "not_supported" if re.match(r"(?i)not", matches[-1].strip()) else "supported"


def parse_correctness(reply: str) -> Correctness:
    matches = _CORRECTNESS.findall(reply or "")
    if not matches:
        return "unparsed"
    return matches[-1].lower()  # type: ignore[return-value]


@dataclass(frozen=True)
class ClaimJudgement:
    sentence: str
    cited_tags: list[str]
    verdict: Verdict


@dataclass
class FaithfulnessResult:
    claims: list[ClaimJudgement]

    @property
    def supported(self) -> int:
        return sum(c.verdict == "supported" for c in self.claims)

    @property
    def unparsed(self) -> int:
        return sum(c.verdict == "unparsed" for c in self.claims)

    @property
    def score(self) -> float:
        """Supported claims over all claims. An unparsed verdict counts against, because a
        judgement nobody can read is not evidence of support."""
        return self.supported / len(self.claims) if self.claims else 1.0


class Judge:
    """Wraps an `LLM`. Deliberately not the model that wrote the answer, by default."""

    def __init__(self, llm: LLM, *, max_tokens: int = MAX_TOKENS) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    @property
    def label(self) -> str:
        return getattr(self.llm, "label", "")

    def _ask(self, system: str, user: str) -> str:
        try:
            return self.llm.chat(
                [
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=user),
                ],
                max_tokens=self.max_tokens,
            ).text
        except ModelServerError:
            return ""

    def faithfulness(self, answer: Answer, passages: dict[str, str]) -> FaithfulnessResult:
        """Judge every cited sentence against the passages that sentence cites.

        Uncited sentences are not judged here: the verifier already labels them
        unsupported, and asking a model to confirm that would only add noise.
        """
        from quarterly_rag.generation.answer import parse_tags

        claims: list[ClaimJudgement] = []
        for sentence in split_sentences(answer.raw_text.strip() or answer.text):
            tags = [t for t in parse_tags(sentence) if t in passages]
            if not tags:
                continue
            cited = "\n\n".join(f"[{t}] {passages[t]}" for t in tags)
            reply = self._ask(FAITHFULNESS_SYSTEM, f"Passage:\n{cited}\n\nClaim: {sentence}")
            claims.append(ClaimJudgement(sentence, tags, parse_verdict(reply)))
        return FaithfulnessResult(claims)

    def correctness(self, question: str, gold: str, answer: str) -> Correctness:
        reply = self._ask(
            CORRECTNESS_SYSTEM,
            f"Question: {question}\n\nReference answer: {gold}\n\nAnswer: {answer}",
        )
        return parse_correctness(reply)


def mean_faithfulness(results: Sequence[FaithfulnessResult]) -> float:
    scored = [r for r in results if r.claims]
    return sum(r.score for r in scored) / len(scored) if scored else 0.0
