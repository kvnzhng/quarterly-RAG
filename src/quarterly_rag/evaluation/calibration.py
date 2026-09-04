"""Is the judge any good? (RAG-012)

A judge is a model, and models are wrong. Before its numbers mean anything it is scored
against something that cannot be wrong: the deterministic number verifier from RAG-010,
which knows whether a figure appears in the passage a sentence cited.

That gives a partial ground truth. A sentence whose figures were all found should be
judged supported; a sentence whose figures were not found is at least suspect. Perfect
agreement is not expected and would be suspicious, because the verifier only checks
figures while the judge reads the whole claim. What matters is the direction and size of
the disagreement, and which way it falls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from quarterly_rag.evaluation.judge import ClaimJudgement


@dataclass(frozen=True)
class CalibrationCase:
    """One sentence, with what the verifier said and what the judge said."""

    sentence: str
    figures_verified: bool
    """True when every figure in the sentence was found in the passage it cited."""
    verdict: str


@dataclass
class Calibration:
    cases: list[CalibrationCase]

    def _count(self, verified: bool, verdict: str) -> int:
        return sum(c.figures_verified is verified and c.verdict == verdict for c in self.cases)

    @property
    def agrees_supported(self) -> int:
        """Figures verified and the judge says supported: both agree."""
        return self._count(True, "supported")

    @property
    def judge_stricter(self) -> int:
        """Figures verified and the judge says not supported.

        The safe disagreement: the judge read something beyond the figures and objected.
        """
        return self._count(True, "not_supported")

    @property
    def judge_looser(self) -> int:
        """Figures **not** verified and the judge says supported.

        The dangerous disagreement: the judge waved through a sentence whose figure is not
        in the passage it cited. A judge that does this often cannot be trusted to catch
        the failure this project exists to catch.
        """
        return self._count(False, "supported")

    @property
    def agrees_unsupported(self) -> int:
        return self._count(False, "not_supported")

    @property
    def verified(self) -> int:
        return sum(c.figures_verified for c in self.cases)

    @property
    def agreement(self) -> float:
        if not self.cases:
            return 0.0
        return (self.agrees_supported + self.agrees_unsupported) / len(self.cases)

    @property
    def looser_rate(self) -> float:
        """Share of unverified sentences the judge called supported."""
        unverified = len(self.cases) - self.verified
        return self.judge_looser / unverified if unverified else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "sentences": len(self.cases),
            "figures_verified": self.verified,
            "agree_supported": self.agrees_supported,
            "agree_unsupported": self.agrees_unsupported,
            "judge_stricter": self.judge_stricter,
            "judge_looser": self.judge_looser,
            "agreement": round(self.agreement, 4),
            "looser_rate": round(self.looser_rate, 4),
        }


def calibrate(claims: Sequence[ClaimJudgement], verified_sentences: set[str]) -> Calibration:
    return Calibration(
        [
            CalibrationCase(
                sentence=claim.sentence,
                figures_verified=claim.sentence in verified_sentences,
                verdict=claim.verdict,
            )
            for claim in claims
        ]
    )
