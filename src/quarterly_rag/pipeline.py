"""Retrieve, gate, generate, verify, gate again: the whole `rag ask` path (RAG-011).

The control flow is plain Python on purpose (ADR-003), so the order of the checks is
readable and every branch is testable without a model.
"""

from __future__ import annotations

from dataclasses import dataclass

from quarterly_rag.chunking.build import iter_chunks
from quarterly_rag.config import Settings
from quarterly_rag.generation.answer import answer_question
from quarterly_rag.generation.base import LLM
from quarterly_rag.generation.refusal import (
    CorpusScope,
    GateOutcome,
    GateSettings,
    check_answer,
    check_retrieval,
    check_scope,
)
from quarterly_rag.retrieval.base import Retriever

TICKERS = ("AAPL", "NVDA")


@dataclass
class Pipeline:
    retriever: Retriever
    llm: LLM
    scope: CorpusScope
    gate: GateSettings
    max_tokens: int = 1024

    @classmethod
    def build(
        cls,
        settings: Settings,
        retriever: Retriever,
        llm: LLM,
        *,
        gate: GateSettings | None = None,
        strategy: str = "fixed",
    ) -> Pipeline:
        scope = CorpusScope.from_chunks(
            c for ticker in TICKERS for c in iter_chunks(settings, ticker, strategy)
        )
        return cls(
            retriever=retriever,
            llm=llm,
            scope=scope,
            gate=gate or GateSettings(min_retrieval_score=settings.min_retrieval_score),
            max_tokens=settings.answer_max_tokens,
        )

    def ask(self, question: str, k: int = 5, where: dict | None = None) -> GateOutcome:
        # Stage 1, before spending a model call: can the corpus hold this answer at all?
        if refusal := check_scope(question, self.scope, self.gate):
            return GateOutcome(refusal=refusal)

        results = self.retriever.retrieve(question, k=k, where=where)
        if refusal := check_retrieval(results, self.gate):
            return GateOutcome(refusal=refusal, results=list(results))

        answer = answer_question(
            self.llm, question, [r.chunk for r in results], max_tokens=self.max_tokens
        )
        # Stage 2: the model has read the passages, and the verifier has read the model.
        if refusal := check_answer(answer, results):
            return GateOutcome(refusal=refusal, answer=answer, results=list(results))
        return GateOutcome(answer=answer, results=list(results))
