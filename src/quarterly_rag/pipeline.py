"""Retrieve, gate, generate, verify, gate again: the whole `rag ask` path (RAG-011).

The control flow is plain Python on purpose (ADR-003), so the order of the checks is
readable and every branch is testable without a model.

Every stage is also a span (RAG-013). The tracer defaults to one that does nothing, so the
traced path and the untraced path are the same code and the tests exercise both. What each
span records is decided here rather than in `observability`, which is why that module knows
nothing about chunks, answers or refusals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quarterly_rag.chunking.build import iter_chunks
from quarterly_rag.config import Settings
from quarterly_rag.generation.answer import (
    DEFAULT_PROMPT_VERSION,
    Answer,
    no_passages,
    respond,
    verify_response,
)
from quarterly_rag.generation.base import LLM
from quarterly_rag.generation.refusal import (
    CorpusScope,
    GateOutcome,
    GateSettings,
    check_answer,
    check_retrieval,
    check_scope,
)
from quarterly_rag.observability.tracing import (
    GENERATION,
    RETRIEVER,
    NullTracer,
    Tracer,
    build_tracer,
    trace_metadata,
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
    prompt_version: str = DEFAULT_PROMPT_VERSION
    tracer: Tracer = field(default_factory=NullTracer)
    metadata: dict = field(default_factory=dict)
    """Run facts attached to every trace: models, prompt version, chunker, store."""

    @classmethod
    def build(
        cls,
        settings: Settings,
        retriever: Retriever,
        llm: LLM,
        *,
        gate: GateSettings | None = None,
        strategy: str = "fixed",
        tracer: Tracer | None = None,
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
            prompt_version=settings.answer_prompt_version,
            tracer=tracer if tracer is not None else build_tracer(settings),
            metadata=trace_metadata(settings, {"chunk_strategy": strategy}),
        )

    def ask(self, question: str, k: int = 5, where: dict | None = None) -> GateOutcome:
        with self.tracer.span(
            "rag ask", input={"question": question, "k": k}, metadata=self.metadata
        ) as trace:
            outcome = self._ask(question, k=k, where=where, trace_id=trace.trace_id)
            trace.update(output=_summary(outcome), level=_level(outcome))
            outcome.trace_id = trace.trace_id
            return outcome

    def _ask(self, question: str, *, k: int, where: dict | None, trace_id: str) -> GateOutcome:
        # Stage 1, before spending a model call: can the corpus hold this answer at all?
        with self.tracer.span("scope-gate", input={"question": question}) as span:
            refusal = check_scope(question, self.scope, self.gate)
            span.update(output={"refused": refusal is not None, "reason": _reason(refusal)})
        if refusal:
            return GateOutcome(refusal=refusal)

        with self.tracer.span(
            "retrieval",
            kind=RETRIEVER,
            input={"question": question},
            metadata={"retriever": self.retriever.name, "k": k, "filters": where},
        ) as span:
            results = self.retriever.retrieve(question, k=k, where=where)
            span.update(
                output={
                    "chunks": [r.chunk.chunk_id for r in results],
                    "top_score": max((r.score for r in results), default=None),
                }
            )

        with self.tracer.span("retrieval-gate") as span:
            refusal = check_retrieval(results, self.gate)
            span.update(
                output={"refused": refusal is not None, "reason": _reason(refusal)},
                metadata={"min_retrieval_score": self.gate.min_retrieval_score},
            )
        if refusal:
            return GateOutcome(refusal=refusal, results=list(results))

        chunks = [r.chunk for r in results]
        if not chunks:
            answer = no_passages(self.llm, self.prompt_version)
        else:
            answer = self._generate(question, chunks)

        # Stage 2: the model has read the passages, and the verifier has read the model.
        with self.tracer.span("answer-gate") as span:
            refusal = check_answer(answer, results)
            span.update(output={"refused": refusal is not None, "reason": _reason(refusal)})
        if refusal:
            return GateOutcome(refusal=refusal, answer=answer, results=list(results))
        return GateOutcome(answer=answer, results=list(results))

    def _generate(self, question: str, chunks: list) -> Answer:
        """The model call and the checking of it, timed apart (RAG-013)."""
        with self.tracer.span(
            "generation",
            kind=GENERATION,
            input={"question": question, "passages": len(chunks)},
            model=self.llm.label,
            model_parameters={"max_tokens": self.max_tokens, "temperature": 0.0},
            metadata={"prompt_version": self.prompt_version},
        ) as span:
            response = respond(
                self.llm,
                question,
                chunks,
                max_tokens=self.max_tokens,
                prompt_version=self.prompt_version,
            )
            span.update(output=response.text, usage_details=_usage(response))

        with self.tracer.span("verification", input={"passages": len(chunks)}) as span:
            answer = verify_response(
                response, chunks, model=self.llm.label, prompt_version=self.prompt_version
            )
            span.update(output=_verification(answer))
        return answer


def _usage(response) -> dict[str, int]:
    """Token counts, when the provider reported them. Absent beats guessed."""
    counts = {"input": response.input_tokens, "output": response.output_tokens}
    return {name: value for name, value in counts.items() if value is not None}


def _verification(answer: Answer) -> dict:
    return {
        "citations": len(answer.citations),
        "cited_sentences": answer.cited_sentences,
        "unsupported_sentences": len(answer.unsupported_sentences),
        "invalid_tags": answer.invalid_tags,
        "calculations": len(answer.calculations),
        "calculations_verified": sum(c.verified for c in answer.calculations),
        "derived_verified": len(answer.verified_derived),
        "derived_unverified": [d.text for d in answer.unverified_derived],
        "fully_grounded": answer.fully_grounded,
        "truncated": answer.truncated,
    }


def _summary(outcome: GateOutcome) -> dict:
    if outcome.refusal is not None:
        return {"refused": True, "reason": outcome.refusal.reason, "detail": outcome.refusal.detail}
    answer = outcome.answer
    if answer is None:  # pragma: no cover - an outcome always carries one or the other
        return {"refused": False}
    return {"refused": False, "answer": answer.text, **_verification(answer)}


def _level(outcome: GateOutcome) -> str:
    """A refusal is not an error, but it is the thing you scroll a trace list to find."""
    return "WARNING" if outcome.refused else "DEFAULT"


def _reason(refusal) -> str | None:
    return refusal.reason if refusal is not None else None
