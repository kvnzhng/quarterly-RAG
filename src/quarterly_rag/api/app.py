"""`POST /ask` over the same pipeline the CLI uses (RAG-014).

The pipeline is built once, in the lifespan, because building it reads every chunk on disk to
work out what the corpus covers. `/health` says 503 until that finishes.

The endpoint is a plain `def`, not `async def`, on purpose: answering calls a model and
blocks for seconds, and Starlette runs a sync endpoint in a threadpool instead of stalling
the event loop with it.

Refusing is a 200. A refusal is an answer to the question "can you answer this?", not a
failure of the request, and a client that treats it as an error would hide the reason.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from quarterly_rag.api.models import (
    MAX_EXCERPT_CHARS,
    MAX_PASSAGE_CHARS,
    AnswerOut,
    AskRequest,
    AskResponse,
    CalculationOut,
    CitationOut,
    HealthResponse,
    PassageOut,
    RefusalOut,
)
from quarterly_rag.config import Settings, get_settings
from quarterly_rag.errors import ModelServerError

log = logging.getLogger(__name__)

TITLE = "quarterly-RAG"
DESCRIPTION = (
    "Answers questions about SEC 10-Q and 10-K filings from the filings themselves, "
    "with every sentence cited and checked, or refuses and says why."
)


def build_pipeline(settings: Settings) -> Any:
    """The default factory. Imported lazily so `create_app` is cheap to import and to test."""
    from quarterly_rag.evaluation.refusal_eval import gate_settings
    from quarterly_rag.generation.llm import build_llm
    from quarterly_rag.indexing.build import build_store
    from quarterly_rag.indexing.embedder import build_embedder
    from quarterly_rag.pipeline import Pipeline
    from quarterly_rag.retrieval.build import build_retriever

    strategy = settings.chunk_strategy
    store = build_store(settings, settings.vector_store, strategy, "context")
    if store.count() == 0:
        raise RuntimeError("the index is empty; run `rag index build` first")
    llm = build_llm(settings)
    retriever = build_retriever(
        settings,
        settings.retrieval_strategy,
        embedder=build_embedder(settings),
        store=store,
        chunk_strategy=strategy,
        variant="context",
        llm=llm if settings.retrieval_strategy == "hybrid-rerank" else None,
    )
    return Pipeline.build(settings, retriever, llm, gate=gate_settings(settings), strategy=strategy)


def create_app(
    settings: Settings | None = None,
    pipeline_factory: Callable[[Settings], Any] | None = None,
) -> FastAPI:
    """The app, with the pipeline injectable so the endpoints are testable without a model."""
    settings = settings or get_settings()
    factory = pipeline_factory or build_pipeline

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        try:
            app.state.pipeline = factory(settings)
        except Exception:
            # A missing index must not stop the server: /health then says so, which is more
            # useful than a process that will not start and a stack trace in a terminal.
            log.exception("could not build the pipeline; /ask will answer 503")
            app.state.pipeline = None
        yield
        pipeline = getattr(app.state, "pipeline", None)
        if pipeline is not None:
            pipeline.tracer.flush()

    app = FastAPI(title=TITLE, description=DESCRIPTION, lifespan=lifespan)
    app.state.settings = settings

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        pipeline = getattr(request.app.state, "pipeline", None)
        if pipeline is None:
            raise HTTPException(status_code=503, detail="the pipeline is not built")
        return HealthResponse(
            status="ok",
            model=pipeline.llm.label,
            prompt_version=pipeline.prompt_version,
            tracing=pipeline.tracer.enabled,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: Request, body: AskRequest) -> AskResponse:
        pipeline = getattr(request.app.state, "pipeline", None)
        if pipeline is None:
            raise HTTPException(status_code=503, detail="the pipeline is not built")
        where = {"ticker": body.ticker.upper()} if body.ticker else None
        try:
            outcome = pipeline.ask(body.question, k=body.k, where=where)
        except ModelServerError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from None
        pipeline.tracer.flush()
        return to_response(outcome)

    return app


def to_response(outcome: Any) -> AskResponse:
    """Map a `GateOutcome` onto the wire contract."""
    trace_id = getattr(outcome, "trace_id", "") or ""
    if outcome.refusal is not None:
        return AskResponse(refusal=_refusal(outcome.refusal), trace_id=trace_id)
    return AskResponse(answer=_answer(outcome.answer, outcome.results), trace_id=trace_id)


def _answer(answer: Any, results: list) -> AnswerOut:
    passages = {r.chunk.chunk_id: r.chunk.text for r in results}
    return AnswerOut(
        text=answer.text,
        prose=answer.prose,
        citations=[_citation(c, passages) for c in answer.citations],
        calculations=[
            CalculationOut(raw=c.raw, verified=c.verified, reason=c.reason, computed=c.computed)
            for c in answer.calculations
        ],
        unsupported_sentences=list(answer.unsupported_sentences),
        verified_derived=[d.text for d in answer.verified_derived],
        unverified_derived=[d.text for d in answer.unverified_derived],
        invalid_tags=list(answer.invalid_tags),
        fully_grounded=answer.fully_grounded,
        truncated=answer.truncated,
        model=answer.model,
        prompt_version=answer.prompt_version,
    )


def _citation(citation: Any, passages: dict[str, str]) -> CitationOut:
    """The cited passage in full, so the UI can show where a figure came from.

    `Citation.quote` is 200 characters, which is not enough to see a table. The whole chunk
    is joined back in from the retrieved results and capped.
    """
    text = passages.get(citation.chunk_id) or citation.quote
    return CitationOut(
        tag=citation.tag,
        chunk_id=citation.chunk_id,
        ticker=citation.ticker,
        form=citation.form,
        period_label=citation.period_label,
        section=citation.section,
        source_url=citation.source_url,
        text=text[:MAX_PASSAGE_CHARS],
    )


def _refusal(refusal: Any) -> RefusalOut:
    return RefusalOut(
        reason=refusal.reason,
        detail=refusal.detail,
        best_chunks=[
            PassageOut(
                ticker=hit.chunk.ticker,
                form=hit.chunk.form,
                period_label=hit.chunk.period_label,
                section=hit.chunk.section,
                source_url=hit.chunk.source_url,
                score=hit.score,
                excerpt=" ".join(hit.chunk.text.split())[:MAX_EXCERPT_CHARS],
            )
            for hit in refusal.best_chunks
        ],
    )
