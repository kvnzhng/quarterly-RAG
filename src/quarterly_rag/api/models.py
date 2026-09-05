"""The shapes `POST /ask` returns (RAG-014).

Separate from the internal `Answer` and `Refusal` on purpose. Two reasons, both practical.

`Refusal.best_chunks` holds whole `RetrievedChunk` objects, each with a full `Chunk` inside,
so returning the internal model would ship several kilobytes of filing text per refusal
without anyone deciding to. And an HTTP contract that mirrors internal models breaks every
time those are refactored, which they have been twice already.

So this module is the contract, and `app.py` maps into it. Nothing here imports the
pipeline; the mapping goes the other way.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

MAX_PASSAGE_CHARS = 2000
"""Cap on a cited passage. Long enough for the UI to show the table a figure came from,
short enough that five citations do not make a 60 KB response."""

MAX_EXCERPT_CHARS = 300
"""Cap on a refusal's closest passages, which are a hint rather than evidence."""


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20, description="Passages to retrieve.")
    ticker: str | None = Field(default=None, description="Restrict to one company.")


class CitationOut(BaseModel):
    tag: str = Field(description="The `c1` label the answer used.")
    chunk_id: str
    ticker: str
    form: str
    period_label: str
    section: str
    source_url: str = Field(description="The filing on EDGAR, so a reader can check.")
    text: str = Field(description=f"The cited passage, capped at {MAX_PASSAGE_CHARS} characters.")


class CalculationOut(BaseModel):
    """One `CALC:` line and what recomputing it found (RAG-021)."""

    raw: str
    verified: bool
    reason: str = Field(description="Empty when verified, otherwise which check failed.")
    computed: float | None = None


class AnswerOut(BaseModel):
    text: str = Field(description="The answer with markers on anything that failed a check.")
    prose: str = Field(description="The same answer without its calculation lines.")
    citations: list[CitationOut] = Field(default_factory=list)
    calculations: list[CalculationOut] = Field(default_factory=list)
    unsupported_sentences: list[str] = Field(default_factory=list)
    verified_derived: list[str] = Field(default_factory=list)
    unverified_derived: list[str] = Field(default_factory=list)
    invalid_tags: list[str] = Field(default_factory=list)
    fully_grounded: bool
    truncated: bool = Field(description="The token budget cut the answer off.")
    model: str
    prompt_version: str


class PassageOut(BaseModel):
    """A passage the retriever found, shown with a refusal so the reader can look themselves."""

    ticker: str
    form: str
    period_label: str
    section: str
    source_url: str
    score: float
    excerpt: str = Field(description=f"Capped at {MAX_EXCERPT_CHARS} characters.")


class RefusalOut(BaseModel):
    reason: str = Field(
        description="out_of_scope | low_confidence | insufficient_evidence | verification_failed"
    )
    detail: str
    best_chunks: list[PassageOut] = Field(default_factory=list)


class AskResponse(BaseModel):
    """Exactly one of `answer` and `refusal` is set. Refusing is a normal 200."""

    answer: AnswerOut | None = None
    refusal: RefusalOut | None = None
    trace_id: str = Field(default="", description="Langfuse trace, empty when tracing is off.")

    @property
    def refused(self) -> bool:
        return self.refusal is not None


class HealthResponse(BaseModel):
    status: str
    model: str = Field(description="Provider and model, never the endpoint address.")
    prompt_version: str
    tracing: bool
