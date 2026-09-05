"""Grounded answers with verified citations (RAG-010) and verified arithmetic (RAG-021).

The model is given passages tagged `[c1]`, `[c2]`, ... and told to cite every sentence.
Nothing it says is trusted: each sentence is checked for a citation that resolves to a
passage that was actually provided, and each figure is checked against the passages that
sentence cites. Four outcomes per sentence, and the answer carries all four:

- cited, every figure found in the cited passage -> a normal sentence
- cited, a figure not found there, but a `CALC:` line recomputes it from cited operands ->
  `derived, verified`: a number the passages imply rather than print
- cited, a figure not found there and nothing recomputes it -> `derived, unverified`, not a
  lie and not a fact
- no citation, or a citation to a passage that was never provided -> unsupported

Prompt v1 forbade arithmetic outright, which made every derived number unverifiable by
construction. Prompt v2 allows it on condition the model shows its working; `calculations.py`
does the recomputation. The refusal policy is RAG-011. This module only labels, and
preserves the generator's own `INSUFFICIENT_EVIDENCE` signal so the gate has something to
act on.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from quarterly_rag.chunking.base import Chunk
from quarterly_rag.generation.base import LLM, ChatMessage, ChatResponse
from quarterly_rag.generation.calculations import (
    Calculation,
    matching_calculation,
    parse_calculation,
    split_calculations,
    verify_calculation,
)
from quarterly_rag.generation.citations import CLOSE, OPEN, TAG, parse_tags, tag_for
from quarterly_rag.generation.numbers import Figure, unsupported_figures

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_PROMPT_VERSION",
    "INSUFFICIENT",
    "Answer",
    "Citation",
    "DerivedNumber",
    "answer_question",
    "load_prompt",
    "no_passages",
    "parse_tags",
    "respond",
    "split_sentences",
    "tag_for",
    "verify",
    "verify_response",
]

DEFAULT_PROMPT_VERSION = "1"
"""v1 answers only from what a passage prints; v2 adds calculation provenance (RAG-021).

v1 is the default because `make eval` measured the cost of v2 and it is not free: with
`gpt-oss:20b` it answers two fewer of the 33 answerable questions. Set
`ANSWER_PROMPT_VERSION=2` to turn calculation provenance on, which is the only way to get an
answer at all to a question whose number no filing prints."""

PROMPTS_DIR = Path(__file__).parent / "prompts"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
DEFAULT_MAX_TOKENS = 1024
"""Generous on purpose. A thinking-mode model spends tokens reasoning before it writes,
and a truncated answer scores as ungrounded, which blames the model for the budget."""

# Sentences end at ., ! or ? followed by whitespace and a capital or an opening quote. A
# citation that trails the full stop belongs to the sentence before it, so a bracket must
# not start a new one.
_SENTENCE_END = re.compile(rf"(?<=[.!?])\s+(?=[A-Z\"'({OPEN}])")
_LEADING_TAGS = re.compile(rf"^\s*((?:[{OPEN}]\s*c[^{CLOSE}]*[{CLOSE}]\s*)+)(.*)$", re.I | re.S)


class Citation(BaseModel):
    tag: str
    """The `c1` label the model wrote, unique within one answer."""
    chunk_id: str
    ticker: str
    form: str
    period_label: str
    section: str
    source_url: str
    quote: str = Field(description="The start of the cited passage, for display")


class DerivedNumber(BaseModel):
    text: str
    """The figure as the answer wrote it, e.g. `$15,381 million`."""
    sentence: str
    cited_tags: list[str]
    verified: bool = False
    """True when a `CALC:` line recomputes this figure from operands its passages state."""
    calculation: Calculation | None = None
    """The calculation claiming this figure, verified or not, so the reason is reportable."""


class Answer(BaseModel):
    text: str
    """The answer with markers appended to sentences that failed a check."""
    raw_text: str
    prose: str = ""
    """The answer without its calculation lines. A `CALC:` line is working, not a claim, so
    the judge and the sentence counts must not see it as one."""
    citations: list[Citation] = Field(default_factory=list)
    cited_sentences: int = 0
    """Sentences carrying a citation that resolves. Calculations cite passages too, so the
    citation list alone no longer tells you whether any *claim* was grounded (RAG-021)."""
    unsupported_sentences: list[str] = Field(default_factory=list)
    derived_numbers: list[DerivedNumber] = Field(default_factory=list)
    calculations: list[Calculation] = Field(default_factory=list)
    insufficient_evidence: bool = False
    invalid_tags: list[str] = Field(default_factory=list)
    """Passage labels the model cited that were never provided to it."""
    prompt_version: str = DEFAULT_PROMPT_VERSION
    model: str = ""
    stop_reason: str | None = None
    """As the provider reported it. `length` means the budget cut the answer off, so an
    unparsed calculation is the budget's doing and not the model's."""
    input_tokens: int | None = None
    output_tokens: int | None = None
    """What the provider charged for this answer, when it says. Carried so the trace can
    record it without the generation layer knowing that tracing exists (RAG-013)."""

    @property
    def truncated(self) -> bool:
        return self.stop_reason in {"length", "max_tokens"}

    @property
    def unverified_derived(self) -> list[DerivedNumber]:
        return [d for d in self.derived_numbers if not d.verified]

    @property
    def verified_derived(self) -> list[DerivedNumber]:
        return [d for d in self.derived_numbers if d.verified]

    @property
    def fully_grounded(self) -> bool:
        """Every sentence cited, every citation real, every figure stated or recomputed.

        A derived number that recomputes from cited operands counts as grounded here, which
        it did not before RAG-021: the passages imply it even though none of them prints it.
        The presence-only rate is still reported alongside as `figures_verified`.
        """
        return not (self.unsupported_sentences or self.unverified_derived or self.invalid_tags)


def load_prompt(version: str = DEFAULT_PROMPT_VERSION) -> str:
    path = PROMPTS_DIR / f"grounded_answer_v{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"no prompt version {version!r} at {path}")
    return path.read_text(encoding="utf-8").strip()


def render_passages(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(f"[{tag_for(i)}] {c.text}" for i, c in enumerate(chunks, start=1))


def split_sentences(text: str) -> list[str]:
    """Sentences, with a citation that trails a full stop kept on the sentence it cites.

    Models place the citation on either side of the stop, so `Sales rose [c1]. Margin fell
    [c2].` and `Sales rose. [c1] Margin fell. [c2]` must yield the same two sentences.
    """
    parts = [p.strip() for p in _SENTENCE_END.split(text.strip()) if p.strip()]
    sentences: list[str] = []
    for part in parts:
        leading = _LEADING_TAGS.match(part) if sentences else None
        if leading:
            # These tags close the previous sentence, whatever follows them.
            sentences[-1] = f"{sentences[-1]} {leading.group(1).strip()}"
            remainder = leading.group(2).strip()
            if remainder:
                sentences.append(remainder)
            continue
        sentences.append(part)
    return sentences


def verify(
    raw_text: str,
    chunks: Sequence[Chunk],
    *,
    model: str = "",
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    stop_reason: str | None = None,
    usage: tuple[int | None, int | None] = (None, None),
) -> Answer:
    """Turn raw model output into a labelled `Answer`. Nothing here calls a model."""
    stripped = raw_text.strip()
    by_tag = {tag_for(i): chunk for i, chunk in enumerate(chunks, start=1)}

    if stripped.upper().startswith(INSUFFICIENT):
        return Answer(
            text=stripped,
            raw_text=raw_text,
            prose=stripped,
            insufficient_evidence=True,
            model=model,
            prompt_version=prompt_version,
            stop_reason=stop_reason,
            input_tokens=usage[0],
            output_tokens=usage[1],
        )

    prose, calc_lines = split_calculations(stripped)
    passages = {tag: chunk.text for tag, chunk in by_tag.items()}
    calculations = [verify_calculation(parse_calculation(line), passages) for line in calc_lines]

    rendered: list[str] = []
    cited_sentences = 0
    unsupported: list[str] = []
    derived: list[DerivedNumber] = []
    invalid: list[str] = []
    used: list[str] = []

    for sentence in split_sentences(prose):
        tags = parse_tags(sentence)
        known = [t for t in tags if t in by_tag]
        unknown = [t for t in tags if t not in by_tag]
        for tag in unknown:
            if tag not in invalid:
                invalid.append(tag)

        if not known:
            unsupported.append(sentence)
            reason = f"cites {', '.join(unknown)}, which was not provided" if unknown else "uncited"
            rendered.append(f"{sentence} [unsupported: {reason}]")
            continue

        cited_sentences += 1
        for tag in known:
            if tag not in used:
                used.append(tag)
        # Strip the citation labels first: `[c1]` would otherwise contribute the figure 1.
        claim = TAG.sub(" ", sentence)
        missing: list[Figure] = unsupported_figures(claim, [by_tag[t].text for t in known])
        if not missing:
            rendered.append(sentence)
            continue
        rendered.append(_mark(sentence, missing, calculations, known, derived))

    for calculation in calculations:
        for tag in calculation.tags:
            if tag not in by_tag:
                if tag not in invalid:
                    invalid.append(tag)
            elif tag not in used:
                used.append(tag)

    text = " ".join(rendered)
    if calculations:
        text = "\n".join([text, *(c.rendered() for c in calculations)]).strip()
    return Answer(
        text=text,
        raw_text=raw_text,
        prose=prose,
        citations=[_citation(tag, by_tag[tag]) for tag in used],
        cited_sentences=cited_sentences,
        unsupported_sentences=unsupported,
        derived_numbers=derived,
        calculations=calculations,
        invalid_tags=invalid,
        model=model,
        prompt_version=prompt_version,
        stop_reason=stop_reason,
        input_tokens=usage[0],
        output_tokens=usage[1],
    )


def _mark(
    sentence: str,
    missing: Sequence[Figure],
    calculations: list[Calculation],
    known: list[str],
    derived: list[DerivedNumber],
) -> str:
    """Label a sentence whose figures are not printed in the passages it cites.

    Each such figure is looked up among the calculations. One that recomputes from cited
    operands is `verified`; anything else keeps the RAG-010 label.
    """
    verified: list[Figure] = []
    unverified: list[Figure] = []
    for figure in missing:
        calculation = matching_calculation(calculations, figure)
        is_verified = calculation is not None and calculation.verified
        derived.append(
            DerivedNumber(
                text=figure.raw,
                sentence=sentence,
                cited_tags=known,
                verified=is_verified,
                calculation=calculation,
            )
        )
        (verified if is_verified else unverified).append(figure)

    marks: list[str] = []
    if verified:
        marks.append(f"[derived, verified: {', '.join(f.raw for f in verified)}]")
    if unverified:
        marks.append(f"[derived, unverified: {', '.join(f.raw for f in unverified)}]")
    return " ".join([sentence, *marks])


def _citation(tag: str, chunk: Chunk) -> Citation:
    quote = " ".join(chunk.text.split())[:200]
    return Citation(
        tag=tag,
        chunk_id=chunk.chunk_id,
        ticker=chunk.ticker,
        form=chunk.form,
        period_label=chunk.period_label,
        section=chunk.section,
        source_url=chunk.source_url,
        quote=quote,
    )


def no_passages(llm: LLM, prompt_version: str = DEFAULT_PROMPT_VERSION) -> Answer:
    """Nothing was retrieved, so there is nothing to ask about and no model call to make."""
    return Answer(
        text=INSUFFICIENT,
        raw_text="",
        prose=INSUFFICIENT,
        insufficient_evidence=True,
        model=getattr(llm, "label", ""),
        prompt_version=prompt_version,
    )


def respond(
    llm: LLM,
    question: str,
    chunks: Sequence[Chunk],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> ChatResponse:
    """The model call on its own, with nothing checked yet.

    Separate from `verify` so a caller that times its stages can tell how long the model
    took from how long checking it took (RAG-013). `answer_question` is still the one call
    for everyone who does not care.
    """
    messages = [
        ChatMessage(role="system", content=load_prompt(prompt_version)),
        ChatMessage(
            role="user",
            content=f"Passages:\n\n{render_passages(chunks)}\n\nQuestion: {question}",
        ),
    ]
    return llm.chat(messages, max_tokens=max_tokens)


def verify_response(
    response: ChatResponse,
    chunks: Sequence[Chunk],
    *,
    model: str = "",
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> Answer:
    """`verify`, given the whole response, so usage and stop reason are not dropped."""
    return verify(
        response.text,
        chunks,
        model=model,
        prompt_version=prompt_version,
        stop_reason=response.stop_reason,
        usage=(response.input_tokens, response.output_tokens),
    )


def answer_question(
    llm: LLM,
    question: str,
    chunks: Sequence[Chunk],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> Answer:
    """Ask the model, then verify what it said against the passages it was given."""
    if not chunks:
        return no_passages(llm, prompt_version)
    response = respond(llm, question, chunks, max_tokens=max_tokens, prompt_version=prompt_version)
    return verify_response(response, chunks, model=llm.label, prompt_version=prompt_version)
