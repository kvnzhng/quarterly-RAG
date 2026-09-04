"""Grounded answers with verified citations (RAG-010).

The model is given passages tagged `[c1]`, `[c2]`, ... and told to cite every sentence.
Nothing it says is trusted: each sentence is checked for a citation that resolves to a
passage that was actually provided, and each figure is checked against the passages that
sentence cites. Three outcomes per sentence, and the answer carries all three:

- cited, every figure found in the cited passage -> a normal sentence
- cited, a figure not found there -> the figure is `derived, unverified`, not a lie and not
  a fact; RAG-021 recomputes it from its operands
- no citation, or a citation to a passage that was never provided -> unsupported

The refusal policy is RAG-011. This module only labels, and preserves the generator's own
`INSUFFICIENT_EVIDENCE` signal so the gate has something to act on.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from quarterly_rag.chunking.base import Chunk
from quarterly_rag.generation.base import LLM, ChatMessage
from quarterly_rag.generation.numbers import Figure, unsupported_figures

PROMPT_VERSION = "1"
PROMPT_PATH = Path(__file__).parent / "prompts" / f"grounded_answer_v{PROMPT_VERSION}.txt"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
DEFAULT_MAX_TOKENS = 1024
"""Generous on purpose. A thinking-mode model spends tokens reasoning before it writes,
and a truncated answer scores as ungrounded, which blames the model for the budget."""

# Models bracket citations differently: `[c1]` and the full-width `\u3010c1\u3011` both appear
# in practice, so both are accepted rather than scored as a missing citation.
_OPEN = "\\[\u3010"
_CLOSE = "\\]\u3011"
_TAG = re.compile(rf"[{_OPEN}]\s*c\s*(\d+)((?:\s*[,;]?\s*c?\s*\d+)*)\s*[{_CLOSE}]", re.I)
_EXTRA_TAG = re.compile(r"c?\s*(\d+)", re.I)
# Sentences end at ., ! or ? followed by whitespace and a capital or an opening quote. A
# citation that trails the full stop belongs to the sentence before it, so a bracket must
# not start a new one.
_SENTENCE_END = re.compile(rf"(?<=[.!?])\s+(?=[A-Z\"'({_OPEN}])")
_LEADING_TAGS = re.compile(rf"^\s*((?:[{_OPEN}]\s*c[^{_CLOSE}]*[{_CLOSE}]\s*)+)(.*)$", re.I | re.S)


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


class Answer(BaseModel):
    text: str
    """The answer with markers appended to sentences that failed a check."""
    raw_text: str
    citations: list[Citation] = Field(default_factory=list)
    unsupported_sentences: list[str] = Field(default_factory=list)
    derived_numbers: list[DerivedNumber] = Field(default_factory=list)
    insufficient_evidence: bool = False
    invalid_tags: list[str] = Field(default_factory=list)
    """Passage labels the model cited that were never provided to it."""
    prompt_version: str = PROMPT_VERSION
    model: str = ""

    @property
    def fully_grounded(self) -> bool:
        return not (self.unsupported_sentences or self.derived_numbers or self.invalid_tags)


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def tag_for(index: int) -> str:
    return f"c{index}"


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


def parse_tags(sentence: str) -> list[str]:
    """Every passage label a sentence cites, in order, deduplicated.

    Tolerates the shapes an 8B model actually writes: `[c1]`, `[c1][c3]`, `[c1, c3]`.
    """
    found: list[str] = []
    for match in _TAG.finditer(sentence):
        numbers = [match.group(1), *_EXTRA_TAG.findall(match.group(2) or "")]
        for number in numbers:
            tag = f"c{int(number)}"
            if tag not in found:
                found.append(tag)
    return found


def verify(raw_text: str, chunks: Sequence[Chunk], *, model: str = "") -> Answer:
    """Turn raw model output into a labelled `Answer`. Nothing here calls a model."""
    stripped = raw_text.strip()
    by_tag = {tag_for(i): chunk for i, chunk in enumerate(chunks, start=1)}

    if stripped.upper().startswith(INSUFFICIENT):
        return Answer(text=stripped, raw_text=raw_text, insufficient_evidence=True, model=model)

    rendered: list[str] = []
    unsupported: list[str] = []
    derived: list[DerivedNumber] = []
    invalid: list[str] = []
    used: list[str] = []

    for sentence in split_sentences(stripped):
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

        for tag in known:
            if tag not in used:
                used.append(tag)
        # Strip the citation labels first: `[c1]` would otherwise contribute the figure 1.
        claim = _TAG.sub(" ", sentence)
        missing: list[Figure] = unsupported_figures(claim, [by_tag[t].text for t in known])
        if missing:
            for figure in missing:
                derived.append(DerivedNumber(text=figure.raw, sentence=sentence, cited_tags=known))
            listed = ", ".join(f.raw for f in missing)
            rendered.append(f"{sentence} [derived, unverified: {listed}]")
        else:
            rendered.append(sentence)

    return Answer(
        text=" ".join(rendered),
        raw_text=raw_text,
        citations=[_citation(tag, by_tag[tag]) for tag in used],
        unsupported_sentences=unsupported,
        derived_numbers=derived,
        invalid_tags=invalid,
        model=model,
    )


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


def answer_question(
    llm: LLM,
    question: str,
    chunks: Sequence[Chunk],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Answer:
    """Ask the model, then verify what it said against the passages it was given."""
    if not chunks:
        return Answer(
            text=INSUFFICIENT,
            raw_text="",
            insufficient_evidence=True,
            model=getattr(llm, "label", ""),
        )
    messages = [
        ChatMessage(role="system", content=load_prompt()),
        ChatMessage(
            role="user",
            content=f"Passages:\n\n{render_passages(chunks)}\n\nQuestion: {question}",
        ),
    ]
    response = llm.chat(messages, max_tokens=max_tokens)
    return verify(response.text, chunks, model=llm.label)
