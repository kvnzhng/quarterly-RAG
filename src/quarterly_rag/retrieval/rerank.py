"""Reranking a candidate pool with the configured chat model (RAG-009).

A cross-encoder such as `bge-reranker-base` is the usual choice and would mean
sentence-transformers and PyTorch, roughly two gigabytes, on a project whose defaults
promise to fit a laptop (ADR-003). The `LLM` protocol already reaches a capable model, so
this scores candidates with it instead: no new dependency, and a measurement that says
whether a dedicated reranker would be worth the weight.

The cost is real: one model call per candidate. At a pool of 20 that is 20 calls a
question, which the tradeoff page reports in seconds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from quarterly_rag.errors import ModelServerError
from quarterly_rag.generation.base import LLM, ChatMessage
from quarterly_rag.retrieval.base import RetrievedChunk, Retriever

SYSTEM = (
    "You judge whether a passage from an SEC filing helps answer a question.\n"
    "Reply with a single integer from 0 to 10 and nothing else.\n"
    "10 means the passage states the answer. 5 means it is about the right topic and "
    "does not state it. 0 means it is unrelated."
)
_SCORE = re.compile(r"\b(10|[0-9])\b")
DEFAULT_POOL = 20


@dataclass
class LLMReranker:
    """Rescores a pool from `inner` and returns the best `k`."""

    inner: Retriever
    llm: LLM
    pool: int = DEFAULT_POOL
    max_chars: int = 1200
    """Passages are truncated: a judgement needs the opening, not the whole table."""

    @property
    def name(self) -> str:
        return f"{self.inner.name}+rerank"

    def retrieve(
        self, question: str, k: int = 5, where: dict[str, object] | None = None
    ) -> list[RetrievedChunk]:
        candidates = self.inner.retrieve(question, k=self.pool, where=where)
        if not candidates:
            return []
        scored: list[tuple[float, RetrievedChunk]] = []
        for candidate in candidates:
            scored.append((self._score(question, candidate.chunk.text), candidate))
        # A tie falls back to the order the inner retriever chose, which is why the
        # original rank is the second sort key.
        scored.sort(key=lambda pair: (-pair[0], pair[1].rank))
        return [
            RetrievedChunk(
                chunk=candidate.chunk, score=score / 10.0, retriever=self.name, rank=rank
            )
            for rank, (score, candidate) in enumerate(scored[:k], start=1)
        ]

    def _score(self, question: str, passage: str) -> float:
        messages = [
            ChatMessage(role="system", content=SYSTEM),
            ChatMessage(
                role="user",
                content=f"Question: {question}\n\nPassage:\n{passage[: self.max_chars]}",
            ),
        ]
        try:
            reply = self.llm.chat(messages, max_tokens=2048).text
        except ModelServerError:
            return 0.0  # a failed judgement must not promote a candidate
        match = _SCORE.search(reply.strip().splitlines()[-1] if reply.strip() else "")
        return float(match.group(1)) if match else 0.0
