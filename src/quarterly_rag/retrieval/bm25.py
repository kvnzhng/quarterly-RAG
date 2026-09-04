"""Lexical retrieval (RAG-009).

Financial questions turn on exact strings. `109,417`, `Q3 FY2026` and `Total net sales`
are tokens a keyword index matches and an embedding blurs, which is the gap RAG-008
measured: dense retrieval found zero of seven quarterly questions.

The index is built in memory from `data/chunks/`. At 1,391 chunks that costs under a
second; a corpus large enough for that to hurt wants a real inverted index instead.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from quarterly_rag.chunking.base import Chunk
from quarterly_rag.indexing.embed_text import CONTEXT, embed_text
from quarterly_rag.retrieval.base import RetrievedChunk
from quarterly_rag.retrieval.query import expand

TOKENIZER_VERSION = "1"
# One token for an alphanumeric run (`fy2026`, `q3`), one for a figure with its separators
# (`109,417`, `46.9`), one for a word. Currency, percent and table pipes are dropped, so
# `$109,417`, `(109,417)` and `109,417` all become the same token.
_TOKEN = re.compile(r"[a-z]+\d[\w.]*|\d[\d,]*(?:\.\d+)?|[a-z]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 over the chunk text plus its provenance header.

    Indexing the `context` variant is what makes a period searchable: the chunk itself
    says "June 27, 2026" and never "FY2026 Q3", which is how the question asks.
    """

    def __init__(
        self, chunks: Sequence[Chunk], *, variant: str = CONTEXT, expand_query: bool = True
    ):
        self.chunks = list(chunks)
        self.variant = variant
        self.expand_query = expand_query
        self._index = (
            BM25Okapi([tokenize(embed_text(c, variant)) for c in self.chunks]) if chunks else None
        )

    @property
    def name(self) -> str:
        return "bm25"

    def retrieve(
        self, question: str, k: int = 5, where: dict[str, object] | None = None
    ) -> list[RetrievedChunk]:
        if self._index is None or not question.strip():
            return []
        query = expand(question) if self.expand_query else question
        scores = self._index.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        kept: list[RetrievedChunk] = []
        largest = max(scores) if len(scores) else 0.0
        for index in order:
            if scores[index] <= 0:
                break
            chunk = self.chunks[index]
            if where and not _matches(chunk, where):
                continue
            kept.append(
                RetrievedChunk(
                    chunk=chunk,
                    # BM25 scores are unbounded, so they are normalised to the best hit for
                    # this query. Comparable within a ranking, never across queries.
                    score=float(scores[index] / largest) if largest else 0.0,
                    retriever=self.name,
                    rank=len(kept) + 1,
                )
            )
            if len(kept) >= k:
                break
        return kept


def _matches(chunk: Chunk, where: dict[str, object]) -> bool:
    return all(getattr(chunk, field, None) == value for field, value in where.items())
