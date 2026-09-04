"""Turning settings and a strategy name into a retriever (RAG-009)."""

from __future__ import annotations

from quarterly_rag.chunking.build import iter_chunks
from quarterly_rag.config import Settings
from quarterly_rag.generation.base import LLM
from quarterly_rag.indexing.base import Embedder, VectorStore
from quarterly_rag.retrieval.base import Retriever
from quarterly_rag.retrieval.bm25 import BM25Retriever
from quarterly_rag.retrieval.dense import DenseRetriever
from quarterly_rag.retrieval.filtered import FilteredRetriever
from quarterly_rag.retrieval.hybrid import HybridRetriever
from quarterly_rag.retrieval.rerank import LLMReranker

TICKERS = ("AAPL", "NVDA")
STRATEGIES = ("dense", "bm25", "hybrid", "hybrid-filter", "hybrid-rerank")
DEFAULT_STRATEGY = "hybrid"


def build_retriever(
    settings: Settings,
    strategy: str,
    *,
    embedder: Embedder,
    store: VectorStore,
    chunk_strategy: str = "fixed",
    variant: str = "context",
    llm: LLM | None = None,
) -> Retriever:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown retrieval strategy {strategy!r}; expected one of {STRATEGIES}")

    dense = DenseRetriever(embedder, store)
    if strategy == "dense":
        return dense

    chunks = [c for t in TICKERS for c in iter_chunks(settings, t, chunk_strategy)]
    if not chunks:
        raise FileNotFoundError(
            f"no chunks for strategy {chunk_strategy!r}; run `rag chunk build` first"
        )
    bm25 = BM25Retriever(chunks, variant=variant)
    if strategy == "bm25":
        return bm25

    hybrid = HybridRetriever(
        [dense, bm25], pool=settings.retrieval_pool, fusion_k=settings.fusion_k
    )
    if strategy == "hybrid":
        return hybrid
    if strategy == "hybrid-filter":
        return FilteredRetriever(hybrid)
    if strategy == "hybrid-rerank":
        if llm is None:
            raise ValueError("hybrid-rerank needs an LLM; pass one in")
        return LLMReranker(hybrid, llm, pool=settings.rerank_pool)
    raise AssertionError(f"unreachable: {strategy}")  # pragma: no cover
