# Retrieval quality

## What it means

If the right passage is not retrieved, nothing downstream can fix it. Retrieval quality is measured, not felt: a labeled set of questions with the chunks that answer them, and metrics over the ranked results.

## Metrics used (RAG-008)

- **recall@k**: is a gold chunk in the top k? The metric that bounds answer quality.
- **MRR**: how high does the first gold chunk rank?
- **nDCG@k**: rank-weighted quality when several chunks are relevant.
- Broken down by company, form type, section, and question type (numeric lookup, definition, comparison across periods, unanswerable).

## Levers compared (RAG-006, RAG-007, RAG-009)

1. Embedding model.
2. Vector store and index type (Chroma vs FAISS flat vs HNSW).
3. Dense vs BM25 vs hybrid with reciprocal rank fusion.
4. Metadata filtering inferred from the question (ticker, period, section).
5. Cross-encoder reranking of the fused top-N.
6. Chunking strategy (see `chunking.md`).

## Measured

_Fill in: baseline table and the best configuration, from `docs/tradeoffs/retrieval-strategies.md` and `vector-stores.md`._

## Talking points

- How the eval set was built (LLM-assisted drafting, human verification) and why gold *chunks* matter more than gold *answers* for retrieval.
- Why BM25 still matters for financial text (exact tokens: ticker symbols, line items, "Q3 FY24").
- Filtering before vs after vector search, and what Chroma vs FAISS let you do.
- Reranking cost vs gain; when top-k is already good enough.

## Related

RAG-006, RAG-007, RAG-008, RAG-009.
