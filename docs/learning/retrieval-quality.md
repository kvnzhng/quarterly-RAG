# Retrieval quality

## What it means

If the right passage is not retrieved, nothing downstream can fix it. Retrieval quality is measured, not felt: a labeled set of questions with the chunks that answer them, and metrics over the ranked results.

## Labels (RAG-019)

Gold evidence is a span into the parsed filing (accession, section, char offsets), human-verified, with a question type (`lookup`, `derived`, `cross_period`, `unanswerable`). A chunk is relevant when it overlaps a span. Labeling spans instead of chunk ids means one eval set scores every chunker, store, and retriever.

The v0 set (RAG-019) holds 43 questions: 23 lookup, 5 derived, 5 cross-period, 10 unanswerable, split evenly between refusal reasons. Every one was verified against the filing. Two known limitations to revisit at RAG-008:

- **6 of the 16 filings carry all the evidence.** The other 10 are only distractors, so the set exercises retrieval precision more than period filtering.
- **30% of spans are prose, the rest tables.** Financial filings are table-heavy so some skew is honest, but a chunking comparison run on this set will weigh table handling more than narrative handling.

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

First numbers, RAG-006, dense retrieval only over 33 answerable questions and 1,391 chunks:

| Query and document text | recall@1 | recall@5 | recall@10 |
|---|---|---|---|
| raw chunk, no task prefix | 3.0% | 15.2% | 24.2% |
| context header, nomic task prefixes | 18.2% | 36.4% | 45.5% |

Two lessons, both found only because the eval set existed before the index did. `nomic-embed-text` is trained with `search_query:` and `search_document:` prefixes and silently under-performs without them, which cost a third of recall and raised no error. And a one-line header naming the company, period and section roughly doubles recall, because a chunk of a financial table contains neither the company name nor the fiscal period. Full table and run record: `docs/tradeoffs/embeddings.md`.

_Fill in: baseline table with its run record (commit, corpus hash, chunker, embedding model, k) from RAG-008, then the best configuration from `docs/tradeoffs/retrieval-strategies.md` and `vector-stores.md`._

## Talking points

- How the eval set was built (LLM-assisted drafting, human verification), why retrieval is scored on gold *evidence* rather than gold *answers*, and why that evidence is a span rather than a chunk id (labels survive re-chunking).
- Why BM25 still matters for financial text (exact tokens: ticker symbols, line items, "Q3 FY24").
- Filtering before vs after vector search, and what Chroma vs FAISS let you do.
- Reranking cost vs gain; when top-k is already good enough.

## Reading

- Karpukhin et al. 2020, [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- Cormack et al. 2009, [Reciprocal Rank Fusion](https://doi.org/10.1145/1571941.1572114)
- Nogueira and Cho 2019, [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085)
- Thakur et al. 2021, [BEIR](https://arxiv.org/abs/2104.08663)
- Full list: README, "Reading and courses".

## Related

RAG-019, RAG-006, RAG-007, RAG-008, RAG-009.
