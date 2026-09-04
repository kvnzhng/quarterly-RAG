# Retrieval quality

## What it means

If the right passage is not retrieved, nothing downstream can fix it. Retrieval quality is measured, not felt: a labeled set of questions with the chunks that answer them, and metrics over the ranked results.

## Labels (RAG-019)

Gold evidence is a span into the parsed filing (accession, section, char offsets), human-verified, with a question type (`lookup`, `derived`, `cross_period`, `unanswerable`). A chunk is relevant when it overlaps a span. Labeling spans instead of chunk ids means one eval set scores every chunker, store, and retriever.

The v0 set (RAG-019) holds 43 questions: 23 lookup, 5 derived, 5 cross-period, 10 unanswerable, split evenly between refusal reasons. Every one was verified against the filing. Two known limitations to revisit at RAG-008:

- **6 of the 16 filings carry all the evidence.** The other 10 are only distractors, so the set exercises retrieval precision more than period filtering.
- **30% of spans are prose, the rest tables.** Financial filings are table-heavy so some skew is honest, but a chunking comparison run on this set will weigh table handling more than narrative handling.

## Metrics used (RAG-008)

- **recall@k**: is a gold chunk in the top k? A hit rate rather than fractional recall: one relevant chunk in the prompt is what lets the generator answer, so the metric asks whether any was found, not how many. It bounds answer quality.
- **MRR**: how high does the first gold chunk rank?
- **nDCG@k**: rank-weighted quality when several chunks are relevant.
- Broken down by company, form type, section, and question type (numeric lookup, definition, comparison across periods, unanswerable).

## Levers compared (RAG-006, RAG-007, RAG-009)

1. Embedding model.
2. Vector store and index type (Chroma vs FAISS flat vs HNSW).
3. Dense vs BM25 vs hybrid with reciprocal rank fusion (decided: hybrid, ADR-008).
3b. Vector store: ChromaDB vs FAISS flat vs FAISS HNSW. Measured identical on every retrieval metric, so the decision was operational (ADR-010). The store is 3% of a retrieval; the embedding call is thirty times larger.
4. Metadata filtering inferred from the question (ticker, period, section).
5. Cross-encoder reranking of the fused top-N.
6. Chunking strategy (see `chunking.md`).

## Baseline

Dense retrieval only, 33 answerable questions, 1,391 chunks from the fixed chunker, measured 2026-09-04 by `rag eval retrieval`.

| Embedded text | recall@1 | recall@3 | recall@5 | recall@10 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| raw chunk | 9.1% | 15.2% | 18.2% | 24.2% | 0.131 | 0.096 |
| with a context header | 18.2% | 30.3% | 36.4% | 45.5% | 0.267 | 0.232 |

Run record for both: commit `e1c3f08`, corpus `ab54dafa27ee5fe1`, eval set `ed55e0b644402717`, parser 1, chunker `fixed` at 350 words with 60 overlap, `openai_compatible/nomic-embed-text` with nomic task prefixes, ChromaDB, dense retriever, relevance = any overlap with a gold span. Reports under `reports/`.

**Read these with the denominator in view.** 33 questions means one question is three percentage points, and the smallest breakdown cell holds five. These numbers separate large effects, not small ones.

### How close retrieval got

The single most useful cut, because it says which lever to pull next.

| The top 5 reached | raw | with context header |
|---|---|---|
| the right filing | 78.8% | 90.9% |
| the right section of it | 45.5% | 63.6% |
| a chunk holding the evidence | 18.2% | 36.4% |

Retrieval almost always finds the right document and then loses inside it. Filtering by company or period would therefore buy little; ranking and chunking are where the loss is.

### Two findings the eval set caught

- **The embedding model was being misused.** `nomic-embed-text` is trained with `search_query:` on questions and `search_document:` on passages, and was getting neither. Nothing errored, the vectors were the right shape and already unit-normalised, and recall@5 was a third lower. Fixed in the `Embedder` interface so the mistake is no longer expressible.
- **A context header roughly doubles recall at every cutoff.** A chunk of a financial table names neither the company nor the fiscal period, so a question naming either has nothing to match. Both variants stay in the repo and both are rebuilt on demand, so the gap keeps being measured.

### The 10-Q result

| Form | questions | recall@5 |
|---|---|---|
| 10-K | 26 | 46.2% |
| 10-Q | 7 | 0.0% |

Not one of the seven quarterly questions found its evidence. Every one of them asks for a figure that lives in the condensed financial statements, and retrieval returned the management discussion of the same filing instead: prose that talks about the number, ranked above the table that contains it. Two consequences. Answer quality on quarterly questions is currently capped at zero, and the fix is exact-term matching, since `109,417` and `Total net sales` are strings that BM25 handles and embeddings do not (RAG-009).

### Where retrieval ended up

The final default, after RAG-009, RAG-026 and RAG-020, on the same 33 questions:

| Configuration | recall@1 | recall@3 | recall@5 | recall@10 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| dense, fixed chunks (RAG-008 baseline) | 18.2% | 30.3% | 36.4% | 45.5% | 0.266 | 0.232 |
| hybrid + quarter filter, fixed chunks | 21.2% | 33.3% | 48.5% | 51.5% | 0.306 | 0.282 |
| **hybrid + quarter filter, section-aware chunks** | **39.4%** | **45.5%** | 48.5% | **57.6%** | **0.440** | **0.406** |

recall@1 more than doubled and MRR rose 65% from the baseline, while recall@5 moved 12 points. Retrieval improved most in *where* the evidence lands, which is what the generator reads first. The remaining ceiling is real: recall@20 is 72.7%, so about a quarter of questions have evidence neither ranking nor chunking reaches.

## Talking points

- How the eval set was built (LLM-assisted drafting, human verification), why retrieval is scored on gold *evidence* rather than gold *answers*, and why that evidence is a span rather than a chunk id (labels survive re-chunking).
- Why BM25 still matters for financial text (exact tokens: ticker symbols, line items, "Q3 FY24").
- Filtering before vs after vector search, and what Chroma vs FAISS let you do.
- Reranking cost vs gain; when top-k is already good enough.
- The near-miss ladder as a diagnostic: right filing, right section, right chunk. Losing at the first step is a filtering problem; losing at the last is a ranking or chunking problem, and they need different fixes.
- Why every reported number carries a run record, and what breaks without one.

## Reading

- Karpukhin et al. 2020, [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- Cormack et al. 2009, [Reciprocal Rank Fusion](https://doi.org/10.1145/1571941.1572114)
- Nogueira and Cho 2019, [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085)
- Thakur et al. 2021, [BEIR](https://arxiv.org/abs/2104.08663)
- Full list: README, "Reading and courses".

## Related

RAG-019, RAG-006, RAG-007, RAG-008, RAG-009.
