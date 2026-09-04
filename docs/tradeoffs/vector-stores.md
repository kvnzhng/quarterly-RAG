# Vector stores: ChromaDB vs FAISS

**Status:** decided (ADR-010; measured on the 2,891-chunk corpus, 2026-09-04)
**Ticket:** RAG-007

## Question

Which store holds the embeddings? The obvious axis is speed, and it turns out to be the wrong one to decide on.

## Candidates

| Candidate | What it is | License |
|---|---|---|
| ChromaDB (default) | An embedded vector database: payloads, metadata filtering, upsert, persistence | Apache-2.0 |
| FAISS flat | An exact similarity index and nothing else | MIT |
| FAISS HNSW | The same, with an approximate navigable small-world graph | MIT |

FAISS is a similarity index, not a database. It maps vectors to integer ids and has no concept of a document, a payload, or a filter, so the adapter keeps chunks in a parallel list persisted as JSONL beside the index, and filters after the search rather than during it.

## Criteria

| Criterion | How measured | Weight |
|---|---|---|
| Retrieval quality | recall@k, MRR and nDCG on the same eval set, only the store changing | high |
| Query latency | p50 and p95 over 30 queries, filtered and unfiltered | medium |
| Operational surface | upsert, delete, filtering, how many files hold the state | high |
| Build time, disk, memory | full rebuild of the corpus | medium |

## Results

2,891 section-aware chunks, 768 dimensions, the same vectors given to every store.

### Retrieval quality is identical

| Store | recall@1 | recall@3 | recall@5 | recall@10 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| ChromaDB | 39.4% | 45.5% | 48.5% | 57.6% | 0.440 | 0.406 |
| FAISS flat | 39.4% | 45.5% | 48.5% | 57.6% | 0.440 | 0.406 |
| FAISS HNSW | 39.4% | 45.5% | 48.5% | 57.6% | 0.440 | 0.406 |

Not close: the same. Both Chroma and FAISS HNSW are approximate, and on random query vectors they agree with an exhaustive search only 90.0% and 87.7% of the time at k=10. None of that disagreement lands on a question in the eval set, because hybrid retrieval fuses two rankings and a passage missed by one is found by the other.

### FAISS is much faster, and it does not matter

| Store | Build (s) | Disk (MB) | p50 (ms) | p95 (ms) | p50 filtered (ms) | Peak RSS (MB) |
|---|---|---|---|---|---|---|
| ChromaDB | 1.5 | 36.5 | 0.98 | 1.25 | 3.44 | 72.5 |
| FAISS flat | 0.0 | 13.2 | 0.14 | 0.17 | 0.18 | 45.4 |
| FAISS HNSW | 0.1 | 14.0 | **0.06** | 0.11 | 0.08 | **14.3** |

FAISS HNSW is 16x faster than Chroma per lookup and uses a fifth of the memory. Then the context:

| Step of one dense retrieval | p50 |
|---|---|
| Embedding the question on the network server | 31.4 ms |
| Looking it up in ChromaDB | 0.98 ms |

**The store is 3% of a retrieval and a rounding error next to the generation call, which takes seconds.** Choosing FAISS would make the pipeline about 3% faster on the step that is not the bottleneck.

### The differences that are not about speed

| | ChromaDB | FAISS |
|---|---|---|
| Payloads | stored with the vector | a parallel JSONL the adapter maintains |
| Metadata filter | native, in the query | after the search, so a filtered query over-fetches 20x |
| Upsert | re-adding an id replaces it | no such operation; this adapter refuses rather than duplicate silently |
| Delete | supported | rebuild the index |
| State | one directory | two files that must be written together, and the adapter checks they agree |
| Vector width | inferred | required before the first vector arrives |

Every row is work the FAISS adapter had to do that Chroma does for free, and every one is a place a bug can hide. The out-of-step check exists because a half-written index would otherwise read as a smaller, working one.

Run record: commit `e8444e5`, section-aware chunks, `nomic-embed-text` with task prefixes, `context` embed variant, hybrid retrieval, k as shown, macOS on Apple silicon.

## Decision

**ChromaDB stays the default** (ADR-010). Retrieval quality is identical, so the decision is operational, and there Chroma is ahead: upsert makes re-indexing a changed filing a one-line operation, native filtering is what RAG-026's period filter uses, and the store is one directory rather than two files that must agree.

**Choose FAISS when the corpus outgrows this one.** At 2,891 chunks the memory difference is 58 MB and the latency difference is invisible. At ten million chunks the same ratios are the whole decision, and FAISS HNSW's sub-linear search and fifth-of-the-memory footprint stop being a rounding error. FAISS is also the right answer for shipping a read-only index as an artifact, where upsert is not wanted.

Both adapters stay in the repo behind one protocol, so this is a setting rather than a rewrite.

## Interview one-liner

FAISS is sixteen times faster per lookup and it changed nothing, because the store is three percent of a retrieval and the embedding call is thirty times larger; the real difference is that FAISS has no upsert, no payloads and no filtering, so everything a vector database does for you becomes adapter code where a bug can hide.
