# ADR-010: ChromaDB is the default vector store; FAISS is for scale

**Date:** 2026-09-04
**Status:** accepted
**Ticket:** RAG-007

## Context

ADR-003 committed to implementing two vector stores and benchmarking them before choosing. Both are now implemented behind one `VectorStore` protocol: ChromaDB, and FAISS in flat and HNSW forms.

## Decision

- **ChromaDB is the default.** Retrieval quality is identical across all three stores on the eval set: the same recall at every k, the same MRR, the same nDCG. The decision is therefore operational, and Chroma is ahead on every operational axis that matters at this size.
- **FAISS is the choice at scale or for a read-only artifact.** It is 16x faster per lookup and uses a fifth of the memory, and both differences are invisible here: the store is 3% of a retrieval, against 31 ms to embed the question and seconds to generate an answer. Those ratios become the whole decision at millions of chunks.
- **The FAISS adapter refuses to re-add an existing id rather than duplicating it.** FAISS has no upsert and no delete; ids are positional. Silently storing a chunk twice would let both copies rank.
- **The FAISS adapter filters after the search**, over-fetching 20x when a filter is present, because FAISS cannot filter. On this corpus that is still faster than Chroma's native filter; on a corpus where the filter is selective it would not be.
- **The FAISS adapter validates that its two files agree.** The index and the payload JSONL are written together, and a mismatch raises rather than presenting a smaller working index.

## Consequences

- Re-indexing a changed filing is a one-line upsert on Chroma and a full rebuild on FAISS. The ingestion path is idempotent and re-runs often, so this is the operational difference that decided it.
- The period filter added in RAG-026 uses native metadata filtering. Keeping it fast on FAISS would mean growing the over-fetch factor with the filter's selectivity, which is exactly the kind of tuning a database exists to avoid.
- `faiss-cpu` is a runtime dependency of roughly 30 MB and is imported only when a FAISS store is built. It is worth carrying because the comparison is part of the project's purpose, and because the decision reverses at a corpus size this project could plausibly reach.
- The benchmark measured what the store costs relative to the pipeline around it. Any future store comparison should quote that ratio rather than a raw latency, because a 16x speedup on 3% of the work is a 3% speedup.
