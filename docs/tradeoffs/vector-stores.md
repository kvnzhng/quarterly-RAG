# Vector stores: ChromaDB vs FAISS (vs LanceDB, Qdrant, pgvector)

**Status:** draft
**Ticket:** RAG-007

## Question

Which vector store is the default for a single-machine RAG over ~10k-50k chunks, and when would the other choice win? Both are implemented behind the same `VectorStore` protocol so the comparison is on the same corpus and queries.

## Candidates

| Candidate | One-line description | License | Runs locally? |
|---|---|---|---|
| ChromaDB | Embedded vector DB with metadata filtering and persistence, Python-native | Apache-2.0 | yes |
| FAISS (faiss-cpu) | Similarity search library from Meta; indexes (Flat, IVF, HNSW), no metadata layer | MIT | yes |
| LanceDB | Embedded, columnar (Lance format), hybrid search built in | Apache-2.0 | yes |
| Qdrant | Server-based, rich filtering, needs docker | Apache-2.0 | yes (docker) |
| pgvector | Postgres extension; good if you already run Postgres | PostgreSQL | yes (docker) |

Primary comparison: **ChromaDB vs FAISS**. Others are documented as "when you outgrow this".

## Criteria

| Criterion | How measured | Weight |
|---|---|---|
| Build time | wall clock to index all chunks | medium |
| Query latency | p50 / p95 over the eval question set, k=10 | high |
| Recall vs exact | recall@10 of the ANN index against a brute-force baseline | high |
| Metadata filtering | can we filter by ticker/period/section natively; cost of pre- vs post-filtering | high |
| Persistence | save/load round trip, size on disk | medium |
| Memory | RSS after load | medium |
| Operational complexity | lines of adapter code, failure modes, upgrade story | medium |

## Results

_To be filled by RAG-007. Command: `uv run rag bench vectorstores --k 10`._

| Candidate | build (s) | p50 (ms) | p95 (ms) | recall@10 vs flat | filtering | disk (MB) | RSS (MB) |
|---|---|---|---|---|---|---|---|
| chroma | | | | | native | | |
| faiss flat | | | | 1.0 | manual (post-filter) | | |
| faiss hnsw | | | | | manual (post-filter) | | |

## Decision

_Pending._ Expected shape of the answer: FAISS wins on raw speed and control over index type; Chroma wins on metadata filtering and developer ergonomics; at this scale a flat index is already fast enough, so filtering ergonomics probably decide it.

## Interview one-liner

_Pending._
