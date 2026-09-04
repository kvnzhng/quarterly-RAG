# ADR-008: Hybrid retrieval is the default; reranking is available and off

**Date:** 2026-09-04
**Status:** accepted
**Ticket:** RAG-009

## Context

Dense retrieval alone reached 36.4% recall@5 and answered none of the seven quarterly questions (RAG-008). Because the refusal gate declines when evidence is missing (RAG-011), answer coverage cannot exceed retrieval, so this was the binding constraint on the whole system.

## Decision

- **`hybrid` is the default retrieval strategy**: reciprocal rank fusion of dense and BM25, candidate pool 50, fusion constant 60. It reaches 45.5% recall@5 against dense's 36.4%.
- **BM25 indexes the chunk plus its provenance header**, and the question is expanded with the corpus's own spelling of any period or company it names. A chunk says "June 27, 2026" and never "FY2026 Q3"; without expansion the two share no token.
- **The tokenizer keeps figures and period labels whole.** A dollar amount, the same amount in parentheses, and the bare digits all become one token, and `FY2026` and `Q3` survive as single tokens rather than splitting into letters and digits. This is the whole result: a tokenizer that splits them makes BM25 pointless on financial text.
- **Metadata filtering is implemented and off by default.** It changes recall not at all on a two-company corpus, because retrieval already reaches the right filing 91% of the time. It would matter at fifty companies.
- **Reranking uses the configured chat model rather than a cross-encoder, and is off by default.** It raises recall@1 from 18% to 30% and lowers recall@5 from 46% to 39%, and the generator receives five passages. A dedicated cross-encoder means PyTorch, about two gigabytes, against ADR-003's promise that the defaults fit a laptop; these numbers do not justify that yet.
- **Fusion happens on rank, never on score.** Cosine similarity and BM25 are not commensurable, and reciprocal rank fusion needs no per-retriever tuning.

## Consequences

- The BM25 index is built in memory from the chunk files at every process start. At 1,391 chunks that is under a second; a corpus large enough for that to hurt wants a real inverted index, and that is the scaling boundary to watch.
- Retrieval is no longer the largest single constraint, but it is still a constraint: pushing hybrid to depth 100 reaches only 69.7%, so ten of 33 questions have their evidence nowhere in the top 100 of 1,391 chunks. Those are mostly financial-statement tables, and the next lever is chunking (RAG-020), not ranking.
- Every retriever implements one protocol, so `rag eval retrieval --retrieval <strategy>` compares them with the same metrics, and the run record names which one produced a number.
