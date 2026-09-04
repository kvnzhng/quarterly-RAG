# Architecture

## Pipeline

```
                 +-----------+    +----------+    +----------+    +-----------+    +------------+
 EDGAR filings ->| ingestion |--->| chunking |--->| indexing |--->| retrieval |--->| generation |---> Answer
 (10-Q, 10-K)    +-----------+    +----------+    +----------+    +-----------+    +------------+      or
                  html->text       Chunker         Embedder        dense + BM25     grounded prompt    Refusal
                  section split    protocol        VectorStore     RRF fusion       citation check     (with reason)
                  provenance       (4 strategies)  (Chroma|FAISS)  metadata filter  refusal gate
                                                                   reranker
                 \___________________________ evaluation (offline) + observability (Langfuse traces) ___________/
```

Each box is a package under `src/quarterly_rag/`. A layer may import from layers to its left only.

## Data model (crosses every boundary)

```
Chunk           chunk_id, strategy, text, word_count, contains_table, plus the section record's provenance
                (ticker, cik, company, form, accession, filing_date, period_of_report, fiscal_year,
                fiscal_quarter, period_label, part, item, section, title, source_url, text_path)
                and char_start/char_end into the same <accession>.txt sections and gold spans use
RetrievedChunk  Chunk + score + retriever ("dense" | "bm25" | "hybrid") + rank
Answer          text, citations: list[Citation], unsupported_sentences: list[str], derived_numbers: list[DerivedNumber], confidence
Citation        chunk_id, quote, char_start, char_end
Refusal         reason: "low_confidence" | "out_of_scope" | "insufficient_evidence" | "verification_failed", detail, best_chunks
EvalQuestion    id, question, ticker, type: "lookup" | "derived" | "cross_period" | "unanswerable", gold_answer, evidence: list[Span]
Span            accession, section, char_start, char_end   (labels are spans, not chunk ids; chunk relevance = overlap)
RunRecord       git_commit, corpus_hash, parser_version, chunker + config, embed provider + model, vector_store, retrieval params, prompt_version, llm provider + model, timestamp
```

Every eval report embeds a `RunRecord`; every number quoted in `docs/` points at one.

## Component choices and alternatives

Every row is a tradeoff page under `docs/tradeoffs/`. A row is **decided** once the page has numbers from this corpus and an ADR records the choice.

| Layer | Decision | Status | Alternatives measured or noted | Page |
|---|---|---|---|---|
| Data source | SEC EDGAR primary documents, 8 quarters per company | decided, ADR-004 | XBRL facts, IR PDFs, transcripts (out of scope) | ADR-004 |
| Parsing | custom block-boundary parser (BeautifulSoup/lxml) | decided, ADR-007 | `sec-parser`, `edgartools`, `unstructured` (not benchmarked; coverage was full first time) | `parsing.md` |
| Chunking | section-aware: cut on the filing's own sub-headings | decided, ADR-009 | fixed, recursive, parent-child (all measured) | `chunking.md` |
| Embeddings | `nomic-embed-text` with task prefixes, provenance header prepended | provisional, ADR-006 | other models not yet measured; the two embed-text variants were | `embeddings.md` |
| Vector store | ChromaDB | decided, ADR-010 | FAISS flat and HNSW (measured, kept for scale) | `vector-stores.md` |
| Retrieval | hybrid dense+BM25, RRF, ticker and quarter filters | decided, ADR-008 | dense, BM25, no filter, LLM rerank (all measured; rerank is off) | `retrieval-strategies.md` |
| Reranker | none by default; LLM reranker available | decided, ADR-008 | cross-encoder deferred: 2 GB dependency for a gain that shows at k=1, not k=5 | `retrieval-strategies.md` |
| LLM | `llama3.1:8b` default for laptops; `qwen3.8-27b` recommended | provisional, ADR-006 | four local models measured on citation discipline and correctness | `llm-serving.md` |
| Orchestration | plain Python; no LangChain anywhere | in practice decided | LangChain, LlamaIndex, Haystack, LangGraph | `orchestration.md` (draft) |
| Evaluation | span-labelled eval set, deterministic figure check, cross-model judge | decided | RAGAS (measured, anti-correlated, rejected) | `evaluation.md` |
| Observability | Langfuse (self-hosted) | planned | Arize Phoenix, MLflow tracing | `observability.md` (draft) |
| Serving | FastAPI + Streamlit | planned | Gradio, Chainlit | RAG-014 |

## Request flow for `rag ask` (as built, `pipeline.py`)

1. **Scope check, before any model call.** A company the corpus does not hold, a fiscal year before the corpus, or a question filings never answer (advice, live prices, transcripts) -> `Refusal(out_of_scope)`.
2. **Read the question** for a company and a fiscal period (`retrieval/query.py`). One named company becomes a ticker filter; a named quarter becomes a period filter; a bare year is never filtered, because filings quote prior years.
3. **Retrieve** the top 50 from dense search and from BM25 (which indexes the chunk plus its provenance header and sees the question expanded with `FY2026 Q3`-style terms), fuse by reciprocal rank, keep 5. A filter that empties the result falls back to no filter.
4. **Retrieval gate.** `MIN_RETRIEVAL_SCORE` is 0 by default: the sweep showed the threshold is worse than useless on this corpus. Kept as a setting.
5. **Generate** with passages tagged `[c1]`..`[c5]`; the prompt requires a citation on every sentence and the sentinel `INSUFFICIENT_EVIDENCE` when the passages do not answer.
6. **Verify deterministically.** Each sentence must cite a passage that was actually provided; each figure must appear in the cited passage after unit scaling. A figure that does not is labelled `derived, unverified`, not rejected. RAG-021 will recompute those from their operands.
7. **Answer gate.** The sentinel -> `Refusal(insufficient_evidence)`; no resolvable citation anywhere -> `Refusal(verification_failed)`.
8. **Return** the `Answer` with citations and inline markers, or the `Refusal` with its reason and the closest passages. Tracing to Langfuse is RAG-013.
