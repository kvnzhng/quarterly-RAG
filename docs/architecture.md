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
Chunk           id, text, ticker, form, fiscal_period, filing_date, section, char_start, char_end, source_url
RetrievedChunk  Chunk + score + retriever ("dense" | "bm25" | "hybrid") + rank
Answer          text, citations: list[Citation], unsupported_sentences: list[str], confidence
Citation        chunk_id, quote, char_start, char_end
Refusal         reason: "low_confidence" | "out_of_scope" | "insufficient_evidence" | "verification_failed", detail, best_chunks
```

## Component choices and alternatives

Every row is a tradeoff page under `docs/tradeoffs/`. "Chosen" is provisional until the page has numbers.

| Layer | Chosen (provisional) | Alternatives to compare | Tradeoff page | Ticket |
|---|---|---|---|---|
| Data source | SEC EDGAR HTML filings | XBRL financial data API, IR PDFs, transcripts | ADR-004 | RAG-003 |
| Parsing | custom HTML -> sections (BeautifulSoup/lxml) | `sec-parser`, `edgartools`, unstructured | `parsing.md` | RAG-004 |
| Chunking | section-aware + parent-child | fixed tokens, recursive, semantic | `chunking.md` | RAG-005 |
| Embeddings | `nomic-embed-text` via the configured embed endpoint (Ollama by default) | `bge-m3`, `all-MiniLM-L6-v2`, `e5` via sentence-transformers | `embeddings.md` | RAG-006 |
| Vector store | ChromaDB | FAISS (flat, HNSW), LanceDB, Qdrant (docker), pgvector | `vector-stores.md` | RAG-007 |
| Sparse retrieval | rank_bm25 | Elasticsearch/OpenSearch, SPLADE | `retrieval-strategies.md` | RAG-009 |
| Reranker | `bge-reranker-base` (cross-encoder) | ColBERT, LLM rerank, none | `retrieval-strategies.md` | RAG-009 |
| LLM | any OpenAI-compatible server, Ollama `llama3.1:8b` by default | other local 7B-8B models; hosted OpenAI-compatible or Anthropic API on the same eval set | `llm-serving.md` | RAG-002 |
| Orchestration | plain Python + LangChain components | full LangChain/LCEL, LlamaIndex, Haystack, LangGraph | `orchestration.md` | RAG-010 |
| Evaluation | custom metrics + local LLM judge | RAGAS, DeepEval, TruLens | `evaluation.md` | RAG-008, RAG-012 |
| Observability | Langfuse (self-hosted) | Arize Phoenix, MLflow tracing, OpenTelemetry only | `observability.md` | RAG-013 |
| Serving | FastAPI + Streamlit | Gradio, Chainlit | - | RAG-014 |

## Request flow for `rag ask`

1. Parse question -> optional metadata filters (ticker, fiscal period, section hints).
2. Retrieve top-k from dense and BM25, fuse with reciprocal rank fusion, rerank.
3. Refusal gate, stage 1: if best rerank score < threshold or filters match nothing -> `Refusal(low_confidence | out_of_scope)`.
4. Generate with a grounded prompt: chunks tagged `[c<id>]`, instructions to cite every sentence and to answer "insufficient evidence" when needed.
5. Verify: every sentence has a citation resolving to a retrieved chunk; numbers in the sentence appear in the cited chunk.
6. Refusal gate, stage 2: generator said insufficient evidence, or verification failed for the core claim -> `Refusal(insufficient_evidence | verification_failed)`.
7. Return `Answer` with citations and any flagged unsupported sentences. Everything is traced to Langfuse.
