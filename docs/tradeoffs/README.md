# Tradeoff pages

One page per "X vs Y" question. A page is a **draft** until it has numbers measured on this corpus; then the decision is recorded in an ADR and the page is marked **decided**.

Use `_template.md` for new pages.

| Page | Question | Status | Ticket |
|---|---|---|---|
| `parsing.md` | custom parser vs sec-parser vs edgartools vs unstructured | **decided** | RAG-004 |
| `chunking.md` | fixed vs recursive vs section-aware vs parent-child | **decided** (ADR-009) | RAG-005, RAG-020 |
| `embeddings.md` | which model, and what text to embed | draft (v1 measured; task prefixes and context headers) | RAG-006, RAG-008 |
| `vector-stores.md` | ChromaDB vs FAISS (vs LanceDB / Qdrant / pgvector) | draft | RAG-007 |
| `retrieval-strategies.md` | dense vs BM25 vs hybrid vs hybrid+rerank | **decided** (ADR-008) | RAG-009 |
| `llm-serving.md` | which local model; local vs hosted | draft (candidates + criteria set) | RAG-002, RAG-012 |
| `orchestration.md` | plain Python vs LangChain vs LlamaIndex vs LangGraph | draft | RAG-010 |
| `evaluation.md` | custom judge vs RAGAS vs DeepEval | draft | RAG-012 |
| `observability.md` | Langfuse vs Phoenix vs MLflow | draft | RAG-013 |
