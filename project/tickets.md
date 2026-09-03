# Tickets -- quarterly-RAG (Prefix: RAG)

> Next ID: RAG-019

Tickets are grouped by the competency they demonstrate. Each ticket names the
artifact it must leave behind (code, an eval number, a tradeoff doc, or an ADR)
so the repo tells the story on its own.

## In Progress

## Backlog

### RAG-002: Model clients, `rag doctor`, and local model setup
- **Type:** chore
- **Created:** 2026-09-03
- **Competency:** foundation
- **Description:** Implement the `LLM` and `Embedder` protocols with the `openai_compatible` provider (Ollama and other local servers, plus hosted OpenAI-style APIs) and the `anthropic` provider (ADR-005). Add `rag doctor`: configured endpoint reachable, configured models listed by the server, one chat round-trip and one embedding call succeed, data dirs writable. `make models` pulls the default Ollama models for people running Ollama themselves (honours `OLLAMA_HOST` for a remote Ollama). Kevin's local AI server address is provided at ticket start and goes in `.env`, never in the repo.
- **Done when:** `rag doctor` passes against a local Ollama and against a remote OpenAI-compatible server; unit tests mock the HTTP layer.
- **Artifacts:** `docs/tradeoffs/llm-serving.md` first pass (which local models were tried and why), ADR-006 model selection.
### RAG-003: SEC EDGAR filing downloader
- **Type:** feat
- **Created:** 2026-09-03
- **Competency:** grounding (the corpus is the ground truth)
- **Description:** `rag ingest download --ticker AAPL --forms 10-Q,10-K --since 2023-01-01`. Resolve ticker to CIK via EDGAR, list filings from the submissions API, download the primary document, and write a manifest (`data/raw/<ticker>/manifest.json`) with accession number, form type, period of report, filing date, and source URL. Respect EDGAR fair-access rules (declared User-Agent, max 10 req/s).
- **Done when:** Apple and Nvidia 10-Q/10-K filings for the last 8 quarters are on disk with a manifest, re-running is idempotent.

### RAG-004: Filing parser with section detection
- **Type:** feat
- **Created:** 2026-09-03
- **Competency:** grounding, chunking
- **Description:** Convert filing HTML to clean text. Detect SEC items (Item 1, 1A Risk Factors, 2/7 MD&A, 3 Quantitative disclosures, 8 Financial Statements). Preserve tables as pipe-delimited text with a table marker. Emit `data/processed/<ticker>/<accession>.jsonl`, one record per section, with provenance fields (ticker, form, period, section, char offsets, source URL).
- **Done when:** parser tests pass on one 10-Q and one 10-K per company, and a section coverage report shows every expected item was found.

### RAG-005: Chunking strategies and chunking experiment harness
- **Type:** feat
- **Created:** 2026-09-03
- **Competency:** chunking
- **Description:** Implement pluggable chunkers: fixed-size token window, recursive character, section-aware (never cross an Item boundary), and parent-child (small chunks for retrieval, parent section returned for generation). Every chunk keeps full provenance. Add a harness that reports chunk count, size distribution, and boundary violations per strategy.
- **Done when:** `docs/tradeoffs/chunking.md` contains a filled comparison table and a recommendation, backed by retrieval metrics from RAG-008.

### RAG-006: Embeddings, VectorStore interface, ChromaDB adapter
- **Type:** feat
- **Created:** 2026-09-03
- **Competency:** retrieval quality
- **Description:** Define a `VectorStore` protocol (add, query with metadata filter, persist, load, stats). Implement the ChromaDB adapter with persistence under `data/indexes/`. Embeddings via Ollama (`nomic-embed-text`) behind an `Embedder` protocol so sentence-transformers models can be swapped in.
- **Done when:** `rag index build --ticker AAPL --store chroma` builds and reloads an index, and a query returns chunks with provenance.
- **Artifacts:** `docs/tradeoffs/embeddings.md` first pass.

### RAG-007: FAISS adapter and Chroma vs FAISS benchmark
- **Type:** feat
- **Created:** 2026-09-03
- **Competency:** retrieval quality
- **Description:** Implement a FAISS adapter (flat and HNSW) behind the same protocol. Benchmark both on the same corpus: build time, query p50/p95 latency, memory, metadata filtering support, persistence story, operational complexity.
- **Done when:** `docs/tradeoffs/vector-stores.md` is filled with measured numbers and ADR-007 records the default choice and when to pick the other.

### RAG-008: Retrieval evaluation set and metrics
- **Type:** feat
- **Created:** 2026-09-03
- **Competency:** retrieval quality
- **Description:** Build `data/eval/retrieval.jsonl`: 50+ questions across both companies with gold answer spans and gold chunk ids (LLM-assisted drafting, human verified). Implement recall@k, MRR, nDCG@k. `rag eval retrieval --k 5` prints a table and writes `reports/retrieval-<timestamp>.json`.
- **Done when:** the eval runs end to end and the first baseline numbers are recorded in `docs/learning/retrieval-quality.md`.

### RAG-009: Hybrid retrieval, metadata filtering, and reranking
- **Type:** feat
- **Created:** 2026-09-03
- **Competency:** retrieval quality
- **Description:** Add BM25 (rank_bm25) alongside dense retrieval with reciprocal rank fusion. Add metadata filters inferred from the question (ticker, fiscal period, section). Add a cross-encoder reranker (`bge-reranker-base`). Compare dense / BM25 / hybrid / hybrid+rerank on the RAG-008 eval.
- **Done when:** the comparison table is in `docs/tradeoffs/retrieval-strategies.md` and the best configuration becomes the default.

### RAG-010: Grounded answer generation with verified citations
- **Type:** feat
- **Created:** 2026-09-03
- **Competency:** grounding, hallucination control
- **Description:** Prompt the LLM with retrieved chunks tagged by id and require inline citations `[c12]`. Post-process: every claim sentence must carry a citation that maps to a retrieved chunk, numbers quoted must appear verbatim in the cited chunk. Return a structured `Answer {text, citations, unsupported_sentences}`.
- **Done when:** `rag ask "What was Apple's Q2 FY24 revenue?"` returns an answer whose citations resolve to real chunks, and unsupported sentences are flagged rather than silently returned.

### RAG-011: Refusal policy and abstention evaluation
- **Type:** feat
- **Created:** 2026-09-03
- **Competency:** when to refuse to answer
- **Description:** Implement a refusal gate with explicit reasons: (a) retrieval confidence below threshold, (b) question outside corpus scope (company or period not indexed, non-financial question), (c) generator reports insufficient evidence, (d) citation verification fails. Build `data/eval/unanswerable.jsonl` (questions that must be refused) and measure abstention precision/recall alongside answer accuracy.
- **Done when:** `docs/learning/refusal.md` reports the tradeoff curve between refusing too much and hallucinating, and thresholds are chosen from it.

### RAG-012: Faithfulness and end-to-end evaluation
- **Type:** feat
- **Created:** 2026-09-03
- **Competency:** hallucination control
- **Description:** Add an LLM-as-judge faithfulness check (claims in answer entailed by cited context) using a local model, and compare against RAGAS. Add answer correctness against gold answers. `make eval` runs retrieval + generation evals and fails if scores regress below a stored baseline.
- **Done when:** `docs/tradeoffs/evaluation.md` compares RAGAS vs custom judge, and `make eval` is wired into CI as an optional job.

### RAG-013: Langfuse tracing (self-hosted)
- **Type:** chore
- **Created:** 2026-09-03
- **Competency:** production readiness
- **Description:** Run Langfuse locally via docker compose (`infra/`). Trace every `rag ask` as ingestion -> retrieval -> rerank -> generation -> verification spans with token counts and latency. Push eval scores to Langfuse as scores on traces.
- **Done when:** a trace for a refused question and an answered question are visible in the local Langfuse UI. `docs/tradeoffs/observability.md` compares Langfuse vs Phoenix vs MLflow.

### RAG-014: API and minimal UI
- **Type:** feat
- **Created:** 2026-09-03
- **Competency:** production readiness
- **Description:** FastAPI `POST /ask` returning the structured answer or refusal. Streamlit page that shows the answer, each citation with the highlighted source passage and filing link, and the refusal reason when refused.
- **Done when:** `make api` and `make ui` work locally and the README has a screenshot.

### RAG-015: Results writeup and interview talking points
- **Type:** docs
- **Created:** 2026-09-03
- **Competency:** all
- **Description:** Fill the README with the architecture diagram, final eval tables, and one paragraph per competency (grounding, chunking, retrieval quality, hallucination control, refusal) explaining what was tried, what was measured, and what was chosen.
- **Done when:** a reader can understand the tradeoffs from the README alone.

## Done

### RAG-001: Project initialization
- **Type:** chore
- **Created:** 2026-09-03 | **Completed:** 2026-09-03
- **Description:** Set up project scaffolding with ticket-based workflow, Python tooling (uv, ruff, pytest, CI), src layout, docs skeleton (ADRs, learning notes, tradeoff docs), and the roadmap below.
- **Commits:** `5094a17`

### RAG-016: Rename project to quarterly-RAG
- **Type:** chore
- **Created:** 2026-09-03 | **Completed:** 2026-09-03
- **Description:** Public name is `quarterly-RAG` (GitHub: kvnzhng/quarterly-RAG). Rename the Python package to `quarterly_rag`, the distribution to `quarterly-rag`, and update docs, config defaults, and the README badge/links. CLI command stays `rag`.
- **Commits:** `606872e`

### RAG-017: Reading list and course material
- **Type:** docs
- **Created:** 2026-09-03 | **Completed:** 2026-09-03
- **Description:** Add a curated, link-checked reading and course list to the README, grouped by competency and mapped to tickets, plus a short Reading section on each `docs/learning/` page.
- **Commits:** `d395fb5`

### RAG-018: Provider-agnostic LLM and embedding configuration
- **Type:** chore
- **Created:** 2026-09-03 | **Completed:** 2026-09-03
- **Description:** The model provider is the user's choice: a local server (Ollama or any OpenAI-compatible endpoint, on this machine or on the network), or a hosted API with a token (OpenAI-compatible or Anthropic). Replace the Ollama-only settings with `LLM_PROVIDER` / `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` and matching `EMBED_*` settings, keep local as the default, update `.env.example`, README, Makefile, RAG-002, and record the decision in ADR-005.
- **Commits:** `a7c6d72`
