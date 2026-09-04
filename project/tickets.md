# Tickets -- quarterly-RAG (Prefix: RAG)

> Next ID: RAG-027

Tickets are grouped by the competency they demonstrate. Each ticket names the
artifact it must leave behind (code, an eval number, a tradeoff doc, or an ADR)
so the repo tells the story on its own.

The backlog is ordered. Phase 1 ships one thin end-to-end path with the eval
set in place before anything is optimised. Phase 2 compares alternatives
against that baseline. Phase 3 is production readiness and the writeup.
Reordered on 2026-09-04 after an external review (see `docs/notes.md`).

## In Progress

## Backlog

### RAG-020: Chunking strategy comparison
- **Type:** feat
- **Created:** 2026-09-04
- **Competency:** chunking
- **Description:** Split from RAG-005. Add recursive character, section-aware with sub-splitting, and parent-child (small chunks for retrieval, parent section returned for generation) chunkers behind the `Chunker` protocol, each keeping full provenance. Extend the harness to report chunk count, size distribution, and boundary violations per strategy, and score every strategy with RAG-008 on the same RAG-019 labels.
- **Done when:** `docs/tradeoffs/chunking.md` contains a filled comparison table with run records and a recommendation, and an ADR records the default chunker.

### Phase 1: one thin end-to-end path, measured

### Phase 2: compare alternatives against the baseline

### RAG-007: FAISS adapter and Chroma vs FAISS benchmark
- **Type:** feat
- **Created:** 2026-09-03
- **Competency:** retrieval quality
- **Description:** Implement a FAISS adapter (flat and HNSW) behind the same protocol. Benchmark both on the same corpus: build time, query p50/p95 latency, memory, metadata filtering support, persistence story, operational complexity.
- **Done when:** `docs/tradeoffs/vector-stores.md` is filled with measured numbers and ADR-007 records the default choice and when to pick the other.

### RAG-012: Faithfulness and end-to-end evaluation
- **Type:** feat
- **Created:** 2026-09-03
- **Competency:** hallucination control
- **Description:** Add an LLM-as-judge faithfulness check (claims in answer entailed by cited context) using a local model, and compare against RAGAS. Add answer correctness against gold answers. `make eval` runs retrieval + generation evals and fails if scores regress below a stored baseline.
- **Done when:** `docs/tradeoffs/evaluation.md` compares RAGAS vs custom judge, and `make eval` is wired into CI as an optional job.

### RAG-021: Calculation provenance for derived numbers
- **Type:** feat
- **Created:** 2026-09-04
- **Competency:** hallucination control
- **Description:** `derived` and `cross_period` questions need arithmetic (growth rates, differences, ratios, unit conversions), and a verbatim number check passes a wrong relationship between two correct numbers. Extend the answer format so the generator emits each derived number as a calculation: operands with citations, the operation, and the result. A deterministic verifier recomputes it from the cited operands and marks the result `verified` only when the recomputation matches within rounding; a growth rate built from the wrong two periods then fails instead of passing.
- **Done when:** the RAG-019 `derived` and `cross_period` questions are scored separately, and `docs/learning/hallucination-control.md` reports the verified rate before and after.

### Phase 3: production readiness and writeup

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

### RAG-022: Ticket enforcement portable across clones and CI
- **Type:** chore
- **Created:** 2026-09-04 | **Completed:** 2026-09-04
- **Competency:** foundation
- **Description:** The commit-msg hook lives only in this checkout's `.git/hooks`, `make setup` installs only the pre-commit stage, CI does not check messages, and the Claude edit hook needs `jq`. Track the check as `scripts/check-commit-msg.sh`, wire it as a pre-commit `commit-msg` stage hook, install both stages from `make setup`, run the same script over the commit range in CI, and let `enforce-ticket.sh` fall back to `python3` when `jq` is missing. Commit `AGENTS.md` as a symlink to `CLAUDE.md` so Codex reads the same instructions.
- **Done when:** a fresh clone gets the commit-msg hook from `make setup`, a commit without a ticket id is rejected locally and in CI, and the existing history passes the CI check.
- **Commits:** `bdd176a`

### RAG-023: README, docs, and conventions match the reviewed plan
- **Type:** docs
- **Created:** 2026-09-04 | **Completed:** 2026-09-04
- **Competency:** all
- **Description:** The review found the README describing planned work in the present tense, a duplicated clone command, a layout line naming "RAG-001 ... RAG-015", and a status list missing RAG-016 to RAG-018; `project/conventions.md` still says the provider is "Ollama today". Fix these, add the run-record requirement to conventions, and update ticket references in `docs/`, `README.md`, and package docstrings for the RAG-005/RAG-020 split and the new RAG-019 and RAG-021.
- **Done when:** every ticket reference in `docs/` and `README.md` resolves to the right ticket and the README status list matches this file.
- **Commits:** `8e81db6`

### RAG-002: Model clients, `rag doctor`, and local model setup
- **Type:** chore
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** foundation
- **Description:** Implement the `LLM` and `Embedder` protocols with the `openai_compatible` provider (Ollama and other local servers, plus hosted OpenAI-style APIs) and the `anthropic` provider (ADR-005). Add `rag doctor`: configured endpoint reachable, configured models listed by the server, one chat round-trip and one embedding call succeed, data dirs writable. `make models` pulls the default Ollama models for people running Ollama themselves (honours `OLLAMA_HOST` for a remote Ollama). Kevin's local AI server address is provided at ticket start and goes in `.env`, never in the repo.
- **Done when:** `rag doctor` passes against a local Ollama and against a remote OpenAI-compatible server; unit tests mock the HTTP layer.
- **Artifacts:** `docs/tradeoffs/llm-serving.md` first pass (which local models were tried and why), ADR-006 model selection.
- **Dependencies added:** `anthropic` (official SDK for the `anthropic` provider: retries, typed errors, and it tracks API changes such as adaptive thinking and removed sampling parameters; a hand-rolled client would need re-verifying on every change). The OpenAI-compatible provider uses the existing `httpx`.
- **Verified:** `rag doctor` and the live integration test pass against Ollama 0.32.13 on the network server (address in `.env`); cold chat 3.8 s, warm 242 ms; embeddings 768-dim. Not exercised: Ollama on this laptop (not installed). `make models` now pulls over Ollama's HTTP API so no local CLI is needed.
- **Commits:** `b34731a`, `b17ae67`, `f0e9fb4`

### RAG-003: SEC EDGAR filing downloader
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** grounding (the corpus is the ground truth)
- **Description:** `rag ingest download --ticker AAPL --forms 10-Q,10-K --since 2023-01-01`. Resolve ticker to CIK via EDGAR, list filings from the submissions API, download the primary document, and write a manifest (`data/raw/<ticker>/manifest.json`) with accession number, form type, period of report, filing date, and source URL. Respect EDGAR fair-access rules (declared User-Agent, max 10 req/s).
- **Done when:** Apple and Nvidia 10-Q/10-K filings for the last 8 quarters are on disk with a manifest, re-running is idempotent.
- **Verified:** `rag ingest download --ticker AAPL --ticker NVDA` fetched 8 filings per company for the default two-year window (exactly eight quarters each, fiscal labels checked against the companies' own naming); the second run downloaded nothing and left both manifests byte-identical. Live EDGAR test in `tests/integration/`. `make test-all`: 72 passed.
- **Commits:** `de44ec4`

### RAG-004: Filing parser with section detection
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** grounding, chunking
- **Description:** Convert filing HTML to clean text. Detect SEC items (Item 1, 1A Risk Factors, 2/7 MD&A, 3 Quantitative disclosures, 8 Financial Statements). Preserve tables as pipe-delimited text with a table marker. Emit `data/processed/<ticker>/<accession>.jsonl`, one record per section, with provenance fields (ticker, form, period, section, char offsets, source URL).
- **Done when:** parser tests pass on one 10-Q and one 10-K per company, and a section coverage report shows every expected item was found.
- **Dependencies added:** `beautifulsoup4` and `lxml` (HTML parsing; `lxml` is the fast, lenient tree builder these 1-2 MB inline-XBRL documents need). Both are runtime dependencies, not dev-only.
- **Verified:** all 16 filings parse with zero missing critical items and zero false headings; every 10-K yields its 23 expected Items, Apple's 10-Qs 11, Nvidia's 9 (Part II Items 3 and 4 genuinely absent). `text[char_start:char_end] == record.text` holds for every record. `make test-all`: 91 passed.
- **Commits:** `1c1027c`, `2e2efc1`

### RAG-024: Section key renders Part IV as "Part IIII"
- **Type:** fix
- **Created:** 2026-09-04 | **Completed:** 2026-09-04
- **Competency:** grounding
- **Description:** `Section.key` builds the roman numeral with `"I" * part`, so Part IV becomes `Part IIII`. It shows on every 10-K Item 15 and 16 record, which is where Nvidia files its consolidated financial statements. Section keys are the handle for gold evidence spans (RAG-019) and citations (RAG-010), so fix before eval data is generated against them.
- **Done when:** `Part IV.Item 15` renders correctly, a test covers all four parts, and the corpus is re-parsed.
- **Commits:** `ef148b3`

### RAG-019: Evaluation set v0 with evidence spans and question types
- **Type:** feat
- **Created:** 2026-09-04 | **Completed:** 2026-09-04
- **Competency:** retrieval quality, hallucination control, refusal
- **Description:** Build `data/eval/questions.jsonl` before any chunker or index exists, so every later choice is measured against the same labels. Each record: `id`, `question`, `ticker`, `type` (`lookup` | `derived` | `cross_period` | `unanswerable`), `gold_answer`, and `evidence`: a list of spans `{accession, section, char_start, char_end}` into the RAG-004 output. Labels are spans, not chunk ids, so they survive a change of chunking strategy; a chunk counts as relevant when it overlaps a span. Start with 30 answerable questions (mostly `lookup`, a few `derived` and `cross_period`) and 10 `unanswerable` seeds across both companies. LLM-assisted drafting is fine; every record is human-verified against the filing. A loader and `rag eval check` verify that every span resolves to text in the processed filings and that the gold answer appears inside the evidence for `lookup` questions (test marked `integration` until sample filings are committed, see the open question in `docs/notes.md`).
- **Done when:** `rag eval check` reports every span resolves, and the set is committed under `data/eval/`.
- **Verified:** 43 questions, every one reviewed against the filing text by Kevin in two rounds (10, then 33). 23 lookup, 5 derived, 5 cross-period, 10 unanswerable split evenly between `out_of_scope` and `insufficient_evidence`. 35/35 spans resolve; `rag eval check` is the gate. Three findings from round two were label presentation, not facts, and are fixed.
- **Known limits for RAG-008:** evidence concentrates in 6 of the 16 filings, so the other 10 act as distractors and period filtering is under-tested; 30% of spans are prose and the rest tables.
- **Commits:** `fbb5f19`, `781ff64`, `d0159d5`

### RAG-005: Chunker protocol and v1 chunker
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** chunking
- **Description:** Define the `Chunker` protocol and the `Chunk` model with mandatory provenance (ticker, form, period, section, char offsets, source URL). Implement one chunker: a fixed token window with overlap applied within each RAG-004 section record, so no chunk crosses an Item boundary and a table is never split. Report chunk count and size distribution. The other strategies and the comparison are RAG-020.
- **Done when:** chunks from one 10-Q round-trip to JSONL with provenance intact, offsets map back into the section text, and unit tests cover the boundary and table rules.
- **Verified:** 1,391 chunks over the 16-filing corpus, median 304 words, p90 347, largest 809. `filing_text[char_start:char_end] == chunk.text` holds for every chunk, none crosses a section boundary, none holds half a table, ids are unique, and all 35 gold evidence spans overlap at least one chunk. 61 chunks exceed the target, every one of them a single table kept whole. `make test-all`: 128 passed.
- **Observations for RAG-020:** Nvidia's Part IV Item 15 yields 61 chunks in the FY2026 10-K alone, all sharing one section label, so a section filter buys nothing there; one gold span (q003) already straddles two chunks, which is the case parent-child chunking exists to fix; and overlap is whole-line, so 398 of 1,179 boundaries get none (325 because the preceding line exceeds the 60-word budget, 73 because a table is never carried forward).
- **Commits:** `6ec7cd3`, `b89bff6`

### RAG-006: Embeddings, VectorStore interface, ChromaDB adapter, dense retrieval
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** retrieval quality
- **Description:** Define a `VectorStore` protocol (add, query with metadata filter, persist, load, stats). Implement the ChromaDB adapter with persistence under `data/indexes/`. Embeddings come from the configured embed endpoint (ADR-005; `nomic-embed-text` on Ollama by default) behind the `Embedder` protocol so sentence-transformers models can be swapped in. Add `retrieve(question, k)` returning `RetrievedChunk`s from dense search only; BM25, filters, and reranking are RAG-009.
- **Done when:** `rag index build --ticker AAPL --store chroma` builds and reloads an index, and a query returns chunks with provenance.
- **Artifacts:** `docs/tradeoffs/embeddings.md` first pass.
- **Verified:** `rag index build --ticker AAPL --ticker NVDA` embeds 1,391 chunks into ChromaDB in 13 s per variant (768 dims, unit-normalised, cosine); reopening the directory finds them; `rag index query` returns ranked chunks whose text still resolves against the filing offsets. `make test-all`: 155 passed.
- **Found by measuring:** `nomic-embed-text` needs `search_query:` / `search_document:` prefixes and was getting neither, costing a third of recall with no error. The `Embedder` protocol is now `embed_documents` / `embed_query` so the mistake is not expressible. Prepending a company/period/section header before embedding roughly doubles recall; both variants are built and kept for RAG-008.
- **Baseline:** dense-only recall@5 is 18.2% raw and 36.4% with context headers, over 33 answerable questions. That is the floor RAG-009's BM25, fusion and reranking are measured against.
- **Dependencies added:** `chromadb` (embedded vector store with metadata filtering and directory persistence; FAISS goes behind the same protocol in RAG-007).
- **Commits:** `3435fc9`

### RAG-008: Retrieval metrics, run record, and baseline numbers
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** retrieval quality
- **Description:** Implement recall@k, MRR, and nDCG@k over the RAG-019 set, with a chunk counted relevant when it overlaps a gold evidence span (overlap rule configurable, default any overlap). Break results down by company, form, section, and question type. Every report embeds a **run record**: git commit, corpus manifest hash, parser version, chunker name and config, embedding provider and model, vector store, retrieval parameters (k, filters), prompt version where relevant, and timestamp. `rag eval retrieval --k 5` prints the table and writes `reports/retrieval-<timestamp>.json`.
- **Done when:** the eval runs end to end on the RAG-005 + RAG-006 baseline and the numbers, with their run record, are in `docs/learning/retrieval-quality.md`.
- **Verified:** `rag eval retrieval -k 5 [--context]` runs end to end on the RAG-005 + RAG-006 baseline, prints overall / near-miss / by-type / by-company / by-form tables and writes `reports/retrieval-<variant>-<timestamp>.json` with a full run record. Numbers are in `docs/learning/retrieval-quality.md`. `make test-all`: 180 passed.
- **Baseline:** context variant recall@5 36.4%, MRR 0.267, nDCG@5 0.232; raw variant 18.2%, 0.131, 0.096.
- **Findings for RAG-009:** the near-miss ladder shows retrieval reaching the right filing 90.9% of the time, the right section 63.6%, and the right chunk 36.4%, so the loss is inside the document and metadata filtering would buy little. All seven 10-Q questions score 0.0% against 46.2% for 10-K, because retrieval returns the management discussion of a filing instead of its condensed financial statements. Both point at exact-term matching (BM25) and reranking.
- **Commits:** `81a7c67`

### RAG-010: Grounded answer generation with verified citations
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** grounding, hallucination control
- **Description:** Prompt the LLM with retrieved chunks tagged by id and require inline citations `[c12]`. Post-process: every claim sentence must carry a citation that maps to a retrieved chunk. Numbers are checked against the cited chunk after parsing and unit scaling (thousands / millions / billions, rounding tolerance): a number found in the chunk is `verified`; one not found is marked `derived, unverified` and the answer says so, instead of being silently passed or hard-failed. Calculation provenance for derived numbers is RAG-021. Return a structured `Answer {text, citations, unsupported_sentences, derived_numbers}`. v1 targets `lookup` questions from RAG-019; `derived` and `cross_period` results are reported separately.
- **Done when:** `rag ask "What was Apple's Q2 FY24 revenue?"` returns an answer whose citations resolve to real chunks, unsupported sentences and derived numbers are flagged rather than silently returned, and baseline citation-resolution and number-match rates by question type are in `docs/learning/grounding.md`.
- **Verified:** `rag ask` returns an answer whose citations resolve to passages that were actually provided, with unsupported sentences and unverified figures labelled inline. Baselines by question type are in `docs/learning/grounding.md`. `make test-all`: 209 passed.
- **Baseline (23 lookup questions, prompt v1):** `gpt-oss:20b` on gold passages, citations resolve 100%, fully grounded 91%, states the gold figure 77%; on retrieved passages, 100% / 87% / 67% with 35% refused. `llama3.1:8b` on gold passages, 50% / 41% / 95%.
- **Finding:** citation discipline is a model capability. The 8B default invents passage labels in half its answers while being the best of three at finding the right figure. ADR-006 amended; `docs/tradeoffs/llm-serving.md` has the table. Set `LLM_MODEL=gpt-oss:20b` on hardware with room.
- **Known limit:** the verifier checks whether a figure is *present* in the cited passage, not whether the claim about it is true, so a wrong-column figure passes. RAG-021 recomputes derived numbers from their operands.
- **Commits:** `5c11778`

### RAG-025: Extend the citation-discipline comparison to qwen3.8-27b
- **Type:** docs
- **Created:** 2026-09-04 | **Completed:** 2026-09-04
- **Competency:** hallucination control, production readiness
- **Description:** RAG-010 measured three models and found citation discipline to be the largest single lever on grounding. The server also holds `qwen3.8-27b`, which was not in that run. Score it on the same 23 lookup questions in both contexts, add per-model answer latency, and update `docs/tradeoffs/llm-serving.md`, `docs/learning/grounding.md`, ADR-006 and the README recommendation with whatever the numbers say.
- **Done when:** the llm-serving table covers four models with a latency column and the recommended `LLM_MODEL` follows the measurement rather than a guess.
- **Verified:** four models scored on gold passages, two on retrieved, with latency. `qwen3.8-27b-64k` is 100% on every grounding measure end to end and 75% on the labelled figure, against 87% and 67% for `gpt-oss:20b`, at 9.6 s versus 3.8 s per answer. Recommendation and ADR-006 updated to match.
- **Commits:** `e041a64`

### RAG-011: Refusal policy and abstention evaluation
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** when to refuse to answer
- **Description:** Implement a refusal gate with explicit reasons: (a) retrieval confidence below threshold, (b) question outside corpus scope (company or period not indexed, non-financial question), (c) generator reports insufficient evidence, (d) citation verification fails. Grow the `unanswerable` seeds from RAG-019 into `data/eval/unanswerable.jsonl` (30+ questions that must be refused) and measure abstention precision/recall alongside answer accuracy.
- **Done when:** `docs/learning/refusal.md` reports the tradeoff curve between refusing too much and hallucinating, and thresholds are chosen from it.
- **Verified:** `docs/learning/refusal.md` carries the threshold sweep and names the operating point. `rag ask` refuses with a reason and shows its closest passages. `make test-all`: 235 passed.
- **Baseline (63 questions, 30 unanswerable, qwen3.8-27b-64k, k=5):** abstention precision 67.4%, recall 96.7%, F1 0.795, answerable coverage 57.6%, one leak.
- **Operating point:** `MIN_RETRIEVAL_SCORE=0.0`, the check off. The sweep shows raising it buys 3.3 points of recall and costs 45 points of coverage, because cosine scores cluster between 0.74 and 0.84. A calibrated signal would have to come from a reranker or token probabilities instead (RAG-009).
- **Deviation from the ticket:** the unanswerable questions live in `data/eval/questions.jsonl` rather than a separate `unanswerable.jsonl`. The schema already carries the type, so a second file would need a second loader, hash and check command for no gain.
- **Finding:** 13 of the 14 over-refusals had no evidence in the top 5, so coverage is bounded by retrieval rather than by the gate. Refusal calibration is also not answer quality: the 8B model has the best abstention F1 and invents citations in half its answers.
- **Commits:** `6a10b9a`

### RAG-009: Hybrid retrieval, metadata filtering, and reranking
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** retrieval quality
- **Description:** Add BM25 (rank_bm25) alongside dense retrieval with reciprocal rank fusion. Add metadata filters inferred from the question (ticker, fiscal period, section). Add a cross-encoder reranker (`bge-reranker-base`). Compare dense / BM25 / hybrid / hybrid+rerank on the RAG-008 eval.
- **Done when:** the comparison table is in `docs/tradeoffs/retrieval-strategies.md` and the best configuration becomes the default.
- **Verified:** `docs/tradeoffs/retrieval-strategies.md` holds the six-row comparison and ADR-008 records the decision. `hybrid` is the default in `Settings` and every command takes `--retrieval`. `make test-all`: 267 passed.
- **Baseline:** hybrid recall@5 45.5% against dense 36.4%, MRR 0.300 against 0.266.
- **Did not work, and worth knowing:** the inferred ticker filter changes recall not at all (the near-miss ladder had already said so); reranking raises recall@1 and lowers recall@5 with either judge, so it makes the system worse while making the ranking look better, and the case for a 2 GB cross-encoder is now weaker rather than stronger.
- **Finding for RAG-020:** every strategy plateaus at 45.5% by k=10, and hybrid at depth 100 reaches only 69.7%. Ten of 33 questions have their gold chunk nowhere in the top 100 of 1,391, mostly financial-statement tables. The ceiling is chunking, not ranking.
- **Dependencies added:** `rank_bm25` (small, pure Python; the index is built in memory at start-up, which is the scaling boundary to watch).
- **Commits:** `984bdb9`

### RAG-026: Filter on the fiscal period, not just the company
- **Type:** feat
- **Created:** 2026-09-04 | **Completed:** 2026-09-04
- **Competency:** retrieval quality
- **Description:** RAG-009 concluded that inferred metadata filtering buys nothing. That was measured on the **ticker** only, and it generalised too far. Filtering on the exact period label when a question names a specific quarter lifts recall@5 from 45.5% to 48.5% and unblocks the 10-Q questions that every strategy had scored at zero, because eight near-identical income statements stop competing. A bare fiscal year must not be filtered: a filing quotes prior years for comparison. Also fixes a Chroma adapter bug found while measuring, where a filter with more than one condition raised instead of filtering.
- **Done when:** the period filter is on by default, `docs/tradeoffs/retrieval-strategies.md` and ADR-008 no longer claim filtering buys nothing, and the multi-condition filter has a test.
- **Verified:** default `hybrid` reaches recall@5 48.5% and recall@10 51.5%, against 45.5% and 45.5% unfiltered; quarterly questions move from 0.0% to 14.3%. ADR-008 amended, tradeoff page corrected.
- **Commits:** `f8fe2dd`
