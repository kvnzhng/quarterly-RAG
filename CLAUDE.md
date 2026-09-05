# quarterly-RAG

A local, open-source Retrieval-Augmented Generation system that answers questions about SEC quarterly and annual filings (10-Q, 10-K) of NASDAQ/NYSE companies (starting with Apple and Nvidia). Built to learn and demonstrate five production RAG competencies: grounding, chunking, retrieval quality, hallucination control, and when to refuse to answer.

**Stack:** Python 3.12, uv, plain Python orchestration (no LangChain), any OpenAI-compatible model server (Ollama by default, local or on the network) or the Anthropic API, embeddings configured separately, ChromaDB (default) and FAISS (measured, kept), rank_bm25, a custom LLM judge (RAGAS was measured and rejected), pytest, ruff, Langfuse (self-hosted, optional), FastAPI, Streamlit, and marimo for the course notebook.

## Current state (2026-09-05)

All three phases are built. The pipeline answers questions from the filings or refuses with a reason, and every layer is measured against a 63-question human-verified eval set. Eleven ADRs record the decisions, the README carries the results writeup (RAG-015), and a Notion course with a marimo notebook, `notebooks/course.py`, teaches it (RAG-034).

| Layer | Decision | Measured |
|---|---|---|
| Corpus | 16 filings, Apple and Nvidia, 8 quarters each (ADR-004) | idempotent download, byte-identical manifests |
| Parser | custom block-boundary parser (ADR-007) | 16/16 filings, 0 missing critical items, 0 false headings |
| Chunking | section-aware (ADR-009) | recall@1 39.4% vs 21.2% fixed; MRR 0.449 vs 0.314 |
| Embeddings | nomic-embed-text with task prefixes, context header prepended (ADR-006) | prefixes and header each roughly double recall |
| Store | ChromaDB (ADR-010); FAISS kept for scale | identical retrieval quality; store is 3% of a retrieval |
| Retrieval | hybrid dense+BM25, RRF pool 50, ticker and quarter filters (ADR-008) | recall@5 48.5% vs 36.4% dense; rerank measured and off |
| Generation | cited answers, deterministic figure check, refusal gate | 100% citations resolve, 87.5% fully grounded end to end |
| Model | llama3.1:8b default for laptops; qwen3.8-27b recommended (ADR-006) | 8B invents citations in half its answers |
| Judge | custom, cross-model, calibrated against the figure check | 86% agreement, 25% miss rate on unverified figures |
| Calculations | derived numbers written as `CALC:` lines and recomputed from cited operands; opt-in via `ANSWER_PROMPT_VERSION=2` (RAG-021) | qwen answers 10/10 derived questions where the default prompt refused 4; 8B model's arithmetic fails 4 of 10; costs 2 of the 33 answerable questions on the gate with `gpt-oss:20b` |
| Gate | `make eval` against `data/eval/baseline.json`, 5-point tolerance | nine metrics, committed; covers `lookup` only; deterministic within a day, the three model-dependent metrics drift by about one question across days (RAG-015) |
| Tracing | self-hosted Langfuse 4.30.0, off unless configured (ADR-011) | 150 ms a question; generation 8,423 ms against verification 4 ms |
| Interface | FastAPI `POST /ask` and a Streamlit page over it (RAG-014) | refusing is a 200; the page highlights the operands the verifier matched |

The binding constraint moved twice: retrieval was the ceiling until hybrid fusion (RAG-009), then chunking was (RAG-020). It is now roughly a quarter of questions whose evidence neither ranking nor chunking reaches (recall@20 72.7%).

**Next:** RAG-032, retrieval is unstable to phrasing and only for Nvidia; the labels come first. See `project/handoff.md` to resume.

## File Structure

<!-- Auto-updated by /project-update. Do not edit manually. -->
```
./.claude/hooks/enforce-ticket.sh
./.claude/settings.json
./.env.example
./.github/workflows/ci.yml
./.gitignore
./.pre-commit-config.yaml
./.python-version
./AGENTS.md
./CLAUDE.md
./LICENSE
./Makefile
./README.md
./data/eval/.gitkeep
./data/eval/baseline.json
./data/eval/questions.jsonl
./data/indexes/.gitkeep
./data/processed/.gitkeep
./data/raw/.gitkeep
./docs/adr/001-initial-setup.md
./docs/adr/002-python-uv-src-layout.md
./docs/adr/003-local-first-open-source-stack.md
./docs/adr/004-corpus-sec-filings.md
./docs/adr/005-model-provider-configurable.md
./docs/adr/006-model-selection.md
./docs/adr/007-custom-filing-parser.md
./docs/adr/008-hybrid-retrieval-default.md
./docs/adr/009-section-aware-chunking.md
./docs/adr/010-chromadb-default-store.md
./docs/architecture.md
./docs/learning/README.md
./docs/learning/chunking.md
./docs/learning/grounding.md
./docs/learning/hallucination-control.md
./docs/learning/refusal.md
./docs/learning/retrieval-quality.md
./docs/notes.md
./docs/tradeoffs/README.md
./docs/tradeoffs/_template.md
./docs/tradeoffs/chunking.md
./docs/tradeoffs/embeddings.md
./docs/tradeoffs/evaluation.md
./docs/tradeoffs/llm-serving.md
./docs/tradeoffs/observability.md
./docs/tradeoffs/orchestration.md
./docs/tradeoffs/parsing.md
./docs/tradeoffs/retrieval-strategies.md
./docs/tradeoffs/vector-stores.md
./infra/README.md
./project/conventions.md
./project/handoff.md
./project/tickets.md
./pyproject.toml
./scripts/check-commit-msg.sh
./scripts/draft_eval_questions.py
./scripts/edit_docs.py
./src/quarterly_rag/__init__.py
./src/quarterly_rag/chunking/__init__.py
./src/quarterly_rag/chunking/base.py
./src/quarterly_rag/chunking/build.py
./src/quarterly_rag/chunking/fixed.py
./src/quarterly_rag/chunking/recursive.py
./src/quarterly_rag/chunking/structural.py
./src/quarterly_rag/cli.py
./src/quarterly_rag/config.py
./src/quarterly_rag/doctor.py
./src/quarterly_rag/errors.py
./src/quarterly_rag/evaluation/__init__.py
./src/quarterly_rag/evaluation/baseline.py
./src/quarterly_rag/evaluation/calibration.py
./src/quarterly_rag/evaluation/generation_eval.py
./src/quarterly_rag/evaluation/judge.py
./src/quarterly_rag/evaluation/metrics.py
./src/quarterly_rag/evaluation/questions.py
./src/quarterly_rag/evaluation/refusal_eval.py
./src/quarterly_rag/evaluation/relevance.py
./src/quarterly_rag/evaluation/retrieval_eval.py
./src/quarterly_rag/generation/__init__.py
./src/quarterly_rag/generation/answer.py
./src/quarterly_rag/generation/anthropic_api.py
./src/quarterly_rag/generation/base.py
./src/quarterly_rag/generation/llm.py
./src/quarterly_rag/generation/numbers.py
./src/quarterly_rag/generation/openai_compatible.py
./src/quarterly_rag/generation/prompts/grounded_answer_v1.txt
./src/quarterly_rag/generation/refusal.py
./src/quarterly_rag/indexing/__init__.py
./src/quarterly_rag/indexing/base.py
./src/quarterly_rag/indexing/build.py
./src/quarterly_rag/indexing/chroma.py
./src/quarterly_rag/indexing/embed_text.py
./src/quarterly_rag/indexing/embedder.py
./src/quarterly_rag/indexing/faiss_store.py
./src/quarterly_rag/indexing/openai_compatible.py
./src/quarterly_rag/ingestion/__init__.py
./src/quarterly_rag/ingestion/download.py
./src/quarterly_rag/ingestion/edgar.py
./src/quarterly_rag/ingestion/fiscal.py
./src/quarterly_rag/ingestion/manifest.py
./src/quarterly_rag/ingestion/parse.py
./src/quarterly_rag/ingestion/records.py
./src/quarterly_rag/observability/__init__.py
./src/quarterly_rag/openai_compatible.py
./src/quarterly_rag/pipeline.py
./src/quarterly_rag/retrieval/__init__.py
./src/quarterly_rag/retrieval/base.py
./src/quarterly_rag/retrieval/bm25.py
./src/quarterly_rag/retrieval/build.py
./src/quarterly_rag/retrieval/dense.py
./src/quarterly_rag/retrieval/filtered.py
./src/quarterly_rag/retrieval/hybrid.py
./src/quarterly_rag/retrieval/query.py
./src/quarterly_rag/retrieval/rerank.py
./tests/chunking/__init__.py
./tests/chunking/test_build.py
./tests/chunking/test_fixed.py
./tests/chunking/test_structural.py
./tests/conftest.py
./tests/evaluation/__init__.py
./tests/evaluation/test_baseline.py
./tests/evaluation/test_judge.py
./tests/evaluation/test_metrics.py
./tests/evaluation/test_questions.py
./tests/evaluation/test_refusal_eval.py
./tests/evaluation/test_relevance.py
./tests/evaluation/test_retrieval_eval.py
./tests/generation/test_answer.py
./tests/generation/test_anthropic_api.py
./tests/generation/test_llm_factory.py
./tests/generation/test_numbers.py
./tests/generation/test_openai_compatible_llm.py
./tests/generation/test_refusal.py
./tests/indexing/test_chroma_store.py
./tests/indexing/test_embed_text.py
./tests/indexing/test_embedder_factory.py
./tests/indexing/test_faiss_store.py
./tests/indexing/test_index_build.py
./tests/indexing/test_openai_compatible_embedder.py
./tests/ingestion/__init__.py
./tests/ingestion/edgar_fixtures.py
./tests/ingestion/fixtures/tenk.htm
./tests/ingestion/fixtures/tenq.htm
./tests/ingestion/test_download.py
./tests/ingestion/test_edgar_client.py
./tests/ingestion/test_fiscal.py
./tests/ingestion/test_parse.py
./tests/ingestion/test_records.py
./tests/integration/test_live_chunks.py
./tests/integration/test_live_doctor.py
./tests/integration/test_live_edgar.py
./tests/integration/test_live_eval_set.py
./tests/integration/test_live_generation.py
./tests/integration/test_live_index.py
./tests/integration/test_live_parse.py
./tests/retrieval/__init__.py
./tests/retrieval/test_bm25.py
./tests/retrieval/test_dense.py
./tests/retrieval/test_hybrid.py
./tests/retrieval/test_query.py
./tests/test_config.py
./tests/test_doctor.py
./uv.lock
```

## Build / Test / Run

- **Setup:** `make setup` (uv sync + pre-commit install)
- **Lint / format:** `make lint` / `make fmt`
- **Test:** `make test`
- **Run CLI:** `uv run rag --help`
- **Doctor:** `make doctor` or `uv run rag doctor` (configured endpoints, models, data dirs)
- **Chunks:** `uv run rag chunk build --ticker AAPL --ticker NVDA` (sections into `data/chunks/<strategy>/`)
- **Index:** `uv run rag index build --ticker AAPL --ticker NVDA [--context]` then `rag index query "..."`
- **API and page:** `make api` (FastAPI `POST /ask` on 127.0.0.1:8000, schema at `/docs`) and `make ui` (Streamlit on 127.0.0.1:8501, talks to the API over HTTP and nothing else)
- **Course notebook:** `make course` (marimo editor over the real pipeline; the course itself is at https://flashy-fur-afc.notion.site/quarterly-RAG-a-course-on-production-RAG-3d21f11d4bc881a6b753c2c819817428); `uv run python notebooks/course.py` runs the ungated cells as a script
- **Ask:** `uv run rag ask "..."` (retrieve, answer, verify every sentence against its source). `ANSWER_PROMPT_VERSION=2` lets the model compute a derived number and shows the arithmetic it is checked against.
- **Generation eval:** `uv run rag eval generation --context gold|retrieved`
- **Refusal eval:** `uv run rag eval refusal` (abstention precision/recall, threshold sweep)
- **Retrieval eval:** `uv run rag eval retrieval -k 5 --context --retrieval hybrid` (recall@k, MRR, nDCG, run record)
- **Eval set:** `uv run rag eval check` (every gold evidence span still resolves)
- **Corpus:** `uv run rag ingest download --ticker AAPL --ticker NVDA` then `rag ingest parse --ticker AAPL --ticker NVDA` (EDGAR into `data/raw/`, sections into `data/processed/`, both idempotent)
- **Models:** `make models` (pulls the models named in `.env` onto the Ollama at `OLLAMA_HOST`, over its HTTP API)
- **Regression gate:** `make eval` (every metric against `data/eval/baseline.json`; calls a model, ~5 min) and `make eval-accept` (overwrite the baseline deliberately)

## Coding Conventions

See [project/conventions.md](project/conventions.md) for detailed conventions, including the process rules learned the hard way in this repo.

## Working in this repo: lessons

- **Measure before writing a number, and verify a claim before committing it.** Twice a commit message described documentation edits that had silently failed to apply. Use `scripts/edit_docs.py`, which applies edits one at a time and reports which did not land, and read its output before writing the commit message.
- **Check that every commit hash in `project/tickets.md` resolves** after any reset or amend. A ticket once pointed at a discarded commit.
- **A generalised conclusion is a bug.** "Filtering buys nothing" was true of the ticker filter and false of the period filter, and it shipped as a general claim. State exactly what was measured.
- **Fair comparisons need fair budgets.** A thinking-mode model scored 43% until `ANSWER_MAX_TOKENS` rose from 400 to 1024; a truncated answer scores as ungrounded.
- **The eval set exists before the index does**, and that ordering found every real bug so far: the missing embedding prefixes, the citation parser mistakes, the presence-check limit, the RAGAS failure.
- **The server address never enters the repo, a commit, or a recap.** Refer to "the network Ollama server".
- **The gate is deterministic within a day and not across days.** Two runs an hour apart agreed, two runs the next day agreed with each other and not with the baseline, by one refused question and two judged sentences, with identical code and model digests. Re-run the gate on the day a number is quoted, compare per question before calling a move a change, and read faithfulness against its 6.25-point granularity.

## Architecture Decisions

See [docs/adr/](docs/adr/) for architecture decision records. Every tooling tradeoff (vector store, embeddings, chunking, evaluation framework, observability) ends in an ADR plus a filled page under [docs/tradeoffs/](docs/tradeoffs/).

## Project Principles

- **Local and free by default; the provider is the user's choice.** Defaults run on a laptop with no paid API. Any OpenAI-compatible server (this machine or the network) or a hosted API with a token is configured in `.env` only (ADR-005). Every documented number names the provider and model that produced it. CI never calls a model. Server addresses and tokens never go in the repo.
- **Measure, don't assume.** A tradeoff doc without numbers from this corpus is a draft, not a decision.
- **Provenance everywhere.** Every chunk carries ticker, form, period, section, and offsets. Every answer sentence carries a citation or is flagged as unsupported.
- **Refusal is a feature.** The system must be able to say "not in the filings" with a reason.

## CRITICAL RULES

### Every edit MUST be linked to a ticket

Before making ANY code change:

1. Check `project/tickets.md` for an existing ticket, or create one
2. Move the ticket to "In Progress" in tickets.md
3. Write the ticket ID to `.claude/active-ticket`:
   ```bash
   echo "RAG-NNN" > .claude/active-ticket
   ```
4. Make your edits
5. Commit with the ticket ID in the message:
   ```
   type(scope): description (RAG-NNN)
   ```
6. Move ticket to "Done" in tickets.md, record the commit hash
7. Clear the active ticket:
   ```bash
   echo "" > .claude/active-ticket
   ```

**If you find yourself editing code without a ticket, STOP. Create the ticket first. This is non-negotiable.**

### Meta files are exempt from ticket enforcement

These files can be edited without an active ticket:
- `project/tickets.md`
- `CLAUDE.md`
- `project/conventions.md`
- `docs/notes.md`
- `docs/adr/*`
- `.claude/active-ticket`
- `.claude/settings.json`
- `.claude/settings.local.json`
- `.gitignore`

### Ticket workflow

| Step | Action |
|------|--------|
| Pick up work | Find or create ticket in `project/tickets.md` |
| Claim ticket | Move to "In Progress", write ID to `.claude/active-ticket` |
| Work | All edits are linked to the active ticket |
| Complete | Commit with ticket ID, move to "Done", record commit hash, clear `.claude/active-ticket` |

### Commit message format

```
type(scope): description (RAG-NNN)
```

Types: `feat`, `fix`, `chore`, `refactor`, `test`, `docs`

Examples:
```
feat(ingestion): download 10-Q filings from EDGAR (RAG-003)
fix(retrieval): handle empty BM25 index (RAG-009)
docs: fill vector store tradeoff table (RAG-007)
```

Enforced by `scripts/check-commit-msg.sh`: `make setup` installs it as a pre-commit `commit-msg` hook, and CI runs it over every push and pull request. `AGENTS.md` is a symlink to this file so Codex follows the same rules.

### Branch naming

```
type/RAG-NNN-short-description
```

Examples: `feat/RAG-003-edgar-downloader`, `feat/RAG-011-refusal-policy`
