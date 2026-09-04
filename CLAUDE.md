# quarterly-RAG

A local, open-source Retrieval-Augmented Generation system that answers questions about SEC quarterly and annual filings (10-Q, 10-K) of NASDAQ/NYSE companies (starting with Apple and Nvidia). Built to learn and demonstrate five production RAG competencies: grounding, chunking, retrieval quality, hallucination control, and when to refuse to answer.

**Stack:** Python 3.12, uv, LangChain (selectively), any OpenAI-compatible model server (Ollama by default, local or on the network) or the Anthropic API, embeddings configured separately, ChromaDB and FAISS (compared), rank_bm25, RAGAS, Langfuse (self-hosted), FastAPI, Streamlit, pytest, ruff.

## File Structure

<!-- Auto-updated by /project-update. Do not edit manually. -->
```
./.claude/active-ticket
./.claude/hooks/enforce-ticket.sh
./.claude/settings.json
./.env.example
./.github/workflows/ci.yml
./.gitignore
./.pre-commit-config.yaml
./.python-version
./AGENTS.md
./CLAUDE.md
./data/eval/.gitkeep
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
./docs/architecture.md
./docs/learning/chunking.md
./docs/learning/grounding.md
./docs/learning/hallucination-control.md
./docs/learning/README.md
./docs/learning/refusal.md
./docs/learning/retrieval-quality.md
./docs/notes.md
./docs/tradeoffs/_template.md
./docs/tradeoffs/chunking.md
./docs/tradeoffs/embeddings.md
./docs/tradeoffs/evaluation.md
./docs/tradeoffs/llm-serving.md
./docs/tradeoffs/observability.md
./docs/tradeoffs/orchestration.md
./docs/tradeoffs/parsing.md
./docs/tradeoffs/README.md
./docs/tradeoffs/retrieval-strategies.md
./docs/tradeoffs/vector-stores.md
./infra/README.md
./LICENSE
./Makefile
./notebooks/.gitkeep
./project/conventions.md
./project/tickets.md
./pyproject.toml
./README.md
./scripts/check-commit-msg.sh
./scripts/draft_eval_questions.py
./src/quarterly_rag/__init__.py
./src/quarterly_rag/chunking/__init__.py
./src/quarterly_rag/chunking/base.py
./src/quarterly_rag/chunking/build.py
./src/quarterly_rag/chunking/fixed.py
./src/quarterly_rag/cli.py
./src/quarterly_rag/config.py
./src/quarterly_rag/doctor.py
./src/quarterly_rag/errors.py
./src/quarterly_rag/evaluation/__init__.py
./src/quarterly_rag/evaluation/metrics.py
./src/quarterly_rag/evaluation/questions.py
./src/quarterly_rag/evaluation/relevance.py
./src/quarterly_rag/evaluation/retrieval_eval.py
./src/quarterly_rag/generation/__init__.py
./src/quarterly_rag/generation/anthropic_api.py
./src/quarterly_rag/generation/base.py
./src/quarterly_rag/generation/llm.py
./src/quarterly_rag/generation/openai_compatible.py
./src/quarterly_rag/indexing/__init__.py
./src/quarterly_rag/indexing/base.py
./src/quarterly_rag/indexing/build.py
./src/quarterly_rag/indexing/chroma.py
./src/quarterly_rag/indexing/embed_text.py
./src/quarterly_rag/indexing/embedder.py
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
./src/quarterly_rag/retrieval/__init__.py
./src/quarterly_rag/retrieval/base.py
./src/quarterly_rag/retrieval/dense.py
./tests/conftest.py
./tests/generation/test_anthropic_api.py
./tests/generation/test_llm_factory.py
./tests/generation/test_openai_compatible_llm.py
./tests/indexing/test_embedder_factory.py
./tests/indexing/test_openai_compatible_embedder.py
./tests/ingestion/edgar_fixtures.py
./tests/ingestion/fixtures/tenk.htm
./tests/ingestion/fixtures/tenq.htm
./tests/ingestion/test_download.py
./tests/ingestion/test_edgar_client.py
./tests/ingestion/test_fiscal.py
./tests/ingestion/test_parse.py
./tests/ingestion/test_records.py
./tests/integration/test_live_doctor.py
./tests/chunking/test_build.py
./tests/chunking/test_fixed.py
./tests/evaluation/test_metrics.py
./tests/evaluation/test_questions.py
./tests/evaluation/test_relevance.py
./tests/evaluation/test_retrieval_eval.py
./tests/integration/test_live_chunks.py
./tests/integration/test_live_edgar.py
./tests/integration/test_live_eval_set.py
./tests/integration/test_live_parse.py
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
- **Retrieval eval:** `uv run rag eval retrieval -k 5 --context` (recall@k, MRR, nDCG, run record)
- **Eval set:** `uv run rag eval check` (every gold evidence span still resolves)
- **Corpus:** `uv run rag ingest download --ticker AAPL --ticker NVDA` then `rag ingest parse --ticker AAPL --ticker NVDA` (EDGAR into `data/raw/`, sections into `data/processed/`, both idempotent)
- **Models:** `make models` (pulls Ollama models, RAG-002)
- **Eval:** `make eval` (RAG-008+)

## Coding Conventions

See [project/conventions.md](project/conventions.md) for detailed conventions.

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
