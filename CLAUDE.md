# rag_project

A local, open-source Retrieval-Augmented Generation system that answers questions about SEC quarterly and annual filings (10-Q, 10-K) of NASDAQ/NYSE companies (starting with Apple and Nvidia). Built to learn and demonstrate five production RAG competencies: grounding, chunking, retrieval quality, hallucination control, and when to refuse to answer.

**Stack:** Python 3.12, uv, LangChain (selectively), Ollama (local LLM + embeddings), ChromaDB and FAISS (compared), rank_bm25, RAGAS, Langfuse (self-hosted), FastAPI, Streamlit, pytest, ruff.

## File Structure

<!-- Auto-updated by /project-update. Do not edit manually. -->
```
./.claude/hooks/enforce-ticket.sh
./.claude/settings.json
./.env.example
./.github/workflows/ci.yml
./.gitignore
./.pre-commit-config.yaml
./.pytest_cache/.gitignore
./.pytest_cache/CACHEDIR.TAG
./.pytest_cache/README.md
./.pytest_cache/v/cache/nodeids
./.python-version
./.ruff_cache/.gitignore
./.ruff_cache/0.16.5/13289032586696463657
./.ruff_cache/0.16.5/13746165939635316503
./.ruff_cache/0.16.5/1474625928649818569
./.ruff_cache/0.16.5/15538562419545072851
./.ruff_cache/0.16.5/17846413621813012552
./.ruff_cache/0.16.5/237377877999443637
./.ruff_cache/0.16.5/2809551034199143135
./.ruff_cache/0.16.5/4418208441682260447
./.ruff_cache/0.16.5/8445398076917616454
./.ruff_cache/CACHEDIR.TAG
./CLAUDE.md
./data/eval/.gitkeep
./data/indexes/.gitkeep
./data/processed/.gitkeep
./data/raw/.gitkeep
./docs/adr/001-initial-setup.md
./docs/adr/002-python-uv-src-layout.md
./docs/adr/003-local-first-open-source-stack.md
./docs/adr/004-corpus-sec-filings.md
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
./src/rag_project/__init__.py
./src/rag_project/__pycache__/__init__.cpython-312.pyc
./src/rag_project/__pycache__/cli.cpython-312.pyc
./src/rag_project/__pycache__/config.cpython-312.pyc
./src/rag_project/chunking/__init__.py
./src/rag_project/cli.py
./src/rag_project/config.py
./src/rag_project/evaluation/__init__.py
./src/rag_project/generation/__init__.py
./src/rag_project/indexing/__init__.py
./src/rag_project/ingestion/__init__.py
./src/rag_project/observability/__init__.py
./src/rag_project/retrieval/__init__.py
./tests/__pycache__/conftest.cpython-312-pytest-9.1.1.pyc
./tests/__pycache__/test_config.cpython-312-pytest-9.1.1.pyc
./tests/conftest.py
./tests/test_config.py
./uv.lock
```

## Build / Test / Run

- **Setup:** `make setup` (uv sync + pre-commit install)
- **Lint / format:** `make lint` / `make fmt`
- **Test:** `make test`
- **Run CLI:** `uv run rag --help`
- **Models:** `make models` (pulls Ollama models, RAG-002)
- **Eval:** `make eval` (RAG-008+)

## Coding Conventions

See [project/conventions.md](project/conventions.md) for detailed conventions.

## Architecture Decisions

See [docs/adr/](docs/adr/) for architecture decision records. Every tooling tradeoff (vector store, embeddings, chunking, evaluation framework, observability) ends in an ADR plus a filled page under [docs/tradeoffs/](docs/tradeoffs/).

## Project Principles

- **Local and free first.** Everything must run on a laptop with no paid API. A hosted model may be added later only as an explicitly compared alternative, never as a requirement.
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

### Branch naming

```
type/RAG-NNN-short-description
```

Examples: `feat/RAG-003-edgar-downloader`, `feat/RAG-011-refusal-policy`
