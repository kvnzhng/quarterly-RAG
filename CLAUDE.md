# rag_project

A local, open-source Retrieval-Augmented Generation system that answers questions about SEC quarterly and annual filings (10-Q, 10-K) of NASDAQ/NYSE companies (starting with Apple and Nvidia). Built to learn and demonstrate five production RAG competencies: grounding, chunking, retrieval quality, hallucination control, and when to refuse to answer.

**Stack:** Python 3.12, uv, LangChain (selectively), Ollama (local LLM + embeddings), ChromaDB and FAISS (compared), rank_bm25, RAGAS, Langfuse (self-hosted), FastAPI, Streamlit, pytest, ruff.

## File Structure

<!-- Auto-updated by /project-update. Do not edit manually. -->
```
(will be updated after init)
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
