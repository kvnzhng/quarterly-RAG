# Coding Conventions -- quarterly-RAG

## General

- Prefer clarity over cleverness
- Functions should do one thing
- Name variables descriptively
- Keep files under 300 lines; split if larger

## Python 3.12 Specifics

- **Package manager:** `uv` only. Add dependencies with `uv add <pkg>` (or `uv add --group dev <pkg>`); never edit `uv.lock` by hand. Commit `uv.lock`.
- **Layout:** `src/quarterly_rag/<layer>/`. Layers follow the pipeline: `ingestion` -> `chunking` -> `indexing` -> `retrieval` -> `generation` -> `evaluation`, plus `observability`. A layer may import from layers to its left, never to its right.
- **Interfaces first:** anything that will be compared (vector store, embedder, chunker, reranker, judge) is a `typing.Protocol` in the layer's `base.py`, with one module per implementation (`chroma.py`, `faiss.py`).
- **Config:** all settings come from `quarterly_rag.config.Settings` (pydantic-settings, `.env` file). No `os.environ` reads elsewhere.
- **Types:** full type hints on public functions. `ruff` enforces style; run `make lint` before committing.
- **Data models:** `pydantic.BaseModel` for anything that crosses a layer boundary (`Chunk`, `RetrievedChunk`, `Answer`, `Refusal`). Provenance fields are required, not optional.
- **Paths:** `pathlib.Path` everywhere; all data lives under `settings.data_dir`.
- **LLM calls:** go through `quarterly_rag.generation.llm` so the configured provider (ADR-005) can be swapped and traced in one place.
- **Tests:** `pytest`, files mirror the package (`tests/retrieval/test_hybrid.py`). Unit tests must not need Ollama or network; mark integration tests with `@pytest.mark.integration` (skipped in CI by default).
- **Notebooks:** exploration only, under `notebooks/`; anything worth keeping moves into `src/` with a test.
- **Docs:** a tradeoff is not decided until `docs/tradeoffs/<topic>.md` has measured numbers from this corpus and an ADR records the choice.
- **Reported numbers:** every number in `docs/` carries a run record: git commit, corpus manifest hash, parser version, chunker name and config, embedding provider and model, vector store, retrieval parameters, prompt version, provider and model, timestamp. Reports under `reports/` embed it; docs quote it. A number without one is a draft.
- **Eval labels:** gold evidence is a span (accession, section, char offsets), never a chunk id. Relevance of a chunk is derived by overlap, so one label set scores every chunker.

## Git

- **Branch naming:** `type/RAG-NNN-short-description`
  - Types: feat, fix, chore, refactor, test, docs
- **Commit messages:** `type(scope): description (RAG-NNN)`
  - One logical change per commit
  - Always include ticket ID; `scripts/check-commit-msg.sh` enforces it as a `commit-msg` hook (installed by `make setup`) and in CI
- **Branching:** Create a branch per ticket for non-trivial work

## Code Quality

- Re-read code before committing
- Check for: edge cases, error handling, naming clarity
- Keep dependencies minimal -- add only what's needed, and note why in the ticket
