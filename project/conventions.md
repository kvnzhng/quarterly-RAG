# Coding Conventions -- rag_project

## General

- Prefer clarity over cleverness
- Functions should do one thing
- Name variables descriptively
- Keep files under 300 lines; split if larger

## Python 3.12 Specifics

- **Package manager:** `uv` only. Add dependencies with `uv add <pkg>` (or `uv add --group dev <pkg>`); never edit `uv.lock` by hand. Commit `uv.lock`.
- **Layout:** `src/rag_project/<layer>/`. Layers follow the pipeline: `ingestion` -> `chunking` -> `indexing` -> `retrieval` -> `generation` -> `evaluation`, plus `observability`. A layer may import from layers to its left, never to its right.
- **Interfaces first:** anything that will be compared (vector store, embedder, chunker, reranker, judge) is a `typing.Protocol` in the layer's `base.py`, with one module per implementation (`chroma.py`, `faiss.py`).
- **Config:** all settings come from `rag_project.config.Settings` (pydantic-settings, `.env` file). No `os.environ` reads elsewhere.
- **Types:** full type hints on public functions. `ruff` enforces style; run `make lint` before committing.
- **Data models:** `pydantic.BaseModel` for anything that crosses a layer boundary (`Chunk`, `RetrievedChunk`, `Answer`, `Refusal`). Provenance fields are required, not optional.
- **Paths:** `pathlib.Path` everywhere; all data lives under `settings.data_dir`.
- **LLM calls:** go through `rag_project.generation.llm` so the provider (Ollama today) can be swapped and traced in one place.
- **Tests:** `pytest`, files mirror the package (`tests/retrieval/test_hybrid.py`). Unit tests must not need Ollama or network; mark integration tests with `@pytest.mark.integration` (skipped in CI by default).
- **Notebooks:** exploration only, under `notebooks/`; anything worth keeping moves into `src/` with a test.
- **Docs:** a tradeoff is not decided until `docs/tradeoffs/<topic>.md` has measured numbers from this corpus and an ADR records the choice.

## Git

- **Branch naming:** `type/RAG-NNN-short-description`
  - Types: feat, fix, chore, refactor, test, docs
- **Commit messages:** `type(scope): description (RAG-NNN)`
  - One logical change per commit
  - Always include ticket ID
- **Branching:** Create a branch per ticket for non-trivial work

## Code Quality

- Re-read code before committing
- Check for: edge cases, error handling, naming clarity
- Keep dependencies minimal -- add only what's needed, and note why in the ticket
