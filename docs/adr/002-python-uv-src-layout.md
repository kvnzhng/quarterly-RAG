# ADR-002: Python 3.12, uv, and src layout

**Date:** 2026-09-03
**Status:** accepted
**Ticket:** RAG-001

## Context

The ML ecosystem (faiss-cpu, torch via sentence-transformers, chromadb) tends to lag the newest CPython by months. The machine has 3.10 through 3.14 installed, plus an Anaconda Python on PATH, which makes environment drift likely.

## Decision

- Pin **Python 3.12** in `.python-version`. Broadest wheel coverage for the planned dependencies at the time of writing; revisit when every dependency ships 3.13 wheels.
- **uv** for environment, lockfile, and scripts. A committed `uv.lock` makes the GitHub repo reproducible.
- **src layout** (`src/rag_project/`) so tests import the installed package, not the working directory.

## Consequences

- Never rely on the system or Anaconda Python; always `uv run`.
- Dependency additions are explicit (`uv add`) and reviewed in the ticket.
