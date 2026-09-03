# ADR-003: Local-first, open-source model stack

**Date:** 2026-09-03
**Status:** accepted, amended by ADR-005 (model provider is configurable)
**Ticket:** RAG-001

## Context

The project must run without paid APIs and on a laptop. It is also meant to make tradeoffs visible, so components need to be swappable.

## Decision

- **LLM and embeddings served by Ollama** (OpenAI-compatible local server). Concrete models are chosen in RAG-002 and recorded in ADR-006.
- **Every replaceable component sits behind a Protocol**: `Embedder`, `VectorStore`, `Chunker`, `Reranker`, `Judge`, `LLM`. Comparisons are run through the same interface.
- **LangChain is used selectively** (document loaders, text splitters, integrations) rather than as the whole framework. The orchestration of retrieve -> verify -> answer/refuse stays in plain Python so the control flow is explicit and testable. Revisit in `docs/tradeoffs/orchestration.md`.
- **Two vector stores** (ChromaDB, FAISS) are implemented and benchmarked before a default is chosen (ADR-007, RAG-007).
- **Observability with self-hosted Langfuse** via docker compose (RAG-013). Alternatives compared in `docs/tradeoffs/observability.md`.
- A hosted model (for example Anthropic's `claude-opus-5`) may be added later strictly as an additional `LLM` implementation for a quality comparison, never as a requirement to run the project.

## Consequences

- Answer quality is bounded by 7B-8B local models; the eval numbers must be read with that in mind and the refusal policy matters more.
- First-run setup includes pulling several GB of model weights and running Docker for Langfuse.
- Every "X vs Y" question in the job description has a place where the answer is measured, not asserted.
