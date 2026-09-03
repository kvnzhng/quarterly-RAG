# ADR-005: Model provider is configurable; local stays the default

**Date:** 2026-09-03
**Status:** accepted (amends ADR-003)
**Ticket:** RAG-018

## Context

ADR-003 assumed Ollama running on the developer's laptop. That is too narrow: the author runs models on a separate machine on the local network, and readers of a public repo may prefer a hosted API with a token or a subscription-based agent runtime. Hard-coding Ollama would make the project less useful to both, and would hide a real production question: what changes when the model is a network dependency with a cost?

## Decision

- Chat and embedding models are reached through `LLM` and `Embedder` protocols; the provider is chosen in settings, never in code.
- Two chat providers are planned: `openai_compatible` (Ollama, vLLM, LM Studio, llama.cpp server, OpenRouter, OpenAI, ...) and `anthropic`. One client per wire protocol, not per vendor. A subscription-based agent runtime that is not an HTTP API would be a third implementation of the same protocol; none is planned yet.
- Embeddings have their own provider and endpoint settings, because Anthropic serves no embeddings and pairing a hosted chat model with local embeddings is the sensible default (the index is rebuilt often, embedding cost adds up).
- The default remains a local OpenAI-compatible endpoint (Ollama on `localhost`) so the repo runs free out of the box. CI never calls a model.
- Every documented number states the provider and model that produced it (`Settings.model_label()`).
- Amendment to ADR-003: "a hosted model may be added only as a compared alternative" becomes "a hosted model is a supported provider; local is the default".

## Consequences

- The OpenAI-compatible surface (chat completions, embeddings) is the lowest common denominator. Provider-specific features such as Anthropic citations or prompt caching live inside that provider's implementation and are optional.
- Comparing a local 7B-8B model with a hosted frontier model on the same eval set becomes a first-class tradeoff (`docs/tradeoffs/llm-serving.md`) rather than a footnote.
- `rag doctor` (RAG-002) checks the configured endpoint, not Ollama specifically. `make models` exists only for people running Ollama themselves.
- Server addresses and tokens live in `.env`, never in the repo.
