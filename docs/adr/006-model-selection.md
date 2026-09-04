# ADR-006: Default models are provisional: `llama3.1:8b` for chat, `nomic-embed-text` for embeddings

**Date:** 2026-09-04
**Status:** proposed (the defaults stand until RAG-008 and RAG-012 numbers say otherwise)
**Ticket:** RAG-002

## Context

RAG-002 has to pick a chat model and an embedding model before any number exists, and ADR-005 says the provider is the user's choice with local as the default. The defaults therefore have to run free on a laptop, be pullable from Ollama by tag, and be good enough that early eval numbers say something about the pipeline rather than about a weak model. "Measure, don't assume" applies: this ADR records a starting point and the plan to test it, not a verdict.

## Decision

- **Chat default: `llama3.1:8b`.** Llama 3.1 community license, about 4.9 GB at the default Ollama quantisation, 128k context, dependable instruction following and JSON output, and ubiquitous enough that numbers are comparable with other people's RAG write-ups.
- **Embedding default: `nomic-embed-text`.** Apache-2.0, 768 dimensions, 8192-token context, about 274 MB. Strong for its size on MTEB retrieval, and the long context matters once parent-child chunking (RAG-020) embeds section-sized parents.
- **Both are defaults only.** `.env` overrides them; every index and eval number records the label (`provider/model`) that produced it.
- **Anthropic is reached through the official `anthropic` SDK**, added as a regular dependency. The SDK owns retries, typed errors and API drift (adaptive thinking on by default, sampling parameters removed on current models). `claude-opus-5` is the hosted comparison point in `docs/tradeoffs/llm-serving.md`.
- **`temperature` is part of the `LLM` protocol but not a guarantee.** The Anthropic provider ignores it because current Claude models reject sampling parameters. Generation code must not rely on temperature for determinism; RAG-010 relies on prompting plus verification instead.
- **Server-side refusal fallbacks are not enabled** in the Anthropic client. `LLM_MODEL` is user-configured and the parameter is model-specific; a provider-side `stop_reason=refusal` is surfaced on `ChatResponse` instead so RAG-011 can treat it explicitly.
- **Candidates to measure** are listed in `docs/tradeoffs/llm-serving.md` (chat, measured at RAG-012 on answer correctness, faithfulness, and refusal calibration) and `docs/tradeoffs/embeddings.md` (embeddings, measured at RAG-006 and RAG-008 on recall@k).

## Consequences

- An 8B model bounds answer quality, so refusal and verification carry more weight (ADR-003).
- Changing a default changes every stored baseline. The run record (RAG-008) names the model, so old numbers stay interpretable next to new ones.
- Whoever configures a hosted model pays for eval runs. Eval sets are sized in the tens of questions so a full run costs cents.
- `rag doctor` checks whatever is configured, not Ollama specifically; `make models` exists only for people running Ollama themselves.
