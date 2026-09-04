# ADR-006: Default models are provisional: `llama3.1:8b` for chat, `nomic-embed-text` for embeddings

**Date:** 2026-09-04
**Status:** proposed, amended 2026-09-04 by the RAG-010 and RAG-025 measurements below
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

## Amendment, 2026-09-04 (RAG-010, RAG-025)

Four models were scored on the same 23 lookup questions with the same prompt, in two contexts: evidence handed to the model, and evidence retrieved by the pipeline.

| Model | Fully grounded (gold) | States the gold figure (gold) | Fully grounded (retrieved) | States the gold figure (retrieved) | Seconds per answer |
|---|---|---|---|---|---|
| `llama3.1:8b` | 41% | 95% | | | |
| `gpt-oss:20b` | 91% | 77% | 87% | 67% | 3.8 |
| `qwen3.6:27b` | 100% | 64% | | | |
| `qwen3.8-27b-64k` | 91% | 91% | 100% | 75% | 9.6 |

Two findings change how this ADR should be read.

**Citation discipline is a model capability.** `llama3.1:8b` produces a resolvable citation for only half its answers: it invents passage labels it was never given. Every model at 20B or above reaches 100% on the identical prompt. The 8B model is simultaneously the best of the four at finding the right figure.

**Grounding and correctness are separate axes.** Two of the larger models ground well and then state the labelled figure only 64% and 77% of the time. `qwen3.8-27b` is the only one measured that does both.

The code default is unchanged, because ADR-003 requires the defaults to run on a laptop and 4.9 GB is what that allows. But the default is now known to be the weakest link in grounding, and `docs/tradeoffs/llm-serving.md` says so with numbers. **Anyone with 18 GB to spare should set `LLM_MODEL=qwen3.8-27b-64k:latest`; choose `gpt-oss:20b` when latency matters more than correctness.**

A third finding worth carrying forward: `ANSWER_MAX_TOKENS` defaults to 1024 rather than 400, because a thinking-mode model spends tokens reasoning before it writes and a truncated answer scores as ungrounded. Measuring one of those at 400 tokens understated it by 20 points.

## Consequences

- An 8B model bounds answer quality, so refusal and verification carry more weight (ADR-003).
- Changing a default changes every stored baseline. The run record (RAG-008) names the model, so old numbers stay interpretable next to new ones.
- Whoever configures a hosted model pays for eval runs. Eval sets are sized in the tens of questions so a full run costs cents.
- `rag doctor` checks whatever is configured, not Ollama specifically; `make models` exists only for people running Ollama themselves.
