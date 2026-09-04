# LLM serving: which local model, and local vs hosted

**Status:** draft (candidates and criteria set at RAG-002; numbers arrive with RAG-010 and RAG-012)
**Ticket:** RAG-002, RAG-012

## Question

Which chat model answers grounded questions about filings well enough, at zero cost, on the hardware at hand, and how much does a hosted frontier model buy on the same eval set? The answer sets the default in `.env.example` and tells a reader when paying for a hosted model is worth it.

## Candidates

All local candidates are pulled by tag on Ollama; the hosted ones are reached through the same `LLM` protocol (ADR-005). Sizes are the default Ollama quantisation.

| Candidate | One-line description | License | Runs locally? |
|---|---|---|---|
| `llama3.1:8b` (default) | Meta 8B instruct, 128k context, the common baseline | Llama 3.1 community | yes, ~4.9 GB |
| `qwen2.5:7b` | Alibaba 7B instruct, strong on structured and numeric text | Apache-2.0 | yes, ~4.7 GB |
| `qwen3:8b` | Qwen3 8B with optional thinking mode | Apache-2.0 | yes, ~5.2 GB |
| `mistral:7b` | Mistral 7B instruct v0.3 | Apache-2.0 | yes, ~4.1 GB |
| `gemma3:12b` | Google 12B, 128k context | Gemma terms | yes, ~8.1 GB |
| `phi4:14b` | Microsoft 14B, strong reasoning for its size | MIT | yes, ~9.1 GB |
| `claude-opus-5` | Anthropic frontier model via `LLM_PROVIDER=anthropic` | hosted, paid | no |
| `claude-sonnet-5` | Anthropic mid-tier via `LLM_PROVIDER=anthropic` | hosted, paid | no |

Tags and sizes are checked with `ollama list` on the serving machine before a candidate is measured; the table is updated then.

## Criteria

| Criterion | How measured | Weight |
|---|---|---|
| Answer correctness | RAG-012 answer correctness on the RAG-019 `lookup` questions, same retrieved context for every model | high |
| Faithfulness | RAG-012 judge: share of cited sentences entailed by their chunk | high |
| Refusal calibration | RAG-011 abstention precision/recall on the `unanswerable` set at the chosen threshold | high |
| Citation format compliance | share of answers whose `[c12]` citations parse and resolve (RAG-010 verifier) | medium |
| Throughput | output tokens/s from `rag doctor` and the eval run, on the serving machine | medium |
| Context window | must hold the grounded prompt with parent chunks (RAG-020); 32k or more preferred | medium |
| Cost per eval run | tokens x price for hosted models; zero for local | low, but reported |

## Results

_Not yet measured. `rag doctor` at RAG-002 only proves the configured endpoint answers; the first comparable numbers come from RAG-010 (citation rates) and RAG-012 (correctness, faithfulness), each with a run record._

### Tried at RAG-002

_Filled after the live `rag doctor` runs: which server, which models it lists, cold and warm chat latency for the default model._

| Endpoint | Model | Chat latency (cold / warm) | Embedding dims | Notes |
|---|---|---|---|---|

## Decision

Pending. `llama3.1:8b` and `nomic-embed-text` stay the defaults (ADR-006) until the RAG-012 table shows a local candidate that beats them on correctness or faithfulness without losing refusal calibration.

## Interview one-liner

The pipeline talks to models through two small interfaces, so swapping a local 8B model for a hosted frontier model is a `.env` change, and the eval set says exactly what that swap buys.
