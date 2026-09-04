# LLM serving: which local model, and local vs hosted

**Status:** draft (candidates and criteria set at RAG-002; numbers arrive with RAG-010 and RAG-012)
**Ticket:** RAG-002, RAG-012

## Question

Which chat model answers grounded questions about filings well enough, at zero cost, on the hardware at hand, and how much does a hosted frontier model buy on the same eval set? The answer sets the default in `.env.example` and tells a reader when paying for a hosted model is worth it.

## Candidates

The serving machine is an Ollama server on the local network that already holds a ladder from 8B to 32B, so the local comparison is not limited to laptop-sized models. Sizes are as reported by `ollama list` there (Q4_K_M unless noted). Hosted candidates go through the same `LLM` protocol (ADR-005). "Uncensored" community variants on the server are excluded.

| Candidate | One-line description | License | Runs locally? |
|---|---|---|---|
| `llama3.1:8b` (default) | Meta 8B instruct, 128k context, the common baseline | Llama 3.1 community | yes, 4.9 GB |
| `gpt-oss:20b` | OpenAI open-weight 20.9B, MXFP4 | Apache-2.0 | yes, 13.8 GB |
| `mistral-small:latest` | Mistral Small, 23.6B | Apache-2.0 | yes, 14.3 GB |
| `gemma4:26b` | Google 25.8B | Gemma terms | yes, 18.0 GB |
| `qwen3.6:27b` | Qwen 27.8B | see model card | yes, 17.4 GB |
| `qwen3.8-27b-64k:latest` (also a 256k-context tag) | Qwen 27.3B, extended context | see model card | yes, 17.7 GB |
| `deepseek-r1:32b` | DeepSeek reasoning model, 32.8B, slow by design | MIT | yes, 19.9 GB |
| `claude-opus-5` | Anthropic frontier model via `LLM_PROVIDER=anthropic` | hosted, paid | no |
| `claude-sonnet-5` | Anthropic mid-tier via `LLM_PROVIDER=anthropic` | hosted, paid | no |

Smaller alternatives that fit a laptop (`qwen2.5:7b`, `mistral:7b`, `gemma3:12b`, `phi4:14b`) are not on the server; pulling one is `make models` with `LLM_MODEL` set.

## Criteria

| Criterion | How measured | Weight |
|---|---|---|
| Answer correctness | RAG-012 answer correctness on the RAG-019 `lookup` questions, same retrieved context for every model | high |
| Faithfulness | RAG-012 judge: share of cited sentences entailed by their chunk | high |
| Refusal calibration | RAG-011 abstention precision/recall on the `unanswerable` set at the chosen threshold | high |
| Citation format compliance | share of answers whose `[c12]` citations parse and resolve (RAG-010 verifier) | medium |
| Throughput | output tokens/s from the eval run, on the serving machine | medium |
| Context window | must hold the grounded prompt with parent chunks (RAG-020); 32k or more preferred | medium |
| Cost per eval run | tokens x price for hosted models; zero for local | low, but reported |

## Results

_Not yet measured. `rag doctor` at RAG-002 only proves the configured endpoint answers; the first comparable numbers come from RAG-010 (citation rates) and RAG-012 (correctness, faithfulness), each with a run record._

### Citation discipline, measured at RAG-010

23 `lookup` questions with the evidence handed to the model, prompt v1, so this measures the generator alone.

| Model | Size | Citations resolve | Every sentence cited | Figures verified | Fully grounded | States the gold figure |
|---|---|---|---|---|---|---|
| `llama3.1:8b` | 4.9 GB | 50% | 50% | 86% | 41% | 95% |
| `gpt-oss:20b` | 13.8 GB | 100% | 91% | 100% | 91% | 77% |
| `qwen3.6:27b` | 17.4 GB | 100% | 100% | 100% | 100% | 64% |

The first real tradeoff in this project, and not the one expected. **The 8B model is the best of the three at finding the right figure and by far the worst at saying where it found it**, inventing passage labels it was never given in half its answers. Neither larger model does that once.

A caveat found while measuring: `qwen3.6:27b` scored 43% on the gold figure until the answer budget rose from 400 to 1024 tokens. A thinking-mode model spends tokens before it writes, and a truncated answer scores as ungrounded, which blames the model for the budget. `ANSWER_MAX_TOKENS` is now a setting.

Run record: commit `0ab6a44`, prompt v1, gold passages, k=5, network Ollama server.

### Tried at RAG-002

`rag doctor` on 2026-09-04, Ollama 0.32.13 on the network server, one-word chat reply and one short embedding. Cold means the model had to be loaded while a 27B model was resident; warm is the immediate second run.

| Endpoint | Model | Chat latency (cold / warm) | Embedding dims | Notes |
|---|---|---|---|---|
| network Ollama server | `llama3.1:8b` | 3804 ms / 242 ms | | 11 models listed; `GET /v1/models` works |
| network Ollama server | `nomic-embed-text` | 4351 ms / 38 ms (embedding call) | 768 | pulled onto the server between the first and second doctor run |
| Ollama on this laptop | | not tried | | Ollama is not installed locally; `make models` no longer needs the CLI |

## Decision

`llama3.1:8b` stays the code default because it is the one that fits a laptop (ADR-003), but it is now known to be the weakest link in grounding: it fails citation discipline in half its answers. **On a machine with room, set `LLM_MODEL=gpt-oss:20b` in `.env`.** The final choice waits on RAG-012, which adds correctness and faithfulness judging; `qwen3.6:27b` is the one to watch there, since it grounds every sentence and currently trails on stating the labelled figure.

## Interview one-liner

The pipeline talks to models through two small interfaces, so swapping a local 8B model for a 27B one on the network or a hosted frontier model is a `.env` change, and the eval set says exactly what that swap buys.
