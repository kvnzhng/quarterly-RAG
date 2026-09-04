# quarterly-RAG

[![ci](https://github.com/kvnzhng/quarterly-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/kvnzhng/quarterly-RAG/actions/workflows/ci.yml)

> Local, open-source Retrieval-Augmented Generation over SEC quarterly and annual filings (10-Q, 10-K), built to learn and demonstrate what "shipping a production RAG system" actually requires: **grounding, chunking, retrieval quality, hallucination control, and knowing when to refuse to answer.**

The plan keeps everything on a laptop with no paid API: Ollama for the LLM and embeddings, ChromaDB and FAISS for vectors, Langfuse self-hosted for traces. The model provider is your choice: point it at a model server on your network or at a hosted API by editing `.env`.

**Current state:** scaffolding, tooling, and the roadmap. The pipeline itself starts with RAG-002; see [Status](#status).

## Why filings?

10-Q and 10-K reports are free, public, long, highly structured, and full of exact numbers. That makes them a good corpus for RAG: retrieval is non-trivial (the same sections repeat every quarter), grounding is checkable (a revenue figure is either in the filing or it is not), and refusal has a concrete meaning (the period or company is not in the corpus).

Starting companies: **Apple (AAPL)** and **Nvidia (NVDA)**. Adding a ticker is a config change.

## The five competencies and where they live

This table is the plan, not a feature list. Each row becomes true when its tickets close.

| Competency | What the project will do | Where to look |
|---|---|---|
| Grounding | Every chunk carries ticker, form, period, section, and offsets. Every answer sentence cites a chunk id that is verified to exist and to contain the quoted numbers. | `docs/learning/grounding.md`, RAG-004, RAG-010 |
| Chunking | One simple chunker first, then fixed, recursive, section-aware, and parent-child compared on retrieval metrics, not on intuition. | `docs/learning/chunking.md`, `docs/tradeoffs/chunking.md`, RAG-005, RAG-020 |
| Retrieval quality | Human-verified eval set labeled with evidence spans, so the labels survive a change of chunker. recall@k, MRR, nDCG with a run record on every number. Dense vs BM25 vs hybrid vs hybrid + reranker. Chroma vs FAISS benchmark. | `docs/learning/retrieval-quality.md`, `docs/tradeoffs/vector-stores.md`, RAG-019, RAG-006 to RAG-009 |
| Hallucination control | Citation verification, number matching after unit normalisation with derived numbers flagged, calculation provenance for arithmetic, LLM-as-judge faithfulness compared with RAGAS, regression gate in CI. | `docs/learning/hallucination-control.md`, RAG-010, RAG-012, RAG-021 |
| Refusal | Explicit refusal gate with reasons (low retrieval confidence, out of scope, insufficient evidence, failed verification) and an unanswerable eval set measuring abstention precision/recall. | `docs/learning/refusal.md`, RAG-011 |

## Architecture

```
EDGAR (10-Q / 10-K) --> ingestion --> chunking --> indexing --> retrieval --> generation --> answer | refusal
                        parse to      pluggable    embed +     dense/BM25/   grounded prompt,
                        sections      chunkers     Chroma/FAISS hybrid+rerank citation check, refusal gate
                                                              \_______ evaluation + Langfuse traces _______/
```

See `docs/architecture.md` for the component table and the alternatives being compared.

## Quickstart

```bash
git clone https://github.com/kvnzhng/quarterly-RAG.git && cd quarterly-RAG

# 1. Python env (uv installs Python 3.12 if needed)
make setup

# 2. Config
cp .env.example .env   # set EDGAR_USER_AGENT (SEC requires a contact), and LLM_*/EMBED_* if not using local Ollama

# 3. Model provider, pick one (see "Choosing a model provider" below)
#    a) Ollama on this machine:
brew install ollama && ollama serve &
make models            # pulls the models named in .env through Ollama's HTTP API
#    b) Ollama on another machine: point LLM_BASE_URL / EMBED_BASE_URL at http://<host>:11434/v1 in .env, then
make models            # same command; the host is read from .env (or set OLLAMA_HOST)
#    c) a hosted API: nothing to install, just the token in .env

# 4. Sanity
uv run rag version
uv run rag config
make test
uv run rag doctor    # endpoint reachable, models listed, one chat + one embedding call, data dirs writable

# 5. Corpus: the last two years of 10-Q / 10-K filings, about eight per company, with a manifest
uv run rag ingest download --ticker AAPL --ticker NVDA
uv run rag ingest parse --ticker AAPL --ticker NVDA   # -> sectioned JSONL with offsets
uv run rag chunk build --ticker AAPL --ticker NVDA    # -> chunks with the same offsets
uv run rag index build --ticker AAPL --ticker NVDA --context   # -> embeddings in ChromaDB
uv run rag index query "What were Apple's total net sales in Q3 FY2026?" --context
uv run rag eval retrieval -k 5 --context   # -> recall@k, MRR, nDCG with a run record
uv run rag ask "How many employees did Apple have at the end of fiscal 2025?"
uv run rag eval refusal                    # -> abstention precision/recall and the threshold sweep
```

Later tickets add `rag index`, `rag ask`, and `rag eval`. The commands are listed in `src/quarterly_rag/cli.py` as they are planned.

## Choosing a model provider

The pipeline reaches models through two small interfaces, `LLM` and `Embedder`, configured entirely from `.env` (ADR-005). The defaults keep everything on your machine and free; which provider you use is your call.

| Setup | `LLM_PROVIDER` | `LLM_BASE_URL` | `LLM_API_KEY` | Notes |
|---|---|---|---|---|
| Ollama on this machine (default) | `openai_compatible` | `http://localhost:11434/v1` | `ollama` (ignored) | `make models` pulls the weights |
| Ollama, vLLM, LM Studio, llama.cpp on another machine | `openai_compatible` | `http://<host>:<port>/v1` (the `/v1` matters) | whatever that server expects | nothing to install locally; for Ollama, `make models` pulls onto that host |
| Ollama on a machine with 18 GB+ (recommended) | `openai_compatible` | as above | as above | `LLM_MODEL=qwen3.8-27b-64k:latest`, the only model measured that is both fully grounded and accurate; `gpt-oss:20b` is 2.5x faster and less accurate. The 8B default fails citation discipline (`docs/tradeoffs/llm-serving.md`) |
| Hosted OpenAI-compatible API (OpenAI, OpenRouter, Groq, ...) | `openai_compatible` | the provider's URL | your token | costs money, so evals stop being free |
| Anthropic API | `anthropic` | unused | your token | `LLM_MODEL=claude-opus-5`; no embeddings endpoint, keep `EMBED_*` local |

Embeddings are configured separately (`EMBED_PROVIDER`, `EMBED_BASE_URL`, `EMBED_MODEL`) because a hosted chat model is usually best paired with local embeddings: the index gets rebuilt often and embedding cost adds up.

Every eval number in `docs/` carries a run record: git commit, corpus manifest hash, chunker config, embedding model, retrieval parameters, prompt version, and the provider and model that produced it. Local 8B model vs hosted frontier model on the same eval set is one of the planned comparisons (`docs/tradeoffs/llm-serving.md`).

## Repository layout

```
src/quarterly_rag/     pipeline layers: ingestion, chunking, indexing, retrieval, generation, evaluation, observability
tests/               unit tests (no network); integration tests are marked and skipped by default
data/                raw/, processed/, indexes/ are gitignored; eval/ sets are committed
docs/adr/            architecture decision records, one per real decision
docs/tradeoffs/      X vs Y comparisons; a page counts once it has measured numbers
docs/learning/       one page per competency: concepts, what this repo does, talking points
project/tickets.md   the roadmap as ordered tickets: one thin end-to-end path first, then comparisons
scripts/             repo tooling (commit message check used by the git hook and CI)
infra/               docker compose for Langfuse
notebooks/           exploration only
```

## Workflow

Work is ticket-driven. Each change references a ticket (`feat(retrieval): add BM25 (RAG-009)`). `make setup` installs a `commit-msg` hook that rejects messages without one, CI runs the same check over every push and pull request, and a Claude Code edit hook blocks edits with no active ticket. `AGENTS.md` is a symlink to `CLAUDE.md` so Codex follows the same rules. See `CLAUDE.md`.

## Status

Ordered as in `project/tickets.md`: one thin, measured end-to-end path first, then the comparisons.

- [x] RAG-001 scaffolding, tooling, roadmap
- [x] RAG-016 rename to quarterly-RAG
- [x] RAG-017 reading list
- [x] RAG-018 provider-agnostic model configuration
- [x] RAG-022 / 023 portable ticket enforcement, docs match the plan
- [x] RAG-002 model clients and `rag doctor`
- [x] RAG-003 EDGAR downloader with manifest
- [x] RAG-004 section parser
- [x] RAG-019 evaluation set v0 (43 questions, evidence spans, question types)
- [x] RAG-005 v1 chunker (fixed window, tables atomic)
- [x] RAG-006 embeddings, Chroma, dense retrieval
- [x] RAG-008 retrieval metrics, run record, baseline
- [x] RAG-010 grounded generation with verified citations
- [x] RAG-011 refusal policy with abstention metrics
- [x] RAG-009 hybrid retrieval (dense + BM25 fusion), the new default
- [x] RAG-020 chunking comparison, section-aware is the new default
- [x] RAG-007 vector store comparison, ChromaDB stays the default
- [x] RAG-012 faithfulness judge and the regression gate
- [ ] RAG-021 calculation provenance
- [ ] RAG-013 Langfuse
- [ ] RAG-014 API + UI
- [ ] RAG-015 writeup

## Reading and courses

Grouped by the competency they support, in roughly the order the tickets need them. Every link was checked when added; papers link to arXiv abstracts.

**Start here**
- Lewis et al. 2020, [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). The paper that named the pattern.
- Gao et al. 2023, [Retrieval-Augmented Generation for LLMs: A Survey](https://arxiv.org/abs/2312.10997). Naive vs advanced vs modular RAG; a map of every lever this repo pulls.
- Barnett et al. 2024, [Seven Failure Points When Engineering a RAG System](https://arxiv.org/abs/2401.05856). Short and practical. Read before RAG-003.
- Eugene Yan, [Patterns for Building LLM-based Systems and Products](https://eugeneyan.com/writing/llm-patterns/). Evals, RAG, and guardrails in one long post.
- Course: DeepLearning.AI, [Building and Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/). Introduces the RAG triad: context relevance, groundedness, answer relevance.

**Grounding** (RAG-004, RAG-010)
- Anthropic, [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval). Prepend document context to each chunk before embedding, with measured gains.
- Min et al. 2023, [FActScore](https://arxiv.org/abs/2305.14251). Split an answer into atomic claims and verify each one. The model for our citation verifier.
- Liu et al. 2023, [Lost in the Middle](https://arxiv.org/abs/2307.03172). Models under-use context placed mid-prompt, which affects how many chunks to pass and in what order.

**Chunking** (RAG-005, RAG-020)
- Jimeno Yepes et al. 2024, [Financial Report Chunking for Effective RAG](https://arxiv.org/abs/2402.05131). Element-based chunking of 10-Ks. Directly relevant.
- Chroma Research, [Evaluating Chunking Strategies for Retrieval](https://research.trychroma.com/evaluating-chunking). Token-level recall metrics for chunkers.
- Pinecone, [Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/). Survey of the common strategies.
- Sarthi et al. 2024, [RAPTOR](https://arxiv.org/abs/2401.18059). Hierarchical summaries as retrieval units, the ambitious version of parent-child chunking.
- Course: DeepLearning.AI, [Preprocessing Unstructured Data for LLM Applications](https://www.deeplearning.ai/short-courses/preprocessing-unstructured-data-for-llm-applications/). Parsing HTML and PDF with tables.

**Retrieval quality** (RAG-006 to RAG-009)
- Karpukhin et al. 2020, [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906). Why bi-encoders work for retrieval.
- Robertson and Zaragoza 2009, [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019). The sparse baseline that still wins on exact terms.
- Cormack et al. 2009, [Reciprocal Rank Fusion](https://doi.org/10.1145/1571941.1572114). The one-formula way to merge dense and sparse results.
- Nogueira and Cho 2019, [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085). Cross-encoder reranking.
- Khattab and Zaharia 2020, [ColBERT](https://arxiv.org/abs/2004.12832). Late interaction, the middle ground between bi-encoders and cross-encoders.
- Gao et al. 2022, [HyDE](https://arxiv.org/abs/2212.10496). Hypothetical document embeddings for query rewriting.
- Thakur et al. 2021, [BEIR](https://arxiv.org/abs/2104.08663) and the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard). How embedding models are benchmarked, and why the leaderboard is not your corpus.
- Malkov and Yashunin 2016, [HNSW](https://arxiv.org/abs/1603.09320) and Johnson et al. 2017, [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734). The index structures behind FAISS and Chroma.
- Weaviate, [Hybrid Search Explained](https://weaviate.io/blog/hybrid-search-explained).
- Courses: DeepLearning.AI, [Advanced Retrieval for AI with Chroma](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/); Hugging Face Cookbook, [Advanced RAG](https://huggingface.co/learn/cookbook/en/advanced_rag).

**Hallucination control** (RAG-010, RAG-012)
- Ji et al. 2023, [Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629). Taxonomy and vocabulary.
- Manakul et al. 2023, [SelfCheckGPT](https://arxiv.org/abs/2303.08896). Sampling-based consistency checks without a reference answer.
- Dhuliawala et al. 2023, [Chain-of-Verification](https://arxiv.org/abs/2309.11495). The model drafts verification questions and answers them before finalizing.
- Asai et al. 2023, [Self-RAG](https://arxiv.org/abs/2310.11511) and Yan et al. 2024, [Corrective RAG](https://arxiv.org/abs/2401.15884). Retrieve, critique, and retry as a loop.

**Refusal** (RAG-011)
- Rajpurkar et al. 2018, [Know What You Don't Know (SQuAD 2.0)](https://arxiv.org/abs/1806.03822). Unanswerable questions as a first-class eval set. The template for `data/eval/unanswerable.jsonl`.
- Kamath et al. 2020, [Selective Question Answering under Domain Shift](https://arxiv.org/abs/2006.09462). Abstention as a calibrated decision, with coverage vs accuracy curves.
- Kadavath et al. 2022, [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221). Whether a model's own confidence is usable as a refusal signal.

**Evaluation** (RAG-019, RAG-008, RAG-012)
- Es et al. 2023, [RAGAS](https://arxiv.org/abs/2309.15217). Reference-free faithfulness, answer relevance, and context precision.
- Saad-Falcon et al. 2023, [ARES](https://arxiv.org/abs/2311.09476). Training lightweight judges with synthetic data.
- Zheng et al. 2023, [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685). Biases of LLM judges and how to check for them.
- Hamel Husain, [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/). The practitioner's version: look at your data and build the harness first.

**Financial documents** (RAG-003, RAG-004, RAG-019)
- Islam et al. 2023, [FinanceBench](https://arxiv.org/abs/2311.11944). QA over 10-Ks with measured RAG failure rates. A source of question styles.
- Chen et al. 2021, [FinQA](https://arxiv.org/abs/2109.00122). Numerical reasoning over financial tables.
- Loukas et al. 2021, [EDGAR-CORPUS](https://arxiv.org/abs/2109.14394). A sectioned 10-K corpus, useful for the Item detection in RAG-004.
- SEC, [EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), [fair access policy](https://www.sec.gov/os/accessing-edgar-data), and [How to Read a 10-K](https://www.sec.gov/files/reada10k.pdf).

**Tooling**
- [LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/), [LlamaIndex production RAG guide](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/), [Chroma](https://docs.trychroma.com/), [FAISS wiki](https://github.com/facebookresearch/faiss/wiki), [sentence-transformers](https://sbert.net/), [RAGAS](https://docs.ragas.io/), [Langfuse](https://langfuse.com/docs), [Ollama](https://docs.ollama.com/).
- Course: DeepLearning.AI, [LangChain: Chat with Your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/).

**Books and foundations**
- Chip Huyen, *AI Engineering* (O'Reilly, 2025). Chapters on RAG, agents, and evaluation methodology.
- Jay Alammar and Maarten Grootendorst, *Hands-On Large Language Models* (O'Reilly, 2024). Embeddings, semantic search, and RAG with code.
- Stanford [CS224N](https://web.stanford.edu/class/cs224n/) for the NLP foundations (attention, retrieval, evaluation).

## License

MIT
