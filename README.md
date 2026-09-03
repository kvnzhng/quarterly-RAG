# quarterly-RAG

[![ci](https://github.com/kvnzhng/quarterly-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/kvnzhng/quarterly-RAG/actions/workflows/ci.yml)

> Local, open-source Retrieval-Augmented Generation over SEC quarterly and annual filings (10-Q, 10-K), built to learn and demonstrate what "shipping a production RAG system" actually requires: **grounding, chunking, retrieval quality, hallucination control, and knowing when to refuse to answer.**

Everything runs on a laptop with no paid API: Ollama for the LLM and embeddings, ChromaDB and FAISS for vectors, Langfuse self-hosted for traces.

## Why filings?

10-Q and 10-K reports are free, public, long, highly structured, and full of exact numbers. That makes them a good corpus for RAG: retrieval is non-trivial (the same sections repeat every quarter), grounding is checkable (a revenue figure is either in the filing or it is not), and refusal has a concrete meaning (the period or company is not in the corpus).

Starting companies: **Apple (AAPL)** and **Nvidia (NVDA)**. Adding a ticker is a config change.

## The five competencies and where they live

| Competency | What the project does | Where to look |
|---|---|---|
| Grounding | Every chunk carries ticker, form, period, section, and offsets. Every answer sentence cites a chunk id that is verified to exist and to contain the quoted numbers. | `docs/learning/grounding.md`, RAG-004, RAG-010 |
| Chunking | Fixed, recursive, section-aware, and parent-child chunkers compared on retrieval metrics, not on intuition. | `docs/learning/chunking.md`, `docs/tradeoffs/chunking.md`, RAG-005 |
| Retrieval quality | Gold eval set with recall@k, MRR, nDCG. Dense vs BM25 vs hybrid vs hybrid + reranker. Chroma vs FAISS benchmark. | `docs/learning/retrieval-quality.md`, `docs/tradeoffs/vector-stores.md`, RAG-006 to RAG-009 |
| Hallucination control | Citation verification, number matching, LLM-as-judge faithfulness compared with RAGAS, regression gate in CI. | `docs/learning/hallucination-control.md`, RAG-010, RAG-012 |
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

git clone https://github.com/kvnzhng/quarterly-RAG.git && cd quarterly-RAG

# 1. Python env (uv installs Python 3.12 if needed)
make setup

# 2. Local models (RAG-002)
brew install ollama && ollama serve &
make models

# 3. Config
cp .env.example .env   # set EDGAR_USER_AGENT to your name and email (SEC requires it)

# 4. Sanity
uv run rag version
uv run rag config
make test
```

Later tickets add `rag ingest`, `rag index`, `rag ask`, and `rag eval`. The commands are listed in `src/quarterly_rag/cli.py` as they are planned.

## Repository layout

```
src/quarterly_rag/     pipeline layers: ingestion, chunking, indexing, retrieval, generation, evaluation, observability
tests/               unit tests (no network); integration tests are marked and skipped by default
data/                raw/, processed/, indexes/ are gitignored; eval/ sets are committed
docs/adr/            architecture decision records, one per real decision
docs/tradeoffs/      X vs Y comparisons; a page counts once it has measured numbers
docs/learning/       one page per competency: concepts, what this repo does, talking points
project/tickets.md   the roadmap, as tickets RAG-001 ... RAG-015
infra/               docker compose for Langfuse
notebooks/           exploration only
```

## Workflow

Work is ticket-driven. Each change references a ticket (`feat(retrieval): add BM25 (RAG-009)`), enforced by a git `commit-msg` hook and a Claude Code edit hook. See `CLAUDE.md`.

## Status

- [x] RAG-001 scaffolding, tooling, roadmap
- [ ] RAG-002 local models
- [ ] RAG-003 / 004 ingestion
- [ ] RAG-005 chunking
- [ ] RAG-006 / 007 vector stores
- [ ] RAG-008 / 009 retrieval quality
- [ ] RAG-010 grounded generation
- [ ] RAG-011 refusal
- [ ] RAG-012 faithfulness eval
- [ ] RAG-013 Langfuse
- [ ] RAG-014 API + UI
- [ ] RAG-015 writeup

## License

MIT
