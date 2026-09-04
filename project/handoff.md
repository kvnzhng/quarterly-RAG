# Handoff: continue quarterly-RAG from a fresh session

Paste everything below the line into a new Claude Code session opened in this repository.

---

You are continuing **quarterly-RAG**, a local RAG system over SEC 10-Q/10-K filings for Apple and Nvidia, built as a job-hunt portfolio piece to demonstrate five competencies with measured tradeoffs: grounding, chunking, retrieval quality, hallucination control, and refusal. The owner is Kevin. Read `CLAUDE.md` first: it holds the current state, the rules, and the lessons. Then `project/tickets.md` for the roadmap and `project/conventions.md` for how work is done here.

## Where things stand

Phases one and two are complete. The pipeline answers from the filings or refuses with a reason, and every layer is measured against a 63-question human-verified eval set (33 answerable, 30 that must be refused). A derived number can carry the arithmetic that produced it, recomputed from the passages its operands cite, behind `ANSWER_PROMPT_VERSION=2` (RAG-021). It is off by default because the gate measured that it costs two answerable questions. Ten ADRs under `docs/adr/` record the decisions; each has a tradeoff page with numbers under `docs/tradeoffs/`. The headline numbers are in the README under "Results so far" and in `data/eval/baseline.json`, which `make eval` gates against.

The next ticket is **RAG-013, Langfuse tracing**, then RAG-014 FastAPI and Streamlit and RAG-015 the results writeup. RAG-029 holds three named limits of the calculation verifier, each with an example, and is worth taking before the writeup if the numbers matter to it. The backlog in `project/tickets.md` has the full text of each.

## Setup facts you need

- Models run on Kevin's **network Ollama server**. Its address is in `.env` and must never appear in the repo, a commit message, a ticket, or your replies. Call it "the network Ollama server". There is no Ollama on the laptop; `make models` pulls over the server's HTTP API.
- The server holds `llama3.1:8b`, `gpt-oss:20b`, `qwen3.6:27b`, `qwen3.8-27b-64k:latest`, `deepseek-r1:32b`, `gemma4:26b`, `mistral-small`, and `nomic-embed-text`. `llama3.1:8b` is the code default because ADR-003 requires laptop-sized defaults, but it invents citations in half its answers. **Use `qwen3.8-27b-64k:latest` for generation and as the judge, and `gpt-oss:20b` when latency matters.** Set `LLM_MODEL` in `.env` or per command with `LLM_MODEL=... uv run rag ...`.
- Corpus, chunks and indexes are on disk under `data/` and gitignored; the eval set and baseline under `data/eval/` are committed. If `data/` is missing, rebuild with the commands in `CLAUDE.md` under Build / Test / Run, in order: download, parse, chunk build, index build with `--context`.
- `make test` runs 365 unit tests with no model. `make test-all` adds the live tests. `make eval` runs the regression gate; it takes about ten minutes and needs `LLM_MODEL=gpt-oss:20b` and `--judge qwen3.8-27b-64k:latest` to reproduce the committed baseline.

## How Kevin works

- One ticket at a time, in roadmap order unless a measurement argues for reordering, in which case say so and do it. Claim the ticket in `project/tickets.md`, write its id to `.claude/active-ticket`, commit with the id, close it with a **Verified** line naming what was actually run, and clear the active ticket. The commit-msg hook rejects a commit without a ticket id.
- Push after each ticket and watch CI with `gh run watch`. Kevin often says "push and continue".
- Kevin asks "explain like I'm five" when a recap uses a term he has not met. Answer plainly, with the project's own numbers as the example, before continuing.
- Kevin reviews eval labels himself. When labels change, publish a review page (the pattern is in the RAG-019 ticket) and wait for verdicts.
- Report negative results with the same weight as positive ones. Reranking made things worse, the retrieval threshold is useless, RAGAS is anti-correlated: those are in the docs with numbers, and Kevin values them.

## Rules that were learned the hard way

- **Measure before writing a number.** Every number in `docs/` carries a run record. An estimate that ships as a fact gets corrected in a later commit, and that has happened.
- **Verify a documentation edit landed before committing a message that says it did.** Use `scripts/edit_docs.py`: it applies each change independently, writes after each success, and prints which anchors it could not find. Two commits once described changes that had silently failed because an all-or-nothing helper discarded everything on one stale anchor.
- **After any reset or amend, check every commit hash in `project/tickets.md` resolves** (`git cat-file -e <hash>^{commit}`).
- **State exactly what was measured.** "The ticker filter buys nothing" was true; "filtering buys nothing" was not, and it shipped.
- **Compare models at equal budgets.** `ANSWER_MAX_TOKENS` is 1024 because a thinking model at 400 scored 20 points low.
- **The eval set comes before the thing it measures.** That order found every real bug so far.

## Open threads worth knowing

- The eval set concentrates its evidence in 6 of 16 filings and is 70% tables; a second labelling round would sharpen every retrieval number.
- recall@20 is 72.7%: about a quarter of questions have evidence neither ranking nor chunking reaches, and nobody has looked at which since the chunker changed.
- `q052` (Nvidia's largest customers) leaks past every model's refusal.
- The gate scores the 23 `lookup` questions only, so calculation provenance is measured but ungated.
- Prompt wording moved the RAG-021 numbers more than the rule did; `docs/notes.md` has the comparison.
- The regression gate is a local command; CI never calls a model, and the `workflow_dispatch` job exists for anyone with a self-hosted runner.

Start by reading `CLAUDE.md`, running `make test` to confirm the tree is healthy, and claiming RAG-013.
