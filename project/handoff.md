# Handoff: continue quarterly-RAG from a fresh session

Paste everything below the line into a new Claude Code session opened in this repository.

---

You are continuing **quarterly-RAG**, a local RAG system over SEC 10-Q/10-K filings for Apple and Nvidia, built as a job-hunt portfolio piece to demonstrate five competencies with measured tradeoffs: grounding, chunking, retrieval quality, hallucination control, and refusal. The owner is Kevin. Read `CLAUDE.md` first: it holds the current state, the rules, and the lessons. Then `project/tickets.md` for the roadmap and `project/conventions.md` for how work is done here.

## Where things stand

All three phases are built. Everything is merged to `main` and CI is green on it. The pipeline answers from the filings or refuses with a reason, every layer is measured against a 63-question human-verified eval set, and there are eleven ADRs each with a tradeoff page of numbers. `make test` runs 432 unit tests with no model and no Docker.

Since the last handoff: derived numbers carry their arithmetic and it is recomputed (RAG-021); every question can be traced to a self-hosted Langfuse (RAG-013); there is a FastAPI `POST /ask` and a Streamlit page over it (RAG-014); the operand check was widened three times (RAG-029); and a question naming two companies now asks each of them separately (RAG-031).

**The remaining ticket is RAG-015, the results writeup.** RAG-032 is also open and is a real piece of retrieval work, but the writeup is the last one in the original plan and is what the repo exists to produce.

## What RAG-015 has to do

Fill the README so a reader understands the tradeoffs from it alone: the architecture, the final eval tables, and one paragraph per competency saying what was tried, what was measured, and what was chosen.

**Check the numbers before quoting them.** RAG-029 changed what the verifier accepts, so start by running the gate:

```
LLM_MODEL=gpt-oss:20b uv run rag eval baseline --judge qwen3.8-27b-64k:latest
```

It should reproduce `data/eval/baseline.json` exactly, because RAG-029 is scoped to calculation operands and the gate scores `lookup` questions under prompt v1, where no calculation is written. If it does not reproduce, that is the first thing to understand and the writeup waits.

**The material is unusually good, and most of it is a negative result.** These are in `docs/learning/` and `docs/notes.md` with the numbers, and they are what makes the writeup worth reading:

- Prompt *wording* moved the results more than the rule did. Two v2 wordings differing only in where the worked example sits: with it last, `llama3.1:8b` verified 9 of 13 calculations and `gpt-oss:20b` lost 11 points of lookup faithfulness; with it in the body, faithfulness came back and the 8B model verified 6 of 10.
- The gate decided a default. Calculation provenance is off by default because `make eval` measured that it costs two of the 33 answerable questions, and a drop this change caused is not a new baseline.
- Measuring one verifier found a defect in another. A unit word on the next line was read as this number's unit, so every answer quoting Apple's `$29,915` had been scored ungrounded since RAG-010.
- A test passed while its feature did not: the tracing test called the scoring function and asserted only the spans.
- Retrieval is unstable to phrasing, and only for Nvidia. One word moves its income statement from rank 2 to outside the top 6, while the same edits leave Apple at rank 1. Term frequency inside one company's filings is the mechanism, and it is a candidate explanation for the recall@20 ceiling of 72.7% that has never been accounted for.
- An 8B model's arithmetic can be internally consistent over an operand it invented, and the judge scores that answer correct. It is the case the recomputation exists to catch.

The README already has "Results so far" and three screenshots of the page. What it does not have is the argument.

## Setup facts you need

- Models run on Kevin's **network Ollama server**. Its address is in `.env` and must never appear in the repo, a commit message, a ticket, or your replies. Call it "the network Ollama server". There is no Ollama on the laptop; `make models` pulls over the server's HTTP API.
- **Use `qwen3.8-27b-64k:latest` for generation and as the judge, and `gpt-oss:20b` when latency matters.** `llama3.1:8b` is the code default only because ADR-003 requires laptop-sized defaults, and it invents citations in half its answers. Set `LLM_MODEL` in `.env` or per command.
- Corpus, chunks and indexes are on disk under `data/` and gitignored; the eval set and baseline under `data/eval/` are committed. If `data/` is missing, rebuild with the commands in `CLAUDE.md`, in order: download, parse, chunk build, index build with `--context`.
- `make api` and `make ui` run the endpoint and the page. `make langfuse-up` starts tracing, which is off unless `LANGFUSE_HOST` and both keys are set. `ANSWER_PROMPT_VERSION=2` turns calculation provenance on.
- **Pushing over HTTPS fails with a 403**: the fine-grained token has metadata read only. Push with `git push git@github.com:kvnzhng/quarterly-RAG.git <branch>`, then `git fetch origin` and set the upstream. CI runs only on `main` and on pull requests.

## How Kevin works

- One ticket at a time, in roadmap order unless a measurement argues for reordering, in which case say so and do it. Claim it in `project/tickets.md`, write its id to `.claude/active-ticket`, commit with the id, close it with a **Verified** line naming what was actually run *and what was not*, and clear the active ticket. The commit-msg hook rejects a commit without a ticket id.
- Kevin asks "explain like I'm five" when a recap uses a term he has not met. Answer plainly, with the project's own numbers as the example, before continuing.
- Kevin reviews eval labels himself. When labels change, publish a review page and wait for verdicts.
- Report negative results with the same weight as positive ones. He asked for RAG-031 specifically because the learning mattered more than the feature, and the learning turned out to be bigger than the fix.
- He notices things from using the page. Two tickets in this session started as "I noticed that...". Take those seriously; both were real.

## Rules that were learned the hard way

- **Measure before writing a number.** Every number in `docs/` carries a run record.
- **Verify a documentation edit landed before committing a message that says it did.** Use `scripts/edit_docs.py` and read its PARTIAL lines.
- **After any reset or amend, check every commit hash in `project/tickets.md` resolves.**
- **State exactly what was measured.** "The ticker filter buys nothing" was true; "filtering buys nothing" was not, and it shipped.
- **A test that passes is not a feature that works.** Assert the thing the ticket promised, not the thing that is easy to assert.
- **Compare models at equal budgets**, and re-measure both sides when the verifier changes underneath them.

## Open threads worth knowing

- RAG-032: retrieval is unstable to phrasing, per company. Needs labelled paraphrase pairs first, and an ADR if a model goes into the retrieval path.
- The eval set concentrates its evidence in 6 of 16 filings, is 70% tables, and contains no comparison or paraphrase questions. A second labelling round would sharpen every retrieval number.
- recall@20 is 72.7% and nobody has explained the missing quarter. RAG-032 is the first real candidate.
- `q052` (Nvidia's largest customers) leaks past every model's refusal.
- `rag eval refusal` and the refusal eval inside `rag eval baseline` disagree on the same measurement, in opposite directions. In `docs/notes.md`, unchased.
- Langfuse scores land in `environment: default` while spans land in `local`, and the UI filters on that field.

Start by reading `CLAUDE.md`, running `make test` to confirm the tree is healthy, then the gate command above, and claiming RAG-015.
