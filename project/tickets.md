# Tickets -- quarterly-RAG (Prefix: RAG)

> Next ID: RAG-038

Tickets are grouped by the competency they demonstrate. Each ticket names the
artifact it must leave behind (code, an eval number, a tradeoff doc, or an ADR)
so the repo tells the story on its own.

The backlog is ordered. Phase 1 ships one thin end-to-end path with the eval
set in place before anything is optimised. Phase 2 compares alternatives
against that baseline. Phase 3 is production readiness and the writeup.
Reordered on 2026-09-04 after an external review (see `docs/notes.md`).

## In Progress

## Backlog


### Phase 1: one thin end-to-end path, measured

### Phase 2: compare alternatives against the baseline




### Phase 3: production readiness and writeup





### RAG-032: Retrieval is unstable to phrasing, and only for Nvidia
- **Type:** fix
- **Created:** 2026-09-05
- **Competency:** retrieval quality
- **Description:** Found while fixing RAG-031, and larger than it. Asking for one company's annual total, filtered to that company: "What was Nvidia's revenue in 2025?" ranks its income statement 2nd, "in fiscal 2025" 1st, and "What was Nvidia's **total** revenue in 2025?" does not return it in the top 6. The same edits leave Apple at rank 1, including when Apple is asked in Nvidia's vocabulary. The mechanism is term frequency within one company's filings: Nvidia says "revenue" and "total revenue" in every geographic and segment table, so neither word discriminates and the income statement does not stand out; Apple's "net sales" is nearly unique to the line item. This is a plausible part of the standing recall@20 ceiling of 72.7%, which has never been explained.
- **Suggested approaches, none chosen yet:** query decomposition or rewriting with a model, which would put an LLM in the retrieval path for the first time and needs an ADR against the project's deterministic-retrieval preference; or a per-company term weighting in the BM25 half, which is deterministic and cheaper but only addresses the lexical side; or accepting it and documenting it. Four deterministic rewrites were already tried and rejected in RAG-031.
- **Done when:** the eval set has a handful of paraphrase pairs and multi-company questions, human-verified, and the chosen approach is measured against them with a before and after. The labels come first, as they did in RAG-019.

## Done

### RAG-037: The course sits near the top of the README
- **Type:** docs
- **Created:** 2026-09-05 | **Completed:** 2026-09-05
- **Competency:** all
- **Description:** "The course" was the second-to-last section of the README, below the reading list's neighbours. Kevin wants it higher and referenced in the intro. Move the section to just after "Why filings?", ahead of the architecture and the results, and add one sentence to the current-state paragraph pointing at it.
- **Done when:** a reader meets the course link before the architecture, and the intro names it.
- **Verified:** the README's section order is now intro, Why filings?, The course, Architecture, Results; the current-state paragraph links to the course; `scripts/edit_docs.py` reported all three edits applied; `make lint` clean. No code changed, so the tests were not re-run.
- **Commits:** `8dda18d`

### RAG-036: `make course` opens the notebook as an app, not the editor
- **Type:** docs
- **Created:** 2026-09-05 | **Completed:** 2026-09-05
- **Competency:** all
- **Description:** `make course` landed in marimo's edit mode. Kevin does not want readers dropped into the editor. Point it at `marimo run`, which serves the controls and the outputs without the code, keep the editor behind `make course-edit`, and say so in the README, `CLAUDE.md`, the notebook's docstring and the Notion setup step.
- **Done when:** `make course` runs `marimo run notebooks/course.py`, and every place that documents opening the notebook says which mode it lands in.
- **Verified:** `make -n course` resolves to `uv run marimo run notebooks/course.py` and `make -n course-edit` to the editor. App mode started headless on a spare port answered HTTP 200 within two seconds and stopped cleanly with nothing in its log. `ruff`, `marimo check`, `make lint` clean; `make test` 432 passed. `scripts/edit_docs.py` reported every Makefile, README, CLAUDE.md and notebook edit applied; the Notion setup step and the code callout say which mode `make course` lands in.
- **Not run:** the app in a browser.
- **Commits:** `41c43e0`

### RAG-035: Publish the course link and document how to run the notebook
- **Type:** docs
- **Created:** 2026-09-05 | **Completed:** 2026-09-05
- **Competency:** all
- **Description:** Kevin published the Notion course to the web. Point the README, the notebook's own links, `CLAUDE.md`, the notes and the handoff at the public address instead of the private workspace URL; add a `make course` target; say in the README what runs on open, what waits for a button, and how to run the notebook as an app or as a script; and put a code pointer at the top of the Notion page and in every chapter's notebook callout so the two halves refer to each other.
- **Done when:** a reader with no Notion account can open every course link in the repo, and the Notion pages link to the repository and the notebook.
- **Verified:** the public address and a chapter fetched by bare id both answer HTTP 200 with no Notion session (`curl`). The notebook holds 13 public links and no workspace URL; `ruff check`, `ruff format --check` and `marimo check` clean; `uv run python notebooks/course.py` exits 0; `make -n course` resolves to the marimo command; `make lint` clean; `make test` 432 passed. `scripts/edit_docs.py` reported every README, Makefile, CLAUDE.md, notes and handoff edit applied. On the Notion side the parent page carries a code callout at the top, and all twelve chapter callouts now link to the notebook on GitHub.
- **Not run:** the notebook in a browser; `make course` itself was not started, because it is a long-running server.
- **Commits:** `bbc06b8`

### RAG-015: Results writeup and interview talking points
- **Type:** docs
- **Created:** 2026-09-03 | **Completed:** 2026-09-05
- **Competency:** all
- **Description:** Fill the README with the architecture diagram, final eval tables, and one paragraph per competency (grounding, chunking, retrieval quality, hallucination control, refusal) explaining what was tried, what was measured, and what was chosen.
- **Done when:** a reader can understand the tradeoffs from the README alone.
- **Verified:** `make test` 432 passed before and after. `LLM_MODEL=gpt-oss:20b uv run rag eval baseline --judge qwen3.8-27b-64k:latest` run twice on 2026-09-05 at `3daa950`: identical to each other; recall@5, MRR, nDCG@5, citation resolution, fully grounded and correct identical to the baseline; answerable coverage 0.667 to 0.636, abstention F1 0.812 to 0.800, faithfulness 0.750 to 0.625, the last beyond the tolerance. Attributed with `rag eval generation --context retrieved --types lookup` (report `generation-retrieved-20260905T084138`) against the baseline-day `generation-retrieved-20260904T193319`: 13 of 23 answers byte-identical, two judge flips on spacing-only changes, judge looser 0. Model digests and the server version unchanged since August. The four chunkers re-measured at `3daa950` with `rag eval retrieval -k 20 --context --strategy <s>` (reports `retrieval-context-20260905T0835*`), reproducing the chunking page exactly and explaining 0.449 against 0.440 as depth 20 against depth 10. Every number in the README's new sections was matched against a docs page, a ticket, a report or the baseline by script; the only tokens it could not find were `2025.`, `63,` and `93.8%`, which is 15 of 16. `scripts/edit_docs.py` reported every README, CLAUDE.md, notes and handoff edit applied.
- **The finding:** the gate is deterministic within a day and not across days, and its five-point tolerance sits below faithfulness's 6.25-point granularity. The baseline was not re-accepted: no code change caused the drop, and accepting it would bury the finding. Recorded in the README, `docs/notes.md`, `CLAUDE.md` and chapter 10 of the course.
- **Not run:** the gate with `qwen3.8-27b-64k` answering; any gold-passage generation eval; `make eval-accept`.
- **Commits:** `4bdf047`

### RAG-034: Course in Notion with a marimo notebook
- **Type:** docs
- **Created:** 2026-09-05 | **Completed:** 2026-09-05
- **Competency:** all
- **Description:** Turn what the repo learned into a course a reader with basic coding and a little RAG can follow: Notion chapters that explain the tooling and the tradeoffs with this project's own numbers, a marimo notebook (`notebooks/course.py`) that lets the reader change chunker, retrieval strategy, k, filters, prompt version and model and see what happens, and a closing chapter on what comes next: further optimisation, weird questions, attacks on the chat, and what building this for a company adds. The pages link to notebook sections and the notebook links back to the pages. Requested by Kevin on 2026-09-05, ahead of finishing RAG-015.
- **Done when:** the course pages exist in Notion with a map on the parent page, the notebook runs against the local corpus and the configured endpoint, `make lint` and `make test` are clean, and the README points at both.
- **Dependencies added:** `marimo` (dev group): the notebook format the course is built on; reactive cells are what make the parameter sweeps one-click.
- **Verified:** twelve chapter pages plus the parent page with the course map exist in Notion, as a private draft in Kevin's workspace; the parent URL is in the README and in the notebook. `notebooks/course.py`: `ruff check` and `ruff format --check` clean, with `E501`, `B018`, `N803` and `N806` ignored for `notebooks/*.py` only, because a marimo cell ends in a bare expression and holds prose; `marimo check` clean; `uv run python notebooks/course.py` runs every ungated cell against the corpus and the configured endpoint and exits 0. Every model-gated cell was then run once through marimo's cell runner with its button pressed, against the live pipeline: the endpoint check, the raw-against-context similarity, the retrieval eval, an answer with `llama3.1:8b` on q001, the judge with `qwen3.8-27b-64k`, a refusal (Tesla, `out_of_scope`), the injected-question preset (refused as `insufficient_evidence`), the passage injection (the model did not follow it) and the phrasing experiment. `make lint` clean, `make test` 432 passed.
- **Not run:** the 63-question refusal sweep cell, which is the only cell that calls the model more than a handful of times; and the notebook was not opened in a browser, so the editor's rendering of the widgets is unverified. Chapter 10 of the course carries the gate drift measured today; the README's version of it lands with RAG-015.
- **Commits:** `d9fbe11`


### RAG-033: Refresh the handoff for the writeup
- **Type:** docs
- **Created:** 2026-09-05 | **Completed:** 2026-09-05
- **Competency:** foundation
- **Description:** `project/handoff.md` still described the state before RAG-013, RAG-014, RAG-021, RAG-029, RAG-030 and RAG-031 landed.
- **Done when:** a fresh session can start RAG-015 from the handoff alone.
- **Verified:** the handoff names what is on main, the 432-test count, the gate command that has to be run before any number is quoted and why it should reproduce, the six findings worth writing about, the SSH push workaround, and the six open threads. No server address appears in it.


### RAG-031: A question naming two companies retrieves only one of them
- **Type:** fix
- **Created:** 2026-09-05 | **Completed:** 2026-09-05
- **Competency:** retrieval quality
- **Description:** "Who made more revenue in 2025, Nvidia or Apple?" was refused while each company answered on its own. Reported by Kevin from the page (RAG-014).
- **Done when:** the comparison question is answered with a citation from each company, and `docs/learning/retrieval-quality.md` reports recall for multi-company questions before and after.
- **Fixed:** the retriever asks each named company separately and interleaves by rank, so both reach the answer. Every wording tested now returns 3 Apple and 3 Nvidia where the worst previously returned 6 Nvidia and 0 Apple. Merging by score was rejected in the design because the scores were what was lopsided. The ticker filter was innocent throughout: it already declines to filter when two companies are named.
- **Not fixed, and this is the ticket's real result:** the comparison question is *still refused*. With three slots each, Apple's total is at rank 1 and Nvidia's at rank 9. Reaching for a query rewrite found the actual problem: retrieval is unstable to phrasing, and only for Nvidia. "What was Nvidia's revenue in 2025?" puts its income statement at rank 2, "in fiscal 2025" at rank 1, and adding the word "total" puts it outside the top 6. The same three edits leave Apple at rank 1, including asking Apple in Nvidia's vocabulary. The mechanism is term frequency inside one company's filings: Nvidia's documents say "revenue" and "total revenue" in every geographic and segment table, so the words do not discriminate; Apple's "net sales" is nearly unique to the line item. A comparison question is not a special case, it is an unusual phrasing that exposed this. Four deterministic rewrites were tried and none reached the total, so no template was shipped: fitting one to a single example is fitting to noise. RAG-032 carries it.
- **Verified:** live retrieval against the corpus, k=6, before and after, all wordings in `docs/learning/retrieval-quality.md`. `make test` 432 passed, 15 deselected, up from 422, with 12 new tests for the split and the interleave including the cases it must not change. `make lint` clean.
- **Measured and unchanged:** `rag eval retrieval -k 5 --context` gives recall@5 48.5% and MRR 0.440, identical to the committed baseline, because not one of the 33 answerable questions names two companies. A fix its own eval cannot see is worth saying out loud.
- **Commits:** `4052c63`
- **Not done:** labelling comparison questions for the eval set. They would measure a question the system still refuses, and they need Kevin's review, so they belong with RAG-032 rather than ahead of it.


### RAG-029: Three named limits of the calculation verifier
- **Type:** fix
- **Created:** 2026-09-04 | **Completed:** 2026-09-05
- **Competency:** hallucination control
- **Description:** RAG-021 left three cases where a correct answer fails the operand check. A unitless operand did not match a percentage; a scale constant written into a prose sentence was flagged as a figure the passage does not state; and a calculation could not use another calculation's result.
- **Done when:** each of the three has a decision with a measurement behind it, and `docs/learning/hallucination-control.md` carries the new numbers.
- **All three reproduced first**, including the chained one the ticket said to confirm.
- **Decision: widen the check, three times, each narrowly.** (1) A unitless operand may match a percentage the passage prints, *for calculation operands only*. In prose, "the rate was 46.9" against a passage's `46.9%` is still unsupported: that was a deliberate decision in RAG-010 with a test asserting it, and the test failed when the first attempt widened the check everywhere. An operand carrying a currency or a scale word still does not match a percentage. (2) An operand may be the result of an earlier *verified* calculation in the same answer; one taken from a line that did not verify is still refused, so the chain is only as good as its first link. (3) A bare scale constant is exempt in prose only when it is also an operand of a calculation the answer actually wrote, so "Apple had 100 stores" is still a figure that has to be in a passage.
- **Not chosen: tightening the prompt.** All three were the model doing something reasonable that the checker read too strictly, and the prompt already asks for what it does. Making the prompt longer to work around a checker is how the RAG-021 wording trap happened.
- **Verified:** gold passages, the 5 `derived` and 5 `cross_period` questions, commit `fdbb7ef` before and this ticket's code after. `qwen3.8-27b-64k` calculations verified 7 of 8 to 8 of 8, with no calculation failures left at all. `llama3.1:8b` 6 of 10 to 7 of 10, and every figure accounted for 6/10 to 7/10. Judged correct unchanged for both, at 10/10 and 8/10, so nothing started passing for the wrong reason. The three failures that remain are all one thing, an operand cited to a passage that does not contain it, which is the check working. Reports `generation-gold-20260905T074351` and `T074808`. `make test` 422 passed, 15 deselected, up from 413; `make lint` clean.
- **Not run:** the regression gate. It scores `lookup` questions, where prompt v1 is the default and no calculation is written, so none of this can reach it.
- **Commits:** `aadb2ba`


### RAG-030: Enter does not submit a question in the page
- **Type:** fix
- **Created:** 2026-09-05 | **Completed:** 2026-09-05
- **Competency:** production readiness
- **Description:** Typing a question in the Streamlit page and pressing Enter did nothing; the Ask button had to be clicked. Streamlit re-runs the script when the text input changes, but the button is a separate widget that reads false on that re-run, so the keypress was swallowed. Reported by Kevin on the first real use of the page (RAG-014).
- **Done when:** Enter in the question box asks the question, the button still works, and the model is not called twice for one question.
- **Commits:** `ed17f86`
- **Verified:** the input and the button are one `st.form`, which submits on either. Used the running page: typing a question and pressing Enter answered it, the page now shows its own "Press Enter to submit form" hint, and one question produced one answer. No unit test, because this is Streamlit widget behaviour and a test of it would be a test of Streamlit.


### RAG-014: API and minimal UI
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-05
- **Competency:** production readiness
- **Description:** FastAPI `POST /ask` returning the structured answer or refusal. Streamlit page that shows the answer, each citation with the highlighted source passage and filing link, and the refusal reason when refused.
- **Done when:** `make api` and `make ui` work locally and the README has a screenshot.
- **Verified:** `make api` serves `POST /ask` and `GET /health` on 127.0.0.1:8000; asked live against `qwen3.8-27b-64k`, the Services-share question returned the answer, one verified `CALC:` line and a citation carrying the whole passage, and the Microsoft question returned a 200 with `out_of_scope`. `make ui` renders both, and the README has three screenshots taken from the running page. `make test` 413 passed, 15 deselected, up from 390; `make lint` clean.
- **Design:** the wire contract is its own module rather than the internal `Answer` and `Refusal`, because `Refusal.best_chunks` holds whole chunks and would have shipped kilobytes of filing text per refusal by accident. The pipeline is built once in the lifespan, the endpoint is a plain `def` so Starlette runs the blocking model call in a threadpool, and refusing is a 200 because a refusal answers the question rather than failing the request. The page talks HTTP and nothing else, so what it shows is what any client sees.
- **Two bugs the screenshots found, both fixed:** Streamlit reads `$...$` as LaTeX, so an answer with two dollar amounts rendered everything between them as an equation; and the highlighter marked the footnote markers in Apple's product table, because `[c1]` parses as the figure 1 and every filing contains a 1. The verifier strips citation tags before checking figures for the same reason, and now so does the page.
- **Dependencies added:** `fastapi` 0.141.1, `uvicorn`, `streamlit` 1.63.0. Fifteen packages, nearly all of them Streamlit's data stack: pandas, pyarrow, altair, pydeck, pillow. That is the heaviest addition in the project and it buys one page.
- **Not done:** no authentication, and both servers bind to the loopback address only. This is a laptop tool; putting it on a network needs a decision that is not this ticket's.
- **Commits:** `d259a7b`


### RAG-013: Langfuse tracing (self-hosted)
- **Type:** chore
- **Created:** 2026-09-03 | **Completed:** 2026-09-05
- **Competency:** production readiness
- **Description:** Run Langfuse locally via docker compose (`infra/`). Trace every `rag ask` as its pipeline stages with token counts and latency. Push eval scores to Langfuse as scores on traces.
- **Done when:** a trace for a refused question and an answered question are visible in the local Langfuse UI. `docs/tradeoffs/observability.md` compares Langfuse vs Phoenix vs MLflow.
- **Verified:** `make langfuse-up` brings 6 containers healthy and creates the project and API keys on first boot, so nothing is clicked. Both traces exist and were read back through the API: the answered question is 7 spans, 9,577 ms total, generation 8,423 ms, retrieval 1,147 ms, verification 4 ms; the refused question is 2 spans, 1 ms, root level WARNING, no model call. The generation span carries the model name `openai_compatible/qwen3.8-27b-64k:latest` and 1,149 tokens, confirmed through the metrics API. `make test` 380 passed, 15 deselected, up from 365; two live tests against a real Langfuse pass and skip when it is absent; `make lint` clean.
- **The UI was confirmed by Kevin on 2026-09-05.** Reading a trace in the browser needs a sign-in and I do not enter passwords, so the API round trip above was my half of the evidence and Kevin looked at the traces himself. Both are the ticket's "visible in the local Langfuse UI".
- **Deviation from the ticket text:** the spans are the stages `rag ask` actually has, which are `scope-gate`, `retrieval`, `retrieval-gate`, `generation`, `verification` and `answer-gate`. Ingestion is offline and has no trace; reranking is off by default and would appear inside the retrieval span when enabled.
- **Measured (`docs/tradeoffs/observability.md`, ADR-011):** tracing costs about 150 ms a question, 3.18 s against 3.03 s median over five runs each way. Langfuse is 6 containers, 2,579 MB at idle and 5.6 GB of images, against Arize Phoenix at 453 MB and one command, and MLflow at 77 MB. Langfuse was chosen for its scores API and is off by default because of the footprint; only Langfuse was integrated, so the scores row for the other two is read from their documentation and says so.
- **Dependencies added:** `langfuse` 4.15.1, which brings `backoff`, `wrapt` and one OpenTelemetry exporter. Four packages only, because `chromadb` already brings most of OpenTelemetry. Imported lazily so `rag --help` and the tests never pay for it.
- **Surprise worth keeping:** Langfuse v4 defaults to `events_only` mode, where the legacy trace API is gone and the SDK's own `trace.get` returns 404. Observations read from `/api/public/v2/observations`, scores from `/api/public/v3/scores`. Recorded in `docs/notes.md`.
- **A bug the tests could not see.** `LangfuseTracer.span` caught the exception thrown into its own `yield` and did not re-raise, which suppresses it for the caller. With tracing on, a model-server error was swallowed, `Pipeline.ask` returned `None`, and the server's message was gone. The `NullTracer` and the recording fake have no try around their yield, which is why 380 tests were green. Fixed by guarding only the close, with two tests that fail against the old shape and one pipeline test that raises a `ModelServerError` through a real tracer.
- **A dead tracer cost eleven seconds.** `rag ask` against a closed Langfuse port still answered but took 13.9 s against a 3.0 s baseline, because the OpenTelemetry exporter retries with backoff inside `flush()`. `build_tracer` now probes the health endpoint once with a two-second timeout: 3.14 s. Wrong keys were never expensive at 3.89 s.
- **Refusals are scored as refusals.** `fully_grounded` is trivially true of an answer that made no claims, and the report computes its rate over answered questions only, so scoring it on a refusal would make Langfuse disagree with the report.
- **A test that passed while the feature did not.** The first integration test asserted the spans and called `tracer.score`, but never read the score back, so it was green while nothing proved scores landed at all. They do, on the right trace: filtering `v3/scores` by the trace id returns the score and a bogus id returns none. The test now asserts that, which is the half of "scores on traces" it was letting through.
- **Commits:** `f832879`, `b14ec8f`, `3fbda5c`, `842800c`


### RAG-021: Calculation provenance for derived numbers
- **Type:** feat
- **Created:** 2026-09-04 | **Completed:** 2026-09-04
- **Competency:** hallucination control
- **Description:** `derived` and `cross_period` questions need arithmetic (growth rates, differences, ratios, unit conversions), and a verbatim number check passes a wrong relationship between two correct numbers. Extend the answer format so the generator emits each derived number as a calculation: operands with citations, the operation, and the result. A deterministic verifier recomputes it from the cited operands and marks the result `verified` only when the recomputation matches within rounding.
- **Done when:** the RAG-019 `derived` and `cross_period` questions are scored separately, and `docs/learning/hallucination-control.md` reports the verified rate before and after.
- **Verified:** gold passages, the 5 `derived` and 5 `cross_period` questions, k=5, `ANSWER_MAX_TOKENS` 1024, commit `fdbb7ef`, corpus `ab54dafa27ee5fe1`, eval set `57ba5e0dfdb94790`, judge always a different model from the generator. `qwen3.8-27b-64k` before: 6 of 10 answered, 4 of the 5 `derived` questions refused outright, 0 derived figures verified because none was written; after: 10 of 10 answered, 7 of 8 calculations verified, 5 of 7 derived figures recomputed, 8 of 10 answers with every figure accounted for, 10 of 10 judged correct. `llama3.1:8b` before: 9 of 10 answered, 7 figures stated that no cited passage contains, 0 verified; after: 10 of 10 answered, 6 of 10 calculations verified, 2 of 10 derived figures recomputed, and all 4 calculation failures are an operand that is not in the passage it cites. Reports `generation-gold-20260904T191702`, `T192200`, `T192612`, `T192915`; lookup control `generation-retrieved-20260904T193319` (v1) and `T194220` (v2).
- **The gate decided the default.** `make eval` under v2 failed on `answerable_coverage`, 0.667 to 0.606, and gave the identical number on a second run. Under `ANSWER_PROMPT_VERSION=1` it reproduced all nine committed metrics to three decimals and passed. Prompt v2 costs two of the 33 answerable questions with `gpt-oss:20b`, so v1 stays the default and calculation provenance is opt-in. Not accepted into the baseline: a drop this change caused is not a new baseline.
- **Two defects in the RAG-010 verifier, found by measuring this one.** A unit word on the next line was read as this number's unit, so Apple's `$29,915` followed by `Percentage of total net sales` parsed as 29,915 percent and every answer quoting that figure had been scored ungrounded since RAG-010 (`242cd5c`). The first fix was then too tight: `gpt-oss:20b` writes a narrow no-break space between a figure and its unit, and restricting the gap to a space or a tab dropped the unit from every figure it wrote (`fdbb7ef`).
- **Known limits, each with an example in the learning doc:** the wrong two real figures still verify; a calculation cannot use another calculation's result; a unitless operand does not match a percentage; a scale constant written into prose is flagged. The last three are RAG-029. The gate scores `lookup` only, so calculation provenance is measured but ungated.
- **Not done:** no ADR, because this is not a tooling tradeoff. `rag eval refusal` and the refusal eval inside `rag eval baseline` disagree on the same measurement; recorded in `docs/notes.md`, not chased.
- **Commits:** `f4183b8`, `242cd5c`, `fdbb7ef`, `d7f5305`, `54a2e9c`


### RAG-001: Project initialization
- **Type:** chore
- **Created:** 2026-09-03 | **Completed:** 2026-09-03
- **Description:** Set up project scaffolding with ticket-based workflow, Python tooling (uv, ruff, pytest, CI), src layout, docs skeleton (ADRs, learning notes, tradeoff docs), and the roadmap below.
- **Commits:** `5094a17`

### RAG-016: Rename project to quarterly-RAG
- **Type:** chore
- **Created:** 2026-09-03 | **Completed:** 2026-09-03
- **Description:** Public name is `quarterly-RAG` (GitHub: kvnzhng/quarterly-RAG). Rename the Python package to `quarterly_rag`, the distribution to `quarterly-rag`, and update docs, config defaults, and the README badge/links. CLI command stays `rag`.
- **Commits:** `606872e`

### RAG-017: Reading list and course material
- **Type:** docs
- **Created:** 2026-09-03 | **Completed:** 2026-09-03
- **Description:** Add a curated, link-checked reading and course list to the README, grouped by competency and mapped to tickets, plus a short Reading section on each `docs/learning/` page.
- **Commits:** `d395fb5`

### RAG-018: Provider-agnostic LLM and embedding configuration
- **Type:** chore
- **Created:** 2026-09-03 | **Completed:** 2026-09-03
- **Description:** The model provider is the user's choice: a local server (Ollama or any OpenAI-compatible endpoint, on this machine or on the network), or a hosted API with a token (OpenAI-compatible or Anthropic). Replace the Ollama-only settings with `LLM_PROVIDER` / `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` and matching `EMBED_*` settings, keep local as the default, update `.env.example`, README, Makefile, RAG-002, and record the decision in ADR-005.
- **Commits:** `a7c6d72`

### RAG-022: Ticket enforcement portable across clones and CI
- **Type:** chore
- **Created:** 2026-09-04 | **Completed:** 2026-09-04
- **Competency:** foundation
- **Description:** The commit-msg hook lives only in this checkout's `.git/hooks`, `make setup` installs only the pre-commit stage, CI does not check messages, and the Claude edit hook needs `jq`. Track the check as `scripts/check-commit-msg.sh`, wire it as a pre-commit `commit-msg` stage hook, install both stages from `make setup`, run the same script over the commit range in CI, and let `enforce-ticket.sh` fall back to `python3` when `jq` is missing. Commit `AGENTS.md` as a symlink to `CLAUDE.md` so Codex reads the same instructions.
- **Done when:** a fresh clone gets the commit-msg hook from `make setup`, a commit without a ticket id is rejected locally and in CI, and the existing history passes the CI check.
- **Commits:** `bdd176a`

### RAG-023: README, docs, and conventions match the reviewed plan
- **Type:** docs
- **Created:** 2026-09-04 | **Completed:** 2026-09-04
- **Competency:** all
- **Description:** The review found the README describing planned work in the present tense, a duplicated clone command, a layout line naming "RAG-001 ... RAG-015", and a status list missing RAG-016 to RAG-018; `project/conventions.md` still says the provider is "Ollama today". Fix these, add the run-record requirement to conventions, and update ticket references in `docs/`, `README.md`, and package docstrings for the RAG-005/RAG-020 split and the new RAG-019 and RAG-021.
- **Done when:** every ticket reference in `docs/` and `README.md` resolves to the right ticket and the README status list matches this file.
- **Commits:** `8e81db6`

### RAG-002: Model clients, `rag doctor`, and local model setup
- **Type:** chore
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** foundation
- **Description:** Implement the `LLM` and `Embedder` protocols with the `openai_compatible` provider (Ollama and other local servers, plus hosted OpenAI-style APIs) and the `anthropic` provider (ADR-005). Add `rag doctor`: configured endpoint reachable, configured models listed by the server, one chat round-trip and one embedding call succeed, data dirs writable. `make models` pulls the default Ollama models for people running Ollama themselves (honours `OLLAMA_HOST` for a remote Ollama). Kevin's local AI server address is provided at ticket start and goes in `.env`, never in the repo.
- **Done when:** `rag doctor` passes against a local Ollama and against a remote OpenAI-compatible server; unit tests mock the HTTP layer.
- **Artifacts:** `docs/tradeoffs/llm-serving.md` first pass (which local models were tried and why), ADR-006 model selection.
- **Dependencies added:** `anthropic` (official SDK for the `anthropic` provider: retries, typed errors, and it tracks API changes such as adaptive thinking and removed sampling parameters; a hand-rolled client would need re-verifying on every change). The OpenAI-compatible provider uses the existing `httpx`.
- **Verified:** `rag doctor` and the live integration test pass against Ollama 0.32.13 on the network server (address in `.env`); cold chat 3.8 s, warm 242 ms; embeddings 768-dim. Not exercised: Ollama on this laptop (not installed). `make models` now pulls over Ollama's HTTP API so no local CLI is needed.
- **Commits:** `b34731a`, `b17ae67`, `f0e9fb4`

### RAG-003: SEC EDGAR filing downloader
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** grounding (the corpus is the ground truth)
- **Description:** `rag ingest download --ticker AAPL --forms 10-Q,10-K --since 2023-01-01`. Resolve ticker to CIK via EDGAR, list filings from the submissions API, download the primary document, and write a manifest (`data/raw/<ticker>/manifest.json`) with accession number, form type, period of report, filing date, and source URL. Respect EDGAR fair-access rules (declared User-Agent, max 10 req/s).
- **Done when:** Apple and Nvidia 10-Q/10-K filings for the last 8 quarters are on disk with a manifest, re-running is idempotent.
- **Verified:** `rag ingest download --ticker AAPL --ticker NVDA` fetched 8 filings per company for the default two-year window (exactly eight quarters each, fiscal labels checked against the companies' own naming); the second run downloaded nothing and left both manifests byte-identical. Live EDGAR test in `tests/integration/`. `make test-all`: 72 passed.
- **Commits:** `de44ec4`

### RAG-004: Filing parser with section detection
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** grounding, chunking
- **Description:** Convert filing HTML to clean text. Detect SEC items (Item 1, 1A Risk Factors, 2/7 MD&A, 3 Quantitative disclosures, 8 Financial Statements). Preserve tables as pipe-delimited text with a table marker. Emit `data/processed/<ticker>/<accession>.jsonl`, one record per section, with provenance fields (ticker, form, period, section, char offsets, source URL).
- **Done when:** parser tests pass on one 10-Q and one 10-K per company, and a section coverage report shows every expected item was found.
- **Dependencies added:** `beautifulsoup4` and `lxml` (HTML parsing; `lxml` is the fast, lenient tree builder these 1-2 MB inline-XBRL documents need). Both are runtime dependencies, not dev-only.
- **Verified:** all 16 filings parse with zero missing critical items and zero false headings; every 10-K yields its 23 expected Items, Apple's 10-Qs 11, Nvidia's 9 (Part II Items 3 and 4 genuinely absent). `text[char_start:char_end] == record.text` holds for every record. `make test-all`: 91 passed.
- **Commits:** `1c1027c`, `2e2efc1`

### RAG-024: Section key renders Part IV as "Part IIII"
- **Type:** fix
- **Created:** 2026-09-04 | **Completed:** 2026-09-04
- **Competency:** grounding
- **Description:** `Section.key` builds the roman numeral with `"I" * part`, so Part IV becomes `Part IIII`. It shows on every 10-K Item 15 and 16 record, which is where Nvidia files its consolidated financial statements. Section keys are the handle for gold evidence spans (RAG-019) and citations (RAG-010), so fix before eval data is generated against them.
- **Done when:** `Part IV.Item 15` renders correctly, a test covers all four parts, and the corpus is re-parsed.
- **Commits:** `ef148b3`

### RAG-019: Evaluation set v0 with evidence spans and question types
- **Type:** feat
- **Created:** 2026-09-04 | **Completed:** 2026-09-04
- **Competency:** retrieval quality, hallucination control, refusal
- **Description:** Build `data/eval/questions.jsonl` before any chunker or index exists, so every later choice is measured against the same labels. Each record: `id`, `question`, `ticker`, `type` (`lookup` | `derived` | `cross_period` | `unanswerable`), `gold_answer`, and `evidence`: a list of spans `{accession, section, char_start, char_end}` into the RAG-004 output. Labels are spans, not chunk ids, so they survive a change of chunking strategy; a chunk counts as relevant when it overlaps a span. Start with 30 answerable questions (mostly `lookup`, a few `derived` and `cross_period`) and 10 `unanswerable` seeds across both companies. LLM-assisted drafting is fine; every record is human-verified against the filing. A loader and `rag eval check` verify that every span resolves to text in the processed filings and that the gold answer appears inside the evidence for `lookup` questions (test marked `integration` until sample filings are committed, see the open question in `docs/notes.md`).
- **Done when:** `rag eval check` reports every span resolves, and the set is committed under `data/eval/`.
- **Verified:** 43 questions, every one reviewed against the filing text by Kevin in two rounds (10, then 33). 23 lookup, 5 derived, 5 cross-period, 10 unanswerable split evenly between `out_of_scope` and `insufficient_evidence`. 35/35 spans resolve; `rag eval check` is the gate. Three findings from round two were label presentation, not facts, and are fixed.
- **Known limits for RAG-008:** evidence concentrates in 6 of the 16 filings, so the other 10 act as distractors and period filtering is under-tested; 30% of spans are prose and the rest tables.
- **Commits:** `fbb5f19`, `781ff64`, `d0159d5`

### RAG-005: Chunker protocol and v1 chunker
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** chunking
- **Description:** Define the `Chunker` protocol and the `Chunk` model with mandatory provenance (ticker, form, period, section, char offsets, source URL). Implement one chunker: a fixed token window with overlap applied within each RAG-004 section record, so no chunk crosses an Item boundary and a table is never split. Report chunk count and size distribution. The other strategies and the comparison are RAG-020.
- **Done when:** chunks from one 10-Q round-trip to JSONL with provenance intact, offsets map back into the section text, and unit tests cover the boundary and table rules.
- **Verified:** 1,391 chunks over the 16-filing corpus, median 304 words, p90 347, largest 809. `filing_text[char_start:char_end] == chunk.text` holds for every chunk, none crosses a section boundary, none holds half a table, ids are unique, and all 35 gold evidence spans overlap at least one chunk. 61 chunks exceed the target, every one of them a single table kept whole. `make test-all`: 128 passed.
- **Observations for RAG-020:** Nvidia's Part IV Item 15 yields 61 chunks in the FY2026 10-K alone, all sharing one section label, so a section filter buys nothing there; one gold span (q003) already straddles two chunks, which is the case parent-child chunking exists to fix; and overlap is whole-line, so 398 of 1,179 boundaries get none (325 because the preceding line exceeds the 60-word budget, 73 because a table is never carried forward).
- **Commits:** `6ec7cd3`, `b89bff6`

### RAG-006: Embeddings, VectorStore interface, ChromaDB adapter, dense retrieval
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** retrieval quality
- **Description:** Define a `VectorStore` protocol (add, query with metadata filter, persist, load, stats). Implement the ChromaDB adapter with persistence under `data/indexes/`. Embeddings come from the configured embed endpoint (ADR-005; `nomic-embed-text` on Ollama by default) behind the `Embedder` protocol so sentence-transformers models can be swapped in. Add `retrieve(question, k)` returning `RetrievedChunk`s from dense search only; BM25, filters, and reranking are RAG-009.
- **Done when:** `rag index build --ticker AAPL --store chroma` builds and reloads an index, and a query returns chunks with provenance.
- **Artifacts:** `docs/tradeoffs/embeddings.md` first pass.
- **Verified:** `rag index build --ticker AAPL --ticker NVDA` embeds 1,391 chunks into ChromaDB in 13 s per variant (768 dims, unit-normalised, cosine); reopening the directory finds them; `rag index query` returns ranked chunks whose text still resolves against the filing offsets. `make test-all`: 155 passed.
- **Found by measuring:** `nomic-embed-text` needs `search_query:` / `search_document:` prefixes and was getting neither, costing a third of recall with no error. The `Embedder` protocol is now `embed_documents` / `embed_query` so the mistake is not expressible. Prepending a company/period/section header before embedding roughly doubles recall; both variants are built and kept for RAG-008.
- **Baseline:** dense-only recall@5 is 18.2% raw and 36.4% with context headers, over 33 answerable questions. That is the floor RAG-009's BM25, fusion and reranking are measured against.
- **Dependencies added:** `chromadb` (embedded vector store with metadata filtering and directory persistence; FAISS goes behind the same protocol in RAG-007).
- **Commits:** `3435fc9`

### RAG-008: Retrieval metrics, run record, and baseline numbers
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** retrieval quality
- **Description:** Implement recall@k, MRR, and nDCG@k over the RAG-019 set, with a chunk counted relevant when it overlaps a gold evidence span (overlap rule configurable, default any overlap). Break results down by company, form, section, and question type. Every report embeds a **run record**: git commit, corpus manifest hash, parser version, chunker name and config, embedding provider and model, vector store, retrieval parameters (k, filters), prompt version where relevant, and timestamp. `rag eval retrieval --k 5` prints the table and writes `reports/retrieval-<timestamp>.json`.
- **Done when:** the eval runs end to end on the RAG-005 + RAG-006 baseline and the numbers, with their run record, are in `docs/learning/retrieval-quality.md`.
- **Verified:** `rag eval retrieval -k 5 [--context]` runs end to end on the RAG-005 + RAG-006 baseline, prints overall / near-miss / by-type / by-company / by-form tables and writes `reports/retrieval-<variant>-<timestamp>.json` with a full run record. Numbers are in `docs/learning/retrieval-quality.md`. `make test-all`: 180 passed.
- **Baseline:** context variant recall@5 36.4%, MRR 0.267, nDCG@5 0.232; raw variant 18.2%, 0.131, 0.096.
- **Findings for RAG-009:** the near-miss ladder shows retrieval reaching the right filing 90.9% of the time, the right section 63.6%, and the right chunk 36.4%, so the loss is inside the document and metadata filtering would buy little. All seven 10-Q questions score 0.0% against 46.2% for 10-K, because retrieval returns the management discussion of a filing instead of its condensed financial statements. Both point at exact-term matching (BM25) and reranking.
- **Commits:** `81a7c67`

### RAG-010: Grounded answer generation with verified citations
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** grounding, hallucination control
- **Description:** Prompt the LLM with retrieved chunks tagged by id and require inline citations `[c12]`. Post-process: every claim sentence must carry a citation that maps to a retrieved chunk. Numbers are checked against the cited chunk after parsing and unit scaling (thousands / millions / billions, rounding tolerance): a number found in the chunk is `verified`; one not found is marked `derived, unverified` and the answer says so, instead of being silently passed or hard-failed. Calculation provenance for derived numbers is RAG-021. Return a structured `Answer {text, citations, unsupported_sentences, derived_numbers}`. v1 targets `lookup` questions from RAG-019; `derived` and `cross_period` results are reported separately.
- **Done when:** `rag ask "What was Apple's Q2 FY24 revenue?"` returns an answer whose citations resolve to real chunks, unsupported sentences and derived numbers are flagged rather than silently returned, and baseline citation-resolution and number-match rates by question type are in `docs/learning/grounding.md`.
- **Verified:** `rag ask` returns an answer whose citations resolve to passages that were actually provided, with unsupported sentences and unverified figures labelled inline. Baselines by question type are in `docs/learning/grounding.md`. `make test-all`: 209 passed.
- **Baseline (23 lookup questions, prompt v1):** `gpt-oss:20b` on gold passages, citations resolve 100%, fully grounded 91%, states the gold figure 77%; on retrieved passages, 100% / 87% / 67% with 35% refused. `llama3.1:8b` on gold passages, 50% / 41% / 95%.
- **Finding:** citation discipline is a model capability. The 8B default invents passage labels in half its answers while being the best of three at finding the right figure. ADR-006 amended; `docs/tradeoffs/llm-serving.md` has the table. Set `LLM_MODEL=gpt-oss:20b` on hardware with room.
- **Known limit:** the verifier checks whether a figure is *present* in the cited passage, not whether the claim about it is true, so a wrong-column figure passes. RAG-021 recomputes derived numbers from their operands.
- **Commits:** `5c11778`

### RAG-025: Extend the citation-discipline comparison to qwen3.8-27b
- **Type:** docs
- **Created:** 2026-09-04 | **Completed:** 2026-09-04
- **Competency:** hallucination control, production readiness
- **Description:** RAG-010 measured three models and found citation discipline to be the largest single lever on grounding. The server also holds `qwen3.8-27b`, which was not in that run. Score it on the same 23 lookup questions in both contexts, add per-model answer latency, and update `docs/tradeoffs/llm-serving.md`, `docs/learning/grounding.md`, ADR-006 and the README recommendation with whatever the numbers say.
- **Done when:** the llm-serving table covers four models with a latency column and the recommended `LLM_MODEL` follows the measurement rather than a guess.
- **Verified:** four models scored on gold passages, two on retrieved, with latency. `qwen3.8-27b-64k` is 100% on every grounding measure end to end and 75% on the labelled figure, against 87% and 67% for `gpt-oss:20b`, at 9.6 s versus 3.8 s per answer. Recommendation and ADR-006 updated to match.
- **Commits:** `e041a64`

### RAG-011: Refusal policy and abstention evaluation
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** when to refuse to answer
- **Description:** Implement a refusal gate with explicit reasons: (a) retrieval confidence below threshold, (b) question outside corpus scope (company or period not indexed, non-financial question), (c) generator reports insufficient evidence, (d) citation verification fails. Grow the `unanswerable` seeds from RAG-019 into `data/eval/unanswerable.jsonl` (30+ questions that must be refused) and measure abstention precision/recall alongside answer accuracy.
- **Done when:** `docs/learning/refusal.md` reports the tradeoff curve between refusing too much and hallucinating, and thresholds are chosen from it.
- **Verified:** `docs/learning/refusal.md` carries the threshold sweep and names the operating point. `rag ask` refuses with a reason and shows its closest passages. `make test-all`: 235 passed.
- **Baseline (63 questions, 30 unanswerable, qwen3.8-27b-64k, k=5):** abstention precision 67.4%, recall 96.7%, F1 0.795, answerable coverage 57.6%, one leak.
- **Operating point:** `MIN_RETRIEVAL_SCORE=0.0`, the check off. The sweep shows raising it buys 3.3 points of recall and costs 45 points of coverage, because cosine scores cluster between 0.74 and 0.84. A calibrated signal would have to come from a reranker or token probabilities instead (RAG-009).
- **Deviation from the ticket:** the unanswerable questions live in `data/eval/questions.jsonl` rather than a separate `unanswerable.jsonl`. The schema already carries the type, so a second file would need a second loader, hash and check command for no gain.
- **Finding:** 13 of the 14 over-refusals had no evidence in the top 5, so coverage is bounded by retrieval rather than by the gate. Refusal calibration is also not answer quality: the 8B model has the best abstention F1 and invents citations in half its answers.
- **Commits:** `6a10b9a`

### RAG-009: Hybrid retrieval, metadata filtering, and reranking
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** retrieval quality
- **Description:** Add BM25 (rank_bm25) alongside dense retrieval with reciprocal rank fusion. Add metadata filters inferred from the question (ticker, fiscal period, section). Add a cross-encoder reranker (`bge-reranker-base`). Compare dense / BM25 / hybrid / hybrid+rerank on the RAG-008 eval.
- **Done when:** the comparison table is in `docs/tradeoffs/retrieval-strategies.md` and the best configuration becomes the default.
- **Verified:** `docs/tradeoffs/retrieval-strategies.md` holds the six-row comparison and ADR-008 records the decision. `hybrid` is the default in `Settings` and every command takes `--retrieval`. `make test-all`: 267 passed.
- **Baseline:** hybrid recall@5 45.5% against dense 36.4%, MRR 0.300 against 0.266.
- **Did not work, and worth knowing:** the inferred ticker filter changes recall not at all (the near-miss ladder had already said so); reranking raises recall@1 and lowers recall@5 with either judge, so it makes the system worse while making the ranking look better, and the case for a 2 GB cross-encoder is now weaker rather than stronger.
- **Finding for RAG-020:** every strategy plateaus at 45.5% by k=10, and hybrid at depth 100 reaches only 69.7%. Ten of 33 questions have their gold chunk nowhere in the top 100 of 1,391, mostly financial-statement tables. The ceiling is chunking, not ranking.
- **Dependencies added:** `rank_bm25` (small, pure Python; the index is built in memory at start-up, which is the scaling boundary to watch).
- **Commits:** `984bdb9`

### RAG-026: Filter on the fiscal period, not just the company
- **Type:** feat
- **Created:** 2026-09-04 | **Completed:** 2026-09-04
- **Competency:** retrieval quality
- **Description:** RAG-009 concluded that inferred metadata filtering buys nothing. That was measured on the **ticker** only, and it generalised too far. Filtering on the exact period label when a question names a specific quarter lifts recall@5 from 45.5% to 48.5% and unblocks the 10-Q questions that every strategy had scored at zero, because eight near-identical income statements stop competing. A bare fiscal year must not be filtered: a filing quotes prior years for comparison. Also fixes a Chroma adapter bug found while measuring, where a filter with more than one condition raised instead of filtering.
- **Done when:** the period filter is on by default, `docs/tradeoffs/retrieval-strategies.md` and ADR-008 no longer claim filtering buys nothing, and the multi-condition filter has a test.
- **Verified:** default `hybrid` reaches recall@5 48.5% and recall@10 51.5%, against 45.5% and 45.5% unfiltered; quarterly questions move from 0.0% to 14.3%. ADR-008 amended, tradeoff page corrected.
- **Commits:** `e9a13fb`

### RAG-020: Chunking strategy comparison
- **Type:** feat
- **Created:** 2026-09-04 | **Completed:** 2026-09-04
- **Competency:** chunking
- **Description:** Split from RAG-005. Add recursive character, section-aware with sub-splitting, and parent-child (small chunks for retrieval, parent section returned for generation) chunkers behind the `Chunker` protocol, each keeping full provenance. Extend the harness to report chunk count, size distribution, and boundary violations per strategy, and score every strategy with RAG-008 on the same RAG-019 labels.
- **Done when:** `docs/tradeoffs/chunking.md` contains a filled comparison table with run records and a recommendation, and an ADR records the default chunker.
- **Verified:** `docs/tradeoffs/chunking.md` holds the four-way comparison with run records and ADR-009 records the decision. `section-aware` is the default in `Settings` and in every CLI command. `make test-all`: 289 passed.
- **Result:** section-aware nearly doubles recall@1 (21.2% to 39.4%) and lifts MRR 43%, while recall@5 is unchanged at 48.5%. End to end the share of answers stating the labelled figure rose from 75% to 93% and abstention coverage from 57.6% to 63.6%.
- **Negative results:** the structure-blind recursive splitter is 21 points worse at k=5 and leaves 221 chunks holding half a table; parent-child came second because a 69-word child cut by word count still crosses topics while a titled block does not.
- **Remaining ceiling:** recall@20 is 72.7%, so about a quarter of questions have evidence neither ranking nor chunking reaches.
- **Commits:** `9b3e350`

### RAG-007: FAISS adapter and Chroma vs FAISS benchmark
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** retrieval quality
- **Description:** Implement a FAISS adapter (flat and HNSW) behind the same protocol. Benchmark both on the same corpus: build time, query p50/p95 latency, memory, metadata filtering support, persistence story, operational complexity.
- **Done when:** `docs/tradeoffs/vector-stores.md` is filled with measured numbers and ADR-007 records the default choice and when to pick the other.
- **Verified:** `docs/tradeoffs/vector-stores.md` holds the benchmark. The decision is recorded in **ADR-010**, not ADR-007 as this ticket said: ADR-007 was taken by the parser decision. `make test-all`: 300 passed.
- **Result:** all three stores give identical retrieval quality (recall@1 39.4%, recall@5 48.5%, MRR 0.440). FAISS HNSW is 16x faster per lookup and uses a fifth of the memory, which is invisible: the store is 3% of a retrieval against 31.4 ms to embed the question. ChromaDB stays the default on operational grounds (upsert, native filtering, one directory of state).
- **Dependencies added:** `faiss-cpu` (~30 MB, imported only when a FAISS store is built; worth carrying because the comparison is part of the project's purpose and the decision reverses at scale).
- **Commits:** `e15d165`

### RAG-027: Derive the CLAUDE.md file tree from git
- **Type:** chore
- **Created:** 2026-09-04 | **Completed:** 2026-09-04
- **Competency:** foundation
- **Description:** The file tree in `CLAUDE.md` was hand-edited after each ticket and had drifted: four test files were missing and the ordering was inconsistent. Regenerate it from `git ls-files` so it cannot drift again.
- **Done when:** every tracked path outside `reports/` and `notebooks/` appears exactly once, in a stable order.
- **Commits:** `b7c68d3`

### RAG-012: Faithfulness and end-to-end evaluation
- **Type:** feat
- **Created:** 2026-09-03 | **Completed:** 2026-09-04
- **Competency:** hallucination control
- **Description:** Add an LLM-as-judge faithfulness check (claims in answer entailed by cited context) using a local model, and compare against RAGAS. Add answer correctness against gold answers. `make eval` runs retrieval + generation evals and fails if scores regress below a stored baseline.
- **Done when:** `docs/tradeoffs/evaluation.md` compares RAGAS vs custom judge, and `make eval` is wired into CI as an optional job.
- **Verified:** `docs/tradeoffs/evaluation.md` compares the custom judge with RAGAS and `make eval` gates on `data/eval/baseline.json`, which is committed. `make test-all`: 331 passed.
- **Judge calibration:** against the deterministic number verifier over 57 cited sentences, 86% agreement, 6 sentences where the judge was stricter and 2 where it was looser. That 25% miss rate on unverified sentences is reported beside every faithfulness number.
- **RAGAS rejected on measurement:** 45 packages, uninstallable alongside langchain-community 0.4, and after pinning it back it scores faithful answers 0.0 and an unfaithful derived claim 1.0 with two different local judges. Removed rather than carried.
- **CI caveat:** the ticket asked for `make eval` as an optional CI job. It calls a model, which a project principle forbids in CI, and GitHub's runners cannot reach the endpoint or the corpus. The job exists behind `workflow_dispatch`; in practice the gate is a local command.
- **Commits:** `16ca8a7`, `b740d66`

### RAG-028: Document the state of the project and write the handoff
- **Type:** docs
- **Created:** 2026-09-04 | **Completed:** 2026-09-04
- **Competency:** all
- **Description:** Phases one and two are complete except RAG-021. Bring every top-level document up to what was actually built and measured: `CLAUDE.md` (stack, current state, lessons about working in this repo), `README.md` (results so far, no more "the plan"), `docs/architecture.md` (decided rows, real request flow), the learning pages' stale placeholders, and `docs/notes.md`. Record the process lessons in `project/conventions.md`. Write `project/handoff.md`: a prompt that lets a fresh session continue from here with the user's preferences and the project's rules intact.
- **Done when:** no top-level doc describes planned work in the present tense, every tradeoff status matches its ADR, and the handoff prompt names the next ticket, the setup facts, and the working rules.
- **Verified:** a grep for the old planning phrases across README, CLAUDE.md, architecture and the learning and tradeoff pages finds none; every tradeoff status names its ADR or says what is undecided; `project/handoff.md` names RAG-021, the server facts, the working rules and the open threads; no server address appears in any tracked file. `make test`: 331 passed.
- **Commits:** `2344195`, `2cca25e`
