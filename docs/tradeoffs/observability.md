# Observability: Langfuse vs Arize Phoenix vs MLflow tracing

**Status:** decided (ADR-011)
**Ticket:** RAG-013

## Question

When an answer is wrong, which part of the pipeline was wrong? Retrieval brought the wrong
passages, the model ignored the right ones, or the verifier flagged something it should not
have. Reading that from a terminal means re-running with print statements. A trace answers it
by construction, and the choice here decides what a bad eval run costs to diagnose.

The second question is narrower and it drove the decision: **can an eval score be attached to
the trace that produced it?** This project already scores every answer for faithfulness,
correctness and grounding. A score sitting in a JSON report and a trace sitting in a UI are
two artefacts nobody joins by hand.

## Candidates

| Candidate | One-line description | License | Runs locally? |
|---|---|---|---|
| Langfuse 4.30.0 | LLM engineering platform: traces, scores, prompts, evals | MIT (core) | yes, docker compose, 6 services |
| Arize Phoenix | OpenTelemetry-native LLM tracing and evaluation UI | ELv2 | yes, one process |
| MLflow tracing | Tracing added to the experiment tracker most teams already run | Apache 2.0 | yes, one process |

## Criteria

| Criterion | How measured | Weight |
|---|---|---|
| Scores on traces | Whether an eval verdict can be attached to the trace it came from | high |
| Cost of running it | Containers or processes, idle memory, image or install size | high |
| Time to a working stack | Cold start to the port answering, packages already cached | medium |
| Cost per question | `rag ask` wall clock with tracing on and off, same question | medium |
| Local and free | No account, no key, no outbound call (ADR-003) | required |

## Results

Measured on this machine, 2026-09-05, OrbStack 29.4.0 with 11.7 GB available to containers.

| Candidate | Processes | Idle memory | Cold start | On-disk |
|---|---|---|---|---|
| Langfuse 4.30.0 | 6 containers | 2,579 MB | about 2 min on first boot, then seconds | 5.6 GB of images |
| Arize Phoenix | 3 processes | 453 MB | 6.2 s | one Python environment |
| MLflow | 2 processes | 77 MB | 5.2 s | one Python environment |

Langfuse memory is the sum of `docker stats` at idle: ClickHouse 937 MB, web 817 MB, worker
672 MB, MinIO 82 MB, Postgres 54 MB, Redis 17 MB. ClickHouse and the two Node services are
the whole cost; the datastores around them are rounding.

**Tracing costs about 150 ms per question.** Five runs of the same question each way,
`qwen3.8-27b-64k` answering, medians of the wall clock:

| | Median | Runs |
|---|---|---|
| tracing on | 3.18 s | 4.49, 3.13, 3.18, 3.08, 3.34 |
| tracing off | 3.02 s | 3.02, 3.02, 2.92, 3.15, 3.03 |

The first traced run cost 4.49 s. That is the SDK import and the first connection, paid once
per process, and it is why the SDK is imported inside `build_tracer` rather than at module
scope. A short-lived CLI also has to call `flush()` explicitly, because the SDK batches on a
five-second timer that a process exiting in three seconds never reaches.

**A tracer pointed at nothing used to cost eleven seconds.** With `LANGFUSE_HOST` on a closed
port, `rag ask` still answered, but took 13.9 s against the 3.0 s baseline: the OpenTelemetry
exporter retries with backoff inside that explicit `flush()`. `build_tracer` now probes
`/api/public/health` once with a two-second timeout and falls back to the tracer that does
nothing, which brings the same command back to 3.14 s. Wrong keys were never expensive, 3.89
s, because authentication fails immediately rather than retrying.

| Langfuse state | `rag ask` wall clock | Answer |
|---|---|---|
| healthy | 3.10 s | traced |
| host unreachable, before the probe | 13.91 s | answered, untraced |
| host unreachable, after the probe | 3.14 s | answered, untraced |
| wrong secret key | 3.89 s | answered, untraced |

**What a trace actually shows.** One answered question, `qwen3.8-27b-64k`, seven spans:

| Span | Time |
|---|---|
| `rag ask` (root) | 9,577 ms |
| `retrieval` | 1,147 ms |
| `generation` | 8,423 ms |
| `verification` | 4 ms |
| `scope-gate`, `retrieval-gate`, `answer-gate` | under 1 ms each |

That last row is the argument for deterministic checking in one line. Every citation
resolved, every figure matched against its passage, in four milliseconds against an eight and
a half second model call. A refused question is two spans and one millisecond, because
`scope-gate` rejects it before anything is retrieved or generated.

**Only Langfuse was integrated.** Phoenix and MLflow were measured for footprint and start
time, not wired into the pipeline, so the scores row below is read from their documentation
rather than from this corpus. Saying otherwise would make this page a draft pretending to be
a decision.

| Criterion | Langfuse | Phoenix | MLflow |
|---|---|---|---|
| Scores on traces | measured here: `faithfulness`, `correct`, `fully_grounded` and `refused` written by a real eval run and read back filtered by trace id | annotations and evaluation runs, not measured | evaluation tables, trace-level scoring weaker, not measured |
| Idle memory | 2,579 MB | 453 MB | 77 MB |
| Cold start | ~2 min first boot | 6.2 s | 5.2 s |
| Runtime cost per question | 150 ms | not measured | not measured |

## Decision

**Langfuse**, self-hosted and pinned to 4.30.0, for the scores API. The eval verdicts this
project already produces now land on the traces that produced them: `refused_correctly` and
`reason_matches_label` from the refusal eval, `faithfulness`, `correct` and `fully_grounded`
from the generation eval. Opening a trace and seeing the judge's verdict on it is the thing
being bought, and it is worth six containers here.

**Phoenix wins on any machine that cannot spare 2.6 GB**, and that is not a hypothetical: the
whole point of ADR-003 is that this runs on a laptop. Tracing is therefore optional and off
unless `LANGFUSE_HOST` and both keys are set, the pipeline uses a tracer that does nothing
otherwise, and no test or CI job needs Docker. Swapping in Phoenix means writing one class
against the `Tracer` protocol; nothing outside `observability/tracing.py` knows which backend
is in use.

**MLflow would win if this project already ran MLflow** for experiments. It does not, and
adding an experiment tracker to get tracing is the tail wagging the dog.

## Interview one-liner

Langfuse costs six containers and 2.6 GB to run, and I took that because it lets the eval
scores hang on the trace that produced them, so a bad faithfulness number is one click from
the retrieval that caused it. Tracing is 150 ms a question and off by default, behind a
protocol with a no-op implementation, so the laptop path never pays for it.
