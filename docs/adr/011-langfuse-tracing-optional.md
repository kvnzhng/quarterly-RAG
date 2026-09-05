# ADR-011: Langfuse is the tracing backend, and tracing is optional

**Date:** 2026-09-05
**Status:** accepted
**Ticket:** RAG-013

## Context

Every stage of `rag ask` already produces something worth reading: which passages retrieval chose, what the model wrote, which sentences survived verification, which gate refused. None of it was visible after the command exited, so diagnosing a bad answer meant re-running with print statements.

Three self-hosted, freely licensed options were considered: Langfuse, Arize Phoenix, and MLflow tracing. `docs/tradeoffs/observability.md` has the measurements.

The deciding requirement was narrower than "show me a trace". This project scores every answer for faithfulness, correctness and grounding, and those scores lived in JSON reports that nobody joins to a trace by hand.

ADR-003 also stands: the defaults run on a laptop with no paid API and no required infrastructure. Langfuse self-hosted is six containers and 2.6 GB of memory at idle, which is not a laptop default.

## Decision

**Langfuse 4.30.0, self-hosted, is the tracing backend**, chosen for its scores API. Eval verdicts are attached to the traces that produced them: `refused_correctly` and `reason_matches_label` from the refusal eval, `faithfulness`, `correct` and `fully_grounded` from the generation eval.

**Tracing is off unless it is configured.** It turns on only when `LANGFUSE_HOST` and both keys are set. Otherwise the pipeline uses `NullTracer`, whose spans are real objects that do nothing, so the code has no `if tracing:` branches and the untraced path is the one the unit tests exercise. No test and no CI job needs Docker.

**Tracing may never change an answer.** Every call into the SDK is wrapped: a server that is down, keys that are wrong, or a value that will not serialise leave the answer untouched. The cost of that promise is that a broken tracer is quiet, so `rag doctor` checks Langfuse explicitly and is the one place that is loud about it.

**The backend is swappable.** `observability/tracing.py` holds a `Tracer` protocol and takes only strings, numbers and plain dicts, so it imports from no other layer. Nothing outside that module knows which backend is in use; moving to Phoenix means writing one class.

## Consequences

- A trace shows where the time goes. On one answered question: generation 8,423 ms, retrieval 1,147 ms, verification 4 ms. The deterministic checking is free next to the model call, and that is now visible rather than argued.
- Tracing costs about 150 ms per question, plus one-off SDK import on the first call in a process. Measured, in the tradeoff page.
- Running Langfuse costs 6 containers, 2.6 GB of memory and 5.6 GB of images. Anyone who cannot spare that gets the same pipeline with tracing off, or writes a Phoenix tracer.
- Langfuse v4 defaults to `events_only` mode, where the legacy `/api/public/traces` endpoint is gone. Reads go through `/api/public/v2/observations` and the metrics API. The integration test uses the v2 endpoint, and anything written against the old API will need updating.
- The compose file is upstream's, pinned, with a healthcheck and telemetry off, so re-pinning to a later Langfuse is a small diff.

## Alternatives

- **Arize Phoenix**: 453 MB and one command, against Langfuse's 2.6 GB and six containers. The better choice on a constrained machine, and the reason the backend sits behind a protocol. Not chosen because its scoring model was not what this project's evals already produce.
- **MLflow tracing**: 77 MB, the lightest by far, and the obvious answer for a team already running MLflow for experiments. This project does not, and adopting an experiment tracker to get tracing is backwards.
- **No tracing, better logging**: cheapest, and it does not put the judge's verdict next to the retrieval that caused it.
