# Orchestration: plain Python and the alternatives

**Status:** draft comparison; plain Python is implemented, alternatives are not benchmarked
**Tickets:** RAG-010, RAG-049

## Question

Who chooses the next step after retrieval, generation or a failed check? The current
answer is explicit Python control flow in [Pipeline](../../src/quarterly_rag/pipeline.py).
A separate question is whether a library should express that flow. Adopting a framework
does not by itself require model-directed control, and writing Python does not prevent it.

The [architecture learning chapter](../learning/architecture.md) explains the overall
design. This page records its orchestration tradeoff without presenting an unrun
framework comparison as a measured decision.

## Current implementation

Factories build the configured retriever, model and tracer. The pipeline receives
those components, checks scope, retrieves, gates the results, generates, verifies and
applies the final answer gate. Early returns carry named refusal reasons. Verification
is a separate deterministic function, and tracing wraps the stages.

LangChain, LlamaIndex, Haystack and LangGraph were named as alternatives during planning;
none is used for this request path. The selective LangChain use proposed in
[ADR-003](../adr/003-local-first-open-source-stack.md) was not adopted in the current
dependency set. No comparative claim about those libraries' current features or
performance follows from that choice.

## Alternatives and tradeoffs

These are design categories. Selecting a concrete library would require checking its
current documentation and implementing the same task before scoring it.

| Approach | Potential benefit | Cost or limitation | Evidence that would justify it here |
|---|---|---|---|
| Explicit Python workflow, as built | Read every branch directly; inject fakes; keep prompts and check results visible | Own the assembly, adapters, error handling and any future state machinery | Current path is inspectable and exercised by pipeline tests; no comparative superiority established |
| Library-managed fixed pipeline | Reuse integrations and a common component/wiring model | Learn and maintain its conventions; keep provenance and refusal semantics intact through wrappers | Fewer maintained integration responsibilities without worse debugging, behaviour or runtime cost |
| Stateful workflow engine | Model resumable stages, repeated steps and explicit transitions | Persist and version state; decide what can safely be retried | A real pause/resume or recovery requirement the current short request cannot handle cleanly |
| Model-directed retrieval loop | Search again or decompose when the first passages are insufficient | More calls, stopping decisions and possible error propagation | Labelled failures improve under a declared budget, with refusal and groundedness preserved |

The workflow/agent distinction follows
[Anthropic's architecture discussion](https://www.anthropic.com/engineering/building-effective-agents).
These categories can overlap: an engine can run a fixed workflow or a model-directed
loop. Compare control policy and implementation mechanism separately.

## What is established, and what is not

| Evidence available in this repository | What it supports | What it does not establish |
|---|---|---|
| [Pipeline tests](../../tests/test_pipeline.py) with fake components | Stages, early returns and trace boundaries can be checked without live models | Better maintainability than a framework implementation |
| [API tests](../../tests/api/test_app.py) with an injected pipeline | The HTTP layer can exercise its contract independently | Throughput or production capacity |
| [Retrieval](retrieval-strategies.md), [generation](../learning/grounding.md) and [refusal](../learning/refusal.md) experiment records | Behaviour of the components and settings actually tested | A quality advantage caused by plain Python |
| [Observability measurements](observability.md) | Visibility into stage latency and exporter failure cost | Framework overhead, since no framework variant was run |

There are no new performance numbers in RAG-049. This page remains a draft comparison
under the repository's rule that a measured tradeoff needs corpus-specific evidence.

## How to compare fairly

1. Keep the corpus, labels, chunk boundaries, retrieved passages, prompts, provider
   settings and answer budgets fixed for a comparison of orchestration mechanisms.
   Verify that both implementations receive the same inputs and expose the same checks.
2. Compare maintained integration code, the work to change one provider, and the ability
   to diagnose a known failed answer. Record concrete changes and observations instead
   of treating line count alone as maintainability.
3. Measure startup time, request latency and memory alongside answer/refusal outcomes.
   Include a provider failure and a tracing failure so the comparison covers error paths.
4. For a retrieval loop, allow the passages to change: that is its proposed benefit.
   Declare the extra search/model budget, use fresh labelled questions, and report quality,
   coverage, latency and calls together. It is a different experiment from changing the
   wiring library while holding behaviour fixed.

## Current position

Continue with the implemented Python workflow while the next work diagnoses retrieval
misses and broadens the labels. Revisit orchestration when a concrete requirement needs
stateful recovery, repeated search or enough integrations to justify another abstraction.
Record the actual comparison and an ADR before claiming that a replacement wins.

## Talking point

“I kept the control flow explicit so I could see which stage failed and test it with
controlled inputs. That is a design rationale, not a benchmark proving frameworks are
worse. I would compare a framework or retrieval loop against a specific requirement
and preserve the same evidence and refusal contract.”
