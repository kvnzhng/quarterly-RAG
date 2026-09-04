# Evaluation: a custom judge vs RAGAS

**Status:** decided (measured 2026-09-04)
**Ticket:** RAG-012

## Question

Two things have gone unmeasured. Whether an answer's sentences are actually supported by the passages they cite, beyond the figures the deterministic verifier can check. And whether an answer is correct, which RAG-010 approximated with "does it state the same figure the label states", a proxy that fails a correct answer phrased at a different scale.

RAGAS is the standard answer. The question is whether it works here.

## Candidates

| Candidate | What it does |
|---|---|
| Custom judge (chosen) | Asks a model whether each sentence is supported by **the passage that sentence cited**, and separately whether the answer matches the reference |
| RAGAS `faithfulness` | Decomposes an answer into statements and checks each against the retrieved context as a whole |

The difference is not incidental. RAGAS scores a claim against everything retrieved; this judge scores it against the citation. An answer stating a true fact while citing the wrong passage is faithful to RAGAS and unfaithful here, and citing the wrong passage is precisely the failure a reader cannot check for themselves.

## Criteria

| Criterion | How measured | Weight |
|---|---|---|
| Discriminates | Does it separate a faithful answer from an unfaithful one on this corpus? | decisive |
| Calibration against ground truth | Agreement with the deterministic number verifier, and which way it errs | high |
| Runs locally and free | Project principle: CI never calls a model, defaults cost nothing | high |
| Dependency weight | Packages added | medium |

## Results

### RAGAS does not discriminate with local models

Four hand-built cases against one passage reading `(In millions) / Total net sales | 109,417 | 94,036`:

| Answer | Should be | RAGAS with `gpt-oss:20b` | RAGAS with `qwen3.8-27b` |
|---|---|---|---|
| "Total net sales were 109,417." | faithful | **0.0** | **0.0** |
| "Apple's total net sales were $109,417 million." | faithful | **0.0** | not run |
| "Apple's total net sales were $200,000 million." | unfaithful | 0.0 | 0.0 |
| "Net sales rose by $15,381 million." | unfaithful | **1.0** | not run |

It scores the faithful answers zero and the unfaithful derived one, which is the dangerous case, at one. On this sample it is anti-correlated with the truth. Nothing here suggests RAGAS is a bad library; it suggests its statement-extraction and entailment prompts are tuned for frontier models and that a pipe-delimited financial table is outside what they handle.

Getting that far took work worth recording. `ragas` pulls 45 packages including LangChain and LangGraph, and the current release cannot be imported alongside `langchain-community` 0.4: it needs a Vertex AI module removed in that version. It runs after pinning `langchain-community<0.4` and adding `pillow`, which the resolver does not pull in.

**RAGAS is not a dependency of this project.** Forty-five packages, a pinned-back LangChain, and a metric that is anti-correlated on the corpus it would be used for is not a trade worth making. Revisit if the pipeline ever points at a frontier model, where RAGAS's prompts were validated.

### The custom judge, and whether it can be trusted

The judge is scored against something that cannot be wrong: the deterministic number verifier, which knows whether a figure appears in the passage a sentence cited. Generator `llama3.1:8b` on gold passages, judge `qwen3.8-27b-64k`, 57 cited sentences:

| | Count |
|---|---|
| Both say supported | 43 |
| Both say not supported | 6 |
| Judge stricter: figures verified, judge objects | 6 |
| **Judge looser: figures not verified, judge says supported** | **2** |
| Agreement | 86% |

The six where the judge is stricter are the safe direction: it read the whole claim while the verifier only checked figures. **The two where it is looser are the dangerous direction**, and at eight unverified sentences that is a 25% miss rate on exactly the case the judge exists to catch. With `gpt-oss:20b` as generator the looser count was zero, but so was the number of unverified sentences, so that run tested nothing.

The judge earns its place by catching what the verifier cannot. RAG-010 documented the limit: an answer claiming "a $15,381 million increase" passes a presence check when the passage happens to contain 15,381 anywhere. The judge marks that sentence not supported. That is the case the whole grounding story turns on, and only the judge sees it.

### Correctness: the judge against the proxy

| Measure | Value |
|---|---|
| `states the gold figure` (proxy, RAG-010) | 91.3% |
| Judged correct | 100% |

The proxy under-reported by two questions, both correct answers written at a different scale. Both are kept in the report so the proxy's error rate stays visible.

Run record: commit `e71c561`, prompt v1, gold passages, k=5, judge prompt v1, section-aware chunks, hybrid retrieval.

## Decision

**A custom judge, cross-model.** The generator and the judge are never the same model by default, because a model grading its own work is a known bias and cheap to avoid here.

**Its numbers are reported with its calibration**, always. A faithfulness score without the looser rate beside it is a claim that a model agreed with itself.

**RAGAS is rejected on measurement**, not on principle, and the measurement is recorded so the decision can be revisited.

### The gate, and why it is not in CI

`make eval` runs the retrieval, generation and refusal evals and compares nine numbers with `data/eval/baseline.json`, which is committed alongside the eval set. The tolerance is five points, because 33 questions means one question is three and a gate that fails on noise is one nobody reads. A metric the baseline names and a run does not produce counts as a failure rather than a skip: dropping the judge would otherwise make a faithfulness regression invisible.

It runs on demand and not on every push. A project principle says CI never calls a model, and the gate calls one for every question; GitHub's runners also cannot reach the endpoint or the corpus. The workflow therefore carries the job behind `workflow_dispatch`, so it can be started by hand on a runner that has both. That is a real limitation rather than a wired-up gate: on this project the regression gate is a local command.

## Interview one-liner

The standard faithfulness library scored our faithful answers zero and our one hallucination one, so we built a judge that checks a sentence against the passage it actually cited, and then calibrated that judge against a deterministic number check to find out it waves through a quarter of the cases it exists to catch.
