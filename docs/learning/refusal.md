# When to refuse to answer

## What it means

A production RAG system must be able to say "I cannot answer that from these documents" with a reason, instead of producing a fluent guess. Refusal is a policy with thresholds, and the thresholds are chosen from measured tradeoffs, not picked by hand.

## Refusal reasons (RAG-011)

| Reason | Trigger | Example |
|---|---|---|
| `out_of_scope` | company or period not in the index, non-financial question | "What is Tesla's revenue?" when only AAPL and NVDA are indexed |
| `low_confidence` | best retrieval/rerank score below threshold | vague question with no matching passage |
| `insufficient_evidence` | generator reports the context does not contain the answer | "What will Q4 revenue be?" (forward-looking, not in filings) |
| `verification_failed` | citation or number check fails for the core claim | model produced a number not present in the cited chunk |

A refusal still returns the best chunks found so the user can look for themselves.

## Measuring it

`data/eval/questions.jsonl` holds 30 questions that must be refused and 33 that must be answered, in one file so a single loader and a single hash cover both. Together they give:

- **Abstention precision**: of the questions refused, how many should have been.
- **Abstention recall**: of the questions that should have been refused, how many were.
- **Answerable coverage**: of the questions the corpus can answer, how many the system was willing to attempt.

`rag eval refusal` runs all 63 and sweeps the retrieval threshold.

## Measured

63 questions, 30 unanswerable, k=5, `context` embed variant, measured 2026-09-04.

| Model | Refused | Correct refusals | Precision | Recall | F1 | Answerable coverage | Leaked |
|---|---|---|---|---|---|---|---|
| `qwen3.8-27b-64k` | 43 | 29 | 67.4% | **96.7%** | 0.795 | 57.6% | 1 |
| `gpt-oss:20b` | 42 | 28 | 66.7% | 93.3% | 0.778 | 57.6% | 2 |
| `llama3.1:8b` | 38 | 28 | 73.7% | 93.3% | **0.824** | **69.7%** | 2 |

Which reason fired, with `qwen3.8-27b-64k`:

| Reason | Count |
|---|---|
| `insufficient_evidence` | 29 |
| `out_of_scope` | 13 |
| `verification_failed` | 1 |
| `low_confidence` | 0 |

Run record: commit `76efad5`, prompt v1, chunker `fixed`, `context` variant, ChromaDB, k=5, `MIN_RETRIEVAL_SCORE=0.0`.

### The 14 "wrong" refusals are mostly not wrong

Abstention precision of 67% reads badly until you ask what the 14 over-refusals have in common. Checking whether retrieval had found the evidence for each:

| | Count |
|---|---|
| Refused although the evidence was in the top 5 | 1 |
| Refused because retrieval never found the evidence | 13 |

Only one is a gate failure. The other thirteen are the system correctly declining to answer a question whose evidence it does not have. **Answerable coverage is bounded by retrieval's 36% recall@5, not by the gate.** Improving retrieval, not loosening the gate, is what raises it.

### The threshold sweep says the threshold is not the lever

Sweeping `MIN_RETRIEVAL_SCORE` with `qwen3.8-27b-64k`:

| Min score | Refused | Precision | Recall | F1 | Answerable coverage |
|---|---|---|---|---|---|
| **0.00 (off)** | 43 | **67.4%** | 96.7% | **0.794** | **57.6%** |
| 0.75 | 43 | 67.4% | 96.7% | 0.794 | 57.6% |
| 0.78 | 45 | 64.4% | 96.7% | 0.773 | 51.5% |
| 0.80 | 49 | 59.2% | 96.7% | 0.734 | 39.4% |
| 0.85 | 59 | 50.8% | 100.0% | 0.674 | 12.1% |
| 0.90 | 63 | 47.6% | 100.0% | 0.645 | 0.0% |

The curve only goes down. Raising the threshold buys 3.3 points of recall and costs 45 points of coverage, because cosine scores on this corpus cluster tightly between about 0.74 and 0.84 and do not separate a good match from a bad one. **The operating point is the threshold switched off**, and the generator's own reading of the passages is what does the work: 29 of 43 refusals came from it.

That is a real finding rather than a shrug. A confidence threshold is the standard first answer to abstention, and on this corpus it is worse than useless. A calibrated signal would have to come from somewhere else, such as a reranker score or the model's own token probabilities.

### Refusal calibration is not the same as answer quality

`llama3.1:8b` has the best abstention F1 and the best coverage: it refuses least and is right more often when it does. It is also the model that invents citations in half its answers (`docs/tradeoffs/llm-serving.md`). The same willingness to assert produces both numbers. Coverage is only worth having if the answers behind it hold up, so the model to prefer is `qwen3.8-27b-64k`, which leaks one unanswerable question rather than two and grounds every sentence it writes.

### What still leaks

`q052` leaks past every model: *which customers accounted for Nvidia's largest direct sales?* The filings discuss customer concentration at length and give percentages without naming anyone, so retrieval returns confident, on-topic passages and the generator answers from them. `q057`, asking who Apple's chief financial officer is, leaks past the two weaker models for the same reason. Both are the hard case this eval set was built to contain: the topic is present and the fact is not.

## Talking points

- Refusal reasons need to be distinct because the fix for each is different (index more data vs tune thresholds vs improve the prompt).
- Why "insufficient evidence" from the model is a signal, not a decision; the verifier has the last word.
- How the operating point would differ for an internal analyst tool vs a public-facing product.
- Why a retrieval-confidence threshold failed here, and what a calibrated signal would need to look like.
- The difference between an over-refusal caused by the gate and one caused by retrieval, and why only the first is a bug.

## Reading

- Rajpurkar et al. 2018, [Know What You Don't Know (SQuAD 2.0)](https://arxiv.org/abs/1806.03822)
- Kamath et al. 2020, [Selective Question Answering under Domain Shift](https://arxiv.org/abs/2006.09462)
- Kadavath et al. 2022, [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221)
- Full list: README, "Reading and courses".

## Related

RAG-011, RAG-012. See also `hallucination-control.md`.
