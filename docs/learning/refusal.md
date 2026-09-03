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

`data/eval/unanswerable.jsonl` holds questions that must be refused. Together with the answerable set this gives:

- **Abstention precision**: of the refusals, how many should have been refused.
- **Abstention recall**: of the questions that should be refused, how many were.
- **Answer accuracy on the answerable set** at the same thresholds.

Sweeping the confidence threshold produces the curve "refuse too much" vs "hallucinate too much"; the operating point is chosen from that curve.

## Measured

_Fill in: the threshold sweep table and the chosen operating point._

## Talking points

- Refusal reasons need to be distinct because the fix for each is different (index more data vs tune thresholds vs improve the prompt).
- Why "insufficient evidence" from the model is a signal, not a decision; the verifier has the last word.
- How the operating point would differ for an internal analyst tool vs a public-facing product.

## Related

RAG-011, RAG-012. See also `hallucination-control.md`.
