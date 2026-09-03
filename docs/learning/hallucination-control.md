# Hallucination control

## What it means

A hallucination in RAG is a claim that is not supported by the retrieved context (or by any context). Control means: reduce the rate, detect what remains, and never present an unverified claim as verified.

## Layers of defense in this repo

1. **Retrieval confidence gate** before generation (RAG-011).
2. **Grounded prompt**: only answer from the provided chunks, cite every sentence, say "insufficient evidence" when needed (RAG-010).
3. **Deterministic verification**: citations resolve, numbers match the cited chunk verbatim (RAG-010).
4. **LLM-as-judge faithfulness**: each claim is checked for entailment against its cited chunk with a local model, compared against RAGAS faithfulness (RAG-012).
5. **Regression gate**: `make eval` fails if faithfulness or abstention metrics drop below the stored baseline (RAG-012).

## Measured

_Fill in: faithfulness score, agreement between the custom judge and RAGAS, examples of caught hallucinations._

## Talking points

- Cheap deterministic checks (number matching) catch a surprising share of financial hallucinations.
- Judge models hallucinate too: how the judge was validated against hand labels.
- Why lower temperature and "answer only from context" prompts are necessary but not sufficient.
- The interaction with refusal: a stricter verifier pushes more questions into refusal, which is measured in `refusal.md`.

## Related

RAG-010, RAG-011, RAG-012.
