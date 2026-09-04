# Hallucination control

## What it means

A hallucination in RAG is a claim that is not supported by the retrieved context (or by any context). Control means: reduce the rate, detect what remains, and never present an unverified claim as verified.

## Layers of defense in this repo

1. **Retrieval confidence gate** before generation (RAG-011).
2. **Grounded prompt**: only answer from the provided chunks, cite every sentence, say "insufficient evidence" when needed (RAG-010).
3. **Deterministic verification**: citations resolve, numbers match the cited chunk after unit normalisation, and a number that does not match is flagged as derived rather than passed (RAG-010).
4. **Calculation provenance**: a derived number (growth rate, difference, ratio) is emitted with its cited operands and operation, and the verifier recomputes it (RAG-021). A verbatim check alone passes a wrong relationship between two correct numbers.
5. **LLM-as-judge faithfulness**: each claim is checked for entailment against its cited chunk with a local model, compared against RAGAS faithfulness (RAG-012).
6. **Regression gate**: `make eval` fails if faithfulness or abstention metrics drop below the stored baseline (RAG-012).

## Measured

_Fill in: faithfulness score, agreement between the custom judge and RAGAS, examples of caught hallucinations._

## Talking points

- Cheap deterministic checks (number matching) catch a surprising share of financial hallucinations.
- Verbatim number checks are not enough for financial QA: the wrong two periods still produce numbers that exist in the source. Recomputing derived numbers from cited operands closes that gap.
- Judge models hallucinate too: how the judge was validated against hand labels.
- Why lower temperature and "answer only from context" prompts are necessary but not sufficient.
- The interaction with refusal: a stricter verifier pushes more questions into refusal, which is measured in `refusal.md`.

## Reading

- Ji et al. 2023, [Survey of Hallucination in NLG](https://arxiv.org/abs/2202.03629)
- Dhuliawala et al. 2023, [Chain-of-Verification](https://arxiv.org/abs/2309.11495)
- Es et al. 2023, [RAGAS](https://arxiv.org/abs/2309.15217)
- Zheng et al. 2023, [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)
- Full list: README, "Reading and courses".

## Related

RAG-010, RAG-011, RAG-012, RAG-021.
