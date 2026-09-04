# Hallucination control

## What it means

A hallucination in RAG is a claim that is not supported by the retrieved context (or by any context). Control means: reduce the rate, detect what remains, and never present an unverified claim as verified.

## Layers of defense in this repo

1. **Retrieval confidence gate** before generation (RAG-011).
2. **Grounded prompt**: only answer from the provided chunks, cite every sentence, say "insufficient evidence" when needed (RAG-010).
3. **Deterministic verification**: citations resolve, numbers match the cited chunk after unit normalisation, and a number that does not match is flagged as derived rather than passed (RAG-010).
4. **Calculation provenance**: a derived number (growth rate, difference, ratio) is emitted with its cited operands and operation, and the verifier recomputes it (RAG-021). A verbatim check alone passes a wrong relationship between two correct numbers.
5. **LLM-as-judge faithfulness**: each claim is checked for entailment against its cited chunk with a local model, compared against RAGAS faithfulness (RAG-012).
6. **Regression gate**: `make eval` fails if any metric drops more than five points below `data/eval/baseline.json`. Five points because 33 questions means one question is three, and a gate that fails on noise is one nobody reads (RAG-012).

## Measured

Judge `qwen3.8-27b-64k`, generator `llama3.1:8b`, gold passages, 23 lookup questions (RAG-012):

| Measure | Value |
|---|---|
| Faithfulness (sentences supported by the passage they cited) | 79% |
| Judged correct | 100% |
| `states the gold figure` proxy | 91.3% |

**The judge is calibrated before it is believed.** Against the deterministic number verifier over 57 cited sentences: 86% agreement, 6 sentences where the judge was stricter than the verifier, and **2 where it was looser**. That second number is the one that matters: of eight sentences whose figures were not in the cited passage, the judge waved two through. A judge that misses a quarter of the cases it exists to catch is reported with that rate attached, never without.

**The hallucination it does catch is the one the verifier cannot.** An answer claiming "a $15,381 million increase" passes a presence check because 15,381 appears somewhere in the income statement. The judge marks it unsupported. That is the whole reason both layers exist.

**RAGAS was measured and rejected**: with local models it scored faithful answers 0.0 and an unfaithful derived claim 1.0, anti-correlated with the truth. Full comparison and the dependency cost: `docs/tradeoffs/evaluation.md`.

## Talking points

- Cheap deterministic checks (number matching) catch a surprising share of financial hallucinations.
- Verbatim number checks are not enough for financial QA: the wrong two periods still produce numbers that exist in the source. Recomputing derived numbers from cited operands closes that gap.
- Judge models hallucinate too: this one was validated against a deterministic number check rather than hand labels, which is cheaper and gives a partial ground truth for free.
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
