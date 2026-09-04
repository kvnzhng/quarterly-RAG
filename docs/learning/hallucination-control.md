# Hallucination control

## What it means

A hallucination in RAG is a claim that is not supported by the retrieved context (or by any context). Control means: reduce the rate, detect what remains, and never present an unverified claim as verified.

## Layers of defense in this repo

1. **Retrieval confidence gate** before generation (RAG-011).
2. **Grounded prompt**: only answer from the provided chunks, cite every sentence, say "insufficient evidence" when needed (RAG-010).
3. **Deterministic verification**: citations resolve, numbers match the cited chunk after unit normalisation, and a number that does not match is flagged as derived rather than passed (RAG-010).
4. **Calculation provenance**: a derived number (growth rate, difference, ratio, share) is written as a `CALC:` line with its operands, each carrying the passage it came from, and three things are checked independently: every operand is cited, every operand is printed in the passage it cites, and the arithmetic gives the stated result (RAG-021). A verbatim check alone passes a wrong relationship between two correct numbers.
5. **LLM-as-judge faithfulness**: each claim is checked for entailment against its cited chunk with a local model, compared against RAGAS faithfulness (RAG-012).
6. **Regression gate**: `make eval` fails if any metric drops more than five points below `data/eval/baseline.json`. Five points because 33 questions means one question is three, and a gate that fails on noise is one nobody reads (RAG-012).

## Measured

Judge `qwen3.8-27b-64k`, generator `llama3.1:8b`, gold passages, 23 lookup questions (RAG-012):

| Measure | Value |
|---|---|
| Faithfulness (sentences supported by the passage they cited) | 79% |
| Judged correct | 100% |
| `states the gold figure` proxy | 91.3% |

**Calculation lines move the calibration, for a reason worth knowing.** The judge is shown the prose without them, while the verifier now passes a sentence whose derived figure a calculation accounts for. On the derived questions the two therefore disagree more, not less: `qwen3.8-27b-64k` judged by `gpt-oss:20b` agreed with the verifier on 8 of 8 sentences under v1 and 8 of 13 under v2, every disagreement being the judge stricter. The judge is reading a claim whose support is on a line it was not given.

**The judge is calibrated before it is believed.** Against the deterministic number verifier over 57 cited sentences: 86% agreement, 6 sentences where the judge was stricter than the verifier, and **2 where it was looser**. That second number is the one that matters: of eight sentences whose figures were not in the cited passage, the judge waved two through. A judge that misses a quarter of the cases it exists to catch is reported with that rate attached, never without.

**The hallucination it does catch is the one the verifier cannot.** An answer claiming "a $15,381 million increase" passes a presence check because 15,381 appears somewhere in the income statement. The judge marks it unsupported. That is the whole reason both layers exist.

**RAGAS was measured and rejected**: with local models it scored faithful answers 0.0 and an unfaithful derived claim 1.0, anti-correlated with the truth. Full comparison and the dependency cost: `docs/tradeoffs/evaluation.md`.

## Measured: calculation provenance (RAG-021)

Gold passages, the 5 `derived` and 5 `cross_period` questions, k=5, `ANSWER_MAX_TOKENS` 1024, commit `fdbb7ef`, corpus `ab54dafa27ee5fe1`, eval set `57ba5e0dfdb94790`, 2026-09-04. Prompt v1 forbids arithmetic; prompt v2 allows it on condition the answer writes a `CALC:` line. The judge is a different model from the generator in both rows. Counts, not rates: ten questions cannot carry a percentage. The run records for all but the first show a dirty tree; only markdown was uncommitted, and the code was `fdbb7ef` throughout.

| Generator (judge) | Prompt | Answered | Calculations verified | Derived figures verified | Every figure accounted for | Judged correct |
|---|---|---|---|---|---|---|
| `qwen3.8-27b-64k` (`gpt-oss:20b`) | v1 | 6/10 | none written | 0 of 0 | 6/6 | 4/6 |
| `qwen3.8-27b-64k` (`gpt-oss:20b`) | v2 | 10/10 | 7 of 8 | 5 of 7 | 8/10 | 10/10 |
| `llama3.1:8b` (`qwen3.8-27b-64k`) | v1 | 9/10 | none written | 0 of 7 | 4/9 | 8/9 |
| `llama3.1:8b` (`qwen3.8-27b-64k`) | v2 | 10/10 | 6 of 10 | 2 of 10 | 6/10 | 8/10 |

**Before, the verified rate is zero by construction, and that is the point.** Prompt v1 tells the model not to compute a figure no passage prints, and a presence check cannot confirm one anyway. The two models obeyed that rule differently. `qwen3.8-27b-64k` refused 4 of the 5 `derived` questions outright, writing `INSUFFICIENT_EVIDENCE` for every growth rate and share; on q033 it answered the two balance-sheet figures and added "the passages do not state the computed change between these two periods". `llama3.1:8b` answered anyway and stated 7 figures that none of its cited passages contain, every one of them unverifiable. A refusal and a hallucination are both what "no calculation provenance" looks like, at the two ends of instruction-following.

**After, the arithmetic is checked rather than trusted.** Both models answer all ten. For `qwen3.8-27b-64k` every `derived` question is now answered, every figure in those five answers is either printed in a cited passage or recomputed from ones that are, and all ten answers are judged correct. That is the ticket's result: the questions that used to be refused are answered, and the numbers in them carry their working.

**The 8B model shows its arithmetic and cites it to the wrong passage.** Only 6 of its 10 calculations verify, and all 4 failures are the same one: an operand that is not in the passage it cites. On q007 it wrote `CALC: (96,221 [c1] - 46,743 [c1]) / 46,743 [c1] * 100 = 106%`. Both figures are real and both are in front of it, but they are in passage c2, which its own sentence cites alongside c1. The judge scored the answer correct. The check refuses it, because an operand attributed to a passage that does not contain it is a citation a reader cannot follow, and following citations is the whole point.

**What `verified` does not mean.** On q030 the same model answered correctly that Apple's operating income was $35,695 million, then wrote `CALC: (35,695 [c1]) / (28,202 [c1]) * 100 = 126%`. The verifier marked it verified, and it is right to: both operands are printed in the passage, and the division is correct. It also answers nothing anybody asked. Verified means the arithmetic is sound over figures the passage states, not that those figures answer the question. Whether they do is what the judge's correctness score measures, which is why the two are reported side by side and never merged.

**The case this check exists for** appeared under the earlier v2 wording, in report `generation-gold-20260904T184449`: `llama3.1:8b` answered q031 with `CALC: 12,914 [c2] - 7,331 [c2] = 5,583 million`. The arithmetic is internally consistent, the answer is right, and the judge scored it correct, because Nvidia's research and development spending did grow by $5,583 million. But 7,331 appears in no passage; the real operands, 12,914 and 18,497, are both printed in the passage it cited. A correct answer reached through an invented figure is what a presence check passes and a judge passes. Under the shipped wording the same model got q031 right, so this one is quoted from the run that produced it and not from the table above.

**The wording of the same rule moved the numbers more than the rule did.** An earlier v2 (replaced in `fdbb7ef`) ended with a worked example: an answer sentence followed by its `CALC:` line. With it, `llama3.1:8b` verified 9 of 13 calculations and accounted for 9 of 10 questions, far better than the 6 of 10 above, and `gpt-oss:20b` lost 11 points of faithfulness on the 23 `lookup` questions, 75% to 64%, writing terser paraphrases in the example's style instead of the filing's words. Moving the example into the body of the explanation restored lookup faithfulness to 73% and cost the 8B model most of its arithmetic. Small models need the example; the example is what pulls every model's prose toward it.

On the `lookup` questions the shipped v2 costs nothing measurable. `gpt-oss:20b` through the real pipeline, 23 `lookup` questions, same judge, same day:

| Prompt | Refused | Fully grounded | Faithfulness | Judged correct | Gold figure present |
|---|---|---|---|---|---|
| v1 | 7/23 | 14 of 16 answered | 75% | 15/16 | 12/16 |
| v2 | 8/23 | 13 of 15 answered | 73% | 15/15 | 12/15 |

Faithfulness is 16 judged sentences here, so one sentence is worth six points and the gate's five-point tolerance sits below the metric's own granularity. That is worth knowing before reading a two-point move as a change.

**The gate turned calculation provenance off by default, and that is the result of this ticket as much as the arithmetic is.** `make eval` under v2 failed on `answerable_coverage`, 0.667 to 0.606. Run again it gave the identical number, so it was not noise; run with `ANSWER_PROMPT_VERSION=1` it reproduced all nine committed metrics to three decimals and passed. Prompt v2 costs two of the 33 answerable questions: telling the model it may compute a figure, and in the same breath that it may not otherwise state one, makes it refuse slightly more often. So v1 stays the default and v2 is opt-in, and a question whose answer no filing prints needs `ANSWER_PROMPT_VERSION=2`.

| Gate run | `answerable_coverage` | `abstention_f1` | `faithfulness` | `correct` | Verdict |
|---|---|---|---|---|---|
| v2, first | 0.606 | 0.771 | 0.733 | 1.000 | one metric regressed |
| v2, second | 0.606 | 0.771 | 0.733 | 1.000 | one metric regressed |
| v1 control | 0.667 | 0.812 | 0.750 | 0.938 | no regression |

Both failing runs are recorded here, not only the control that passed. Note also that `rag eval refusal` on its own reported 63.6% under v1 and 66.7% under v2, disagreeing with the same evaluation run inside `rag eval baseline`. Two ways of running one measurement should not differ; which one is right is unresolved, and it is in `docs/notes.md` as an open question.

**A defect in the older verifier, found by measuring this one.** The calculation checker reported "an operand is not in the passage it cites" for q034, whose two operands are both printed in the passage it cites. Apple's operating expenses table ends the research and development row with `$29,915`, and the next line begins `Percentage of total net sales`; the figure pattern allowed any whitespace between a number and its unit, so it crossed the line break and read 29,915 percent. Every answer quoting that figure had been reported as stating a figure the passage does not contain, since RAG-010. Fixed in `242cd5c`, with the opposite mistake, restricting the gap to a space or a tab, fixed in `fdbb7ef`: `gpt-oss:20b` writes a narrow no-break space there and lost its units entirely.

### What this check still cannot do

- **The wrong real figures**, as q030 above. Both operands present, arithmetic sound, question unanswered.
- **A unitless operand does not match a percentage.** Both models wrote q032 as `CALC: 15.6 [c1] - 24.1 [c1]` against a passage printing `15.6%` and `24.1%`, and the check refused it. Their answers were correct and judged correct.
- **A scale constant written into prose is flagged.** `llama3.1:8b` wrote its working into a sentence on q008, "we need to calculate (54,770 / 109,417) * 100 = 50%", and the 100 in that sentence is a figure the passage does not state, so an otherwise clean answer loses its accounted mark for it.
- **A calculation cannot use another calculation's result.** Under the earlier v2 wording (`generation-gold-20260904T184449`) `llama3.1:8b` wrote `CALC: 8.5% [c1] / 24.1% [c1] * 100 = 35.3%` where its own previous line produced the 8.5%, so the operand cites a passage for a figure no passage prints.

The last three are RAG-029. The first is what the judge is for.

**The gate does not cover any of this.** `make eval` scores the 23 `lookup` questions; the 10 that need arithmetic are not in it, so calculation provenance has no regression gate yet.

## Talking points

- Cheap deterministic checks (number matching) catch a surprising share of financial hallucinations.
- Verbatim number checks are not enough for financial QA: the wrong two periods still produce numbers that exist in the source. Recomputing a derived number from cited operands closes part of that gap and not all of it, and the part it leaves open is worth being precise about.
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
