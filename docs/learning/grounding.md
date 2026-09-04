# Grounding

## What it means

An answer is grounded when every claim in it can be traced to a specific passage in the source corpus, and the passage actually supports the claim. Grounding is a property of the *whole pipeline*: the corpus must carry provenance, retrieval must return it, the prompt must expose it, the generator must cite it, and a verifier must check it.

## What this repo does

- **Sections carry offsets into one canonical string**: the parser writes the normalized filing text next to the section records, and every offset (section, chunk, gold evidence span, citation) indexes into that same string (RAG-004).
- **The corpus is reproducible**: `rag ingest download` writes `data/raw/<TICKER>/manifest.json` with accession number, form, period of report, fiscal period label, filing date, source URL, and byte count for every filing on disk, and re-running it is a no-op (RAG-003).
- **Provenance is mandatory** on every `Chunk`: ticker, form type, fiscal period, filing date, SEC section, character offsets, and source URL (RAG-004).
- **Chunk ids are visible to the model**: retrieved chunks are rendered as `[c17] ...text...` and the prompt requires inline citations per sentence (RAG-010).
- **Citations are verified, not trusted**: each citation must resolve to a passage that was actually provided, and numbers in a cited sentence must appear in that passage after unit normalisation (RAG-010). A model that cites `[c9]` when it was given five passages is caught, and `llama3.1:8b` does exactly that in half its answers. Sentences that fail are returned as `unsupported_sentences`, and numbers that are not in the chunk are returned as `derived_numbers`, not silently kept. Derived numbers get calculation provenance in RAG-021: operands cited, operation stated, result recomputed.
- **The UI shows the evidence**: citation, highlighted passage, and a link to the filing on EDGAR (RAG-014).

## Measured

23 `lookup` questions, prompt v1, measured 2026-09-04 by `rag eval generation`. **Gold passages** hands the model the chunks that hold the evidence, which isolates the generator and the verifier. **Retrieved passages** runs the real pipeline, so retrieval's recall is inside the number.

| Model | Passages | Refused | Citations resolve | Every sentence cited | Figures verified | Fully grounded | States the gold figure |
|---|---|---|---|---|---|---|---|
| `qwen3.8-27b-64k` | gold | 0% | 100% | 96% | 91% | 91% | 91% |
| `qwen3.8-27b-64k` | retrieved | 30% | 100% | 100% | 100% | 100% | 75% |
| `gpt-oss:20b` | gold | 4% | 100% | 91% | 100% | 91% | 77% |
| `gpt-oss:20b` | retrieved | 35% | 100% | 87% | 100% | 87% | 67% |
| `llama3.1:8b` | gold | 4% | 50% | 50% | 86% | 41% | 95% |

Run record: commit `0ab6a44`, prompt v1, chunker `fixed`, `context` embed variant, ChromaDB, k=5, `ANSWER_MAX_TOKENS=1024`. Reports under `reports/`.

### What the numbers say

- **Citation discipline is a model capability, not a prompting problem.** The same prompt gets 100% resolvable citations from every model at 20B or above and 50% from the 8B one. `llama3.1:8b` invents passage labels it was never given, in half its answers. It is the better model at finding the right figure (95% against 77%) and the worse one at saying where it found it.
- **Grounding holds when retrieval degrades; correctness does not.** Moving from gold passages to retrieved ones leaves citation resolution at 100% and fully-grounded at 87%, while refusals rise from 4% to 35%. The system does not start making things up when the evidence thins out. It says it cannot answer.
- **Grounding and correctness are separate axes.** Two models ground well and then state the labelled figure only 64% and 77% of the time. `qwen3.8-27b` is the only one measured that does both, and the only one at 100% on every grounding measure end to end. Full table with latency: `docs/tradeoffs/llm-serving.md`.
- **Refusal is doing real work already.** 30% to 35% of end-to-end questions get `INSUFFICIENT_EVIDENCE`, which tracks retrieval's 36% recall@5. Turning that signal into a policy with reasons is RAG-011.

### What the verifier cannot do

It asks whether a figure is **present** in the cited passage, not whether the claim about it is true. Two consequences seen in this run:

- An answer stating "a $15,381 million increase" passes when the passage happens to contain 15,381 anywhere, including as an unrelated line item.
- An answer reading the wrong column states a real number from the right table and passes.

Checking the relationship rather than the presence needs the operands and the operation, which is RAG-021. `derived, unverified` is the honest label until then: the model may have computed correctly, and this verifier cannot tell.

`states the gold figure` is a strict proxy for correctness, not a judge. It requires the same figure the label writes, so a correct answer phrased at a different scale fails it. RAG-012 adds an actual faithfulness judge.

## Talking points

- Why provenance has to be designed in at ingestion (you cannot bolt it on later).
- The difference between "the model produced a citation" and "the citation supports the claim".
- What happens to grounding when chunks are too small (context lost) or too large (citation too coarse to verify).
- Fiscal vs calendar periods as a grounding trap in financial documents.

## Reading

- Anthropic, [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- Min et al. 2023, [FActScore](https://arxiv.org/abs/2305.14251)
- Liu et al. 2023, [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- Full list: README, "Reading and courses".

## Related

RAG-004, RAG-010, RAG-014, RAG-021. See also `hallucination-control.md`.
