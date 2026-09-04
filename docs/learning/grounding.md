# Grounding

## What it means

An answer is grounded when every claim in it can be traced to a specific passage in the source corpus, and the passage actually supports the claim. Grounding is a property of the *whole pipeline*: the corpus must carry provenance, retrieval must return it, the prompt must expose it, the generator must cite it, and a verifier must check it.

## What this repo does

- **The corpus is reproducible**: `rag ingest download` writes `data/raw/<TICKER>/manifest.json` with accession number, form, period of report, fiscal period label, filing date, source URL, and byte count for every filing on disk, and re-running it is a no-op (RAG-003).
- **Provenance is mandatory** on every `Chunk`: ticker, form type, fiscal period, filing date, SEC section, character offsets, and source URL (RAG-004).
- **Chunk ids are visible to the model**: retrieved chunks are rendered as `[c17] ...text...` and the prompt requires inline citations per sentence (RAG-010).
- **Citations are verified, not trusted**: each citation must resolve to a retrieved chunk, and numbers in a cited sentence must appear in that chunk after unit normalisation (RAG-010). Sentences that fail are returned as `unsupported_sentences`, and numbers that are not in the chunk are returned as `derived_numbers`, not silently kept. Derived numbers get calculation provenance in RAG-021: operands cited, operation stated, result recomputed.
- **The UI shows the evidence**: citation, highlighted passage, and a link to the filing on EDGAR (RAG-014).

## Measured

_Fill in: citation resolution rate, number-match rate, share of answers with at least one unsupported sentence._

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
