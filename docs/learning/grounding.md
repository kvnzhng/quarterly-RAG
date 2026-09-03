# Grounding

## What it means

An answer is grounded when every claim in it can be traced to a specific passage in the source corpus, and the passage actually supports the claim. Grounding is a property of the *whole pipeline*: the corpus must carry provenance, retrieval must return it, the prompt must expose it, the generator must cite it, and a verifier must check it.

## What this repo does

- **Provenance is mandatory** on every `Chunk`: ticker, form type, fiscal period, filing date, SEC section, character offsets, and source URL (RAG-004).
- **Chunk ids are visible to the model**: retrieved chunks are rendered as `[c17] ...text...` and the prompt requires inline citations per sentence (RAG-010).
- **Citations are verified, not trusted**: each citation must resolve to a retrieved chunk, and numbers in a cited sentence must appear in that chunk (RAG-010). Sentences that fail are returned as `unsupported_sentences`, not silently kept.
- **The UI shows the evidence**: citation, highlighted passage, and a link to the filing on EDGAR (RAG-014).

## Measured

_Fill in: citation resolution rate, number-match rate, share of answers with at least one unsupported sentence._

## Talking points

- Why provenance has to be designed in at ingestion (you cannot bolt it on later).
- The difference between "the model produced a citation" and "the citation supports the claim".
- What happens to grounding when chunks are too small (context lost) or too large (citation too coarse to verify).
- Fiscal vs calendar periods as a grounding trap in financial documents.

## Related

RAG-004, RAG-010, RAG-014. See also `hallucination-control.md`.
