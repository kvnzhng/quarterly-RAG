# Chunking

## What it means

Chunking decides the unit of retrieval and the unit of evidence. Too small and a chunk loses the context needed to answer (a number without its label, a table row without its header). Too large and retrieval gets noisy and citations get vague. The right strategy depends on the document structure and the question types.

## Strategies implemented (RAG-005)

| Strategy | Idea | Expected strength | Expected weakness |
|---|---|---|---|
| Fixed-size tokens | N tokens with overlap | simple, predictable | cuts sentences, tables, and sections |
| Recursive character | split on paragraphs, then sentences, then words | respects natural boundaries | still crosses section boundaries |
| Section-aware | split within SEC Items only, never across | chunks are semantically coherent, section metadata is exact | uneven sizes, long sections still need sub-splitting |
| Parent-child | retrieve small child chunks, return the parent section to the generator | precise retrieval and rich context | more complex, larger prompts |

Tables get special handling: a table is never split, and its header row is repeated in every chunk that contains part of it.

## What this repo does

Chunkers implement one `Chunker` protocol. A harness reports chunk count, size distribution, and how often a chunk crosses a section boundary. The strategies are then scored on the retrieval eval set (RAG-008), so the chunking decision is made on recall@k and answer faithfulness, not on intuition.

## Measured

_Fill in from `docs/tradeoffs/chunking.md`._

## Talking points

- Chunking is a retrieval decision and an evidence decision at the same time.
- Overlap is a band-aid for boundary problems; structure-aware splitting is the real fix when structure exists.
- Why financial tables need different treatment than prose.
- Small-to-big (parent-child) as the usual production answer, and its cost.

## Related

RAG-004, RAG-005, RAG-008.
