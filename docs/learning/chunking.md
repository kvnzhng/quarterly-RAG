# Chunking

## What it means

Chunking decides the unit of retrieval and the unit of evidence. Too small and a chunk loses the context needed to answer (a number without its label, a table row without its header). Too large and retrieval gets noisy and citations get vague. The right strategy depends on the document structure and the question types.

## Strategies (v1 in RAG-005, the rest compared in RAG-020)

| Strategy | Idea | Expected strength | Expected weakness |
|---|---|---|---|
| Fixed-size tokens | N tokens with overlap | simple, predictable | cuts sentences, tables, and sections |
| Recursive character | split on paragraphs, then sentences, then words | respects natural boundaries | still crosses section boundaries |
| Section-aware | split within SEC Items only, never across | chunks are semantically coherent, section metadata is exact | uneven sizes, long sections still need sub-splitting |
| Parent-child | retrieve small child chunks, return the parent section to the generator | precise retrieval and rich context | more complex, larger prompts |

Tables get special handling: a table is never split, and its header row is repeated in every chunk that contains part of it. The parser (RAG-004) makes this possible by rendering each table as pipe-delimited rows between `[TABLE]` and `[/TABLE]` markers, with the header row labelled.

## What this repo does

Chunkers implement one `Chunker` protocol. The first chunker (RAG-005) is deliberately simple: a fixed window of whole lines inside each parsed section, so the end-to-end path and its baseline numbers exist before any chunking is optimised. RAG-020 adds the other strategies and a harness that reports chunk count, size distribution, and how often a chunk crosses a section boundary. Every strategy is scored on the same eval set (RAG-019 labels, RAG-008 metrics); because the labels are evidence spans rather than chunk ids, re-chunking never invalidates them. The chunking decision is made on recall@k and answer faithfulness, not on intuition.

## Measured

The v1 fixed-window chunker (RAG-005) over the 16-filing corpus, target 350 words, overlap 60:

| Chunks | Median | p90 | Largest | Under 50 words | Over target | Holding a table |
|---|---|---|---|---|---|---|
| 1,391 | 304 | 347 | 809 | 78 | 61 | 473 |

Every chunk resolves back into the filing text it came from, none crosses an Item boundary, and none holds half a table. The 61 oversized chunks are all a single table kept whole. Full numbers and what they say about the corpus: `docs/tradeoffs/chunking.md`. The strategy comparison is RAG-020.

## Talking points

- Chunking is a retrieval decision and an evidence decision at the same time.
- Overlap is a band-aid for boundary problems; structure-aware splitting is the real fix when structure exists.
- Why financial tables need different treatment than prose.
- Small-to-big (parent-child) as the usual production answer, and its cost.
- Chunk sizes here are counted in whitespace words, not model tokens. A word averages 6.4 characters on this corpus and a subword tokenizer splits a figure like `$109,417` into several tokens, so the two numbers are not interchangeable and the code does not pretend otherwise.
- Every chunk's offsets point into the same filing text the gold evidence spans do, so re-chunking re-scores against the same labels instead of invalidating them.

## Reading

- Jimeno Yepes et al. 2024, [Financial Report Chunking for Effective RAG](https://arxiv.org/abs/2402.05131)
- Chroma Research, [Evaluating Chunking Strategies for Retrieval](https://research.trychroma.com/evaluating-chunking)
- Sarthi et al. 2024, [RAPTOR](https://arxiv.org/abs/2401.18059)
- Full list: README, "Reading and courses".

## Related

RAG-004, RAG-005, RAG-019, RAG-008, RAG-020.
