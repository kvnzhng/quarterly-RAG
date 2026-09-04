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

Four strategies over the 16-filing corpus, scored with the same eval labels and the same retriever (RAG-020):

| Strategy | Chunks | Median words | recall@1 | recall@5 | MRR | Half tables |
|---|---|---|---|---|---|---|
| fixed | 1,391 | 304 | 21.2% | 48.5% | 0.314 | 0 |
| recursive | 1,213 | 321 | 15.2% | 27.3% | 0.243 | 221 |
| **section-aware** | 2,891 | 96 | **39.4%** | 48.5% | **0.449** | 0 |
| parent-child | 4,704 | 69 | 27.3% | 45.5% | 0.348 | 0 |

Cutting on the filing's own sub-headings nearly doubles recall@1 and leaves recall@5 unchanged, which says chunking decides *where* the evidence lands rather than *whether* it is found. End to end that mattered more than the recall number suggested: the share of answers stating the labelled figure rose from 75% to 93%.

Every chunk resolves back into the filing text it came from and none crosses an Item boundary. Only the structure-blind recursive splitter holds half a table, 221 times, which is what ignoring the document costs. The 61 oversized chunks are all a single table kept whole. Overlap is applied by whole lines, and filings write long paragraphs as single lines, so a third of chunk boundaries get no overlap at all. Full numbers and what they say about the corpus: `docs/tradeoffs/chunking.md`. The strategy comparison is RAG-020.

## Talking points

- Chunking is a retrieval decision and an evidence decision at the same time.
- Overlap is a band-aid for boundary problems; structure-aware splitting is the real fix when structure exists.
- Why financial tables need different treatment than prose.
- Small-to-big (parent-child) as the usual production answer, and why it came second here: its children are the smallest and its embeddings the most focused, and a 69-word child cut by word count still crosses topics while a titled block does not. The gain comes from where the cut falls, not from how small the piece is.
- Why a ceiling that no reranker could lift turned out to be a chunking problem, and how the eval set made that visible.
- Chunk sizes here are counted in whitespace words, not model tokens. A word averages 6.4 characters on this corpus and a subword tokenizer splits a figure like `$109,417` into several tokens, so the two numbers are not interchangeable and the code does not pretend otherwise.
- Every chunk's offsets point into the same filing text the gold evidence spans do, so re-chunking re-scores against the same labels instead of invalidating them.

## Reading

- Jimeno Yepes et al. 2024, [Financial Report Chunking for Effective RAG](https://arxiv.org/abs/2402.05131)
- Chroma Research, [Evaluating Chunking Strategies for Retrieval](https://research.trychroma.com/evaluating-chunking)
- Sarthi et al. 2024, [RAPTOR](https://arxiv.org/abs/2401.18059)
- Full list: README, "Reading and courses".

## Related

RAG-004, RAG-005, RAG-019, RAG-008, RAG-020.
