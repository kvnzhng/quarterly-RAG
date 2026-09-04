# Chunking: fixed vs recursive vs section-aware vs parent-child

**Status:** draft. The v1 chunker is built and measured (RAG-005); the comparison is RAG-020.
**Ticket:** RAG-005, RAG-020

## Question

What is the unit of retrieval, and therefore the unit of evidence? Too small and a chunk loses the context that makes it answerable, such as a figure without the header naming its period and unit. Too large and retrieval gets noisy and a citation is too coarse to verify. The answer is decided on recall@k and faithfulness (RAG-008, RAG-012), not on intuition.

## Candidates

| Candidate | One-line description | Built? | Ticket |
|---|---|---|---|
| Fixed window (v1) | Pack whole lines to a word target inside one section, overlap by whole lines, never split a table | yes | RAG-005 |
| Recursive character | Split on paragraphs, then sentences, then words | no | RAG-020 |
| Section-aware with sub-splitting | Split on structure inside a section, such as note headings | no | RAG-020 |
| Parent-child | Retrieve small children, hand the parent section to the generator | no | RAG-020 |

## Criteria

| Criterion | How measured | Weight |
|---|---|---|
| recall@k | RAG-008 over the RAG-019 spans; a chunk is relevant when it overlaps a gold span | high |
| Answer faithfulness | RAG-012 judge over answers built from each strategy's chunks | high |
| Evidence integrity | a figure and the header naming its period and unit land in the same chunk | high |
| Citation precision | how much text a citation drags along | medium |
| Chunk count and size spread | index size and embedding cost | medium |

## Results

### v1 fixed window, measured 2026-09-04

`rag chunk build --ticker AAPL --ticker NVDA`, target 350 words, overlap 60, on the 16-filing corpus.

| Ticker | Chunks | Smallest | Median | p90 | Largest | Under 50w | Over target | Holding a table |
|---|---|---|---|---|---|---|---|---|
| AAPL | 511 | 6 | 294 | 345 | 809 | 42 | 16 | 196 |
| NVDA | 880 | 1 | 308 | 348 | 585 | 36 | 45 | 277 |
| Both | 1,391 | 1 | 304 | 347 | 809 | 78 | 61 | 473 |

Invariants checked across all 1,391 chunks: `filing_text[char_start:char_end] == chunk.text` holds everywhere, no chunk crosses a section boundary, no chunk holds half a table, and every chunk id is unique. All 35 gold evidence spans overlap at least one chunk, so none is stranded between chunks.

Run record: commit `754126c`, strategy `fixed`, `CHUNK_WORDS=350`, `CHUNK_OVERLAP_WORDS=60`, corpus manifests `data/raw/{AAPL,NVDA}/manifest.json`, no model involved.

### What the numbers say about the corpus

- **61 chunks exceed the target, all because a table is atomic.** A table runs 84 words at the median and 801 at the largest, so keeping tables whole costs 4% oversized chunks and buys a rule with no exceptions. Splitting them would mean repeating the header row in each part, which breaks the offset invariant and is deferred to RAG-020.
- **78 chunks fall under 50 words**, because a short Item is one short chunk. Apple's "Defaults Upon Senior Securities" is 45 characters. These are honest, not padding.
- **One gold span already straddles two chunks.** Apple's gross margin question (q003) needs a dollar table and a percentage table that sit next to each other. Retrieval must return both, which is precisely the kind of case a parent-child strategy is meant to fix.
- **Nvidia's Part IV Item 15 alone yields roughly 50 chunks**, because it carries the whole set of financial statements. Every one of them is labelled with the same section, so a section filter buys nothing there. Sub-splitting on note headings is the RAG-020 idea this corpus argues for.

## Decision

Pending. `fixed` is the default so the end-to-end path exists and has a baseline; the choice is made in RAG-020 against RAG-008 numbers, and recorded in an ADR then.

## Interview one-liner

The first chunker is deliberately the simplest thing that respects the documents: whole lines up to a word target, never crossing an Item boundary and never cutting a table, with every chunk's offsets pointing into the same text the gold labels do, so changing strategy later re-scores against the same labels rather than invalidating them.
