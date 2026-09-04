# ADR-009: Section-aware chunking is the default

**Date:** 2026-09-04
**Status:** accepted
**Ticket:** RAG-020

## Context

RAG-009 measured a ceiling that ranking could not lift. Every retrieval strategy plateaued at the same recall by depth 10, and the best of them reached only 69.7% at depth 100, leaving ten of 33 questions with their evidence nowhere in the top 100 of 1,391 chunks. Inspecting those chunks showed two boundary problems: a condensed financial statement shares almost no words with a natural-language question, and an answering sentence frequently sat at the end of a chunk whose opening was about something else.

## Decision

- **`section-aware` replaces `fixed` as the default chunker.** It cuts at the filing's own sub-headings and packs a block with the fixed window only when the block exceeds the word target.
- **A heading is a short title line**: one to eight words, not a table row, not ending in sentence punctuation, starting with a capital, and never inside a rendered table. Deliberately strict, because a false heading fragments a section and fragmentation is what this exists to avoid.
- **`parent-child` stays implemented and available.** It scores second and its case is a longer generation prompt, which RAG-012 can judge properly.
- **`recursive` stays as the structure-blind baseline**, including the fact that it cuts tables. That is what it costs to ignore the document, and the comparison needs it priced.
- **The `Chunk` model gains an optional parent span.** Relevance is measured against a chunk's effective span, so a parent-child strategy is judged on the passage the generator would receive rather than on the smaller one embedded to find it. Every other strategy is unaffected, because a chunk without a parent stands for itself.

## Consequences

- The index roughly doubles, from 1,391 chunks to 2,891, and a full rebuild takes eight seconds longer. 1,148 chunks fall under 50 words, which is a filing having many short titled blocks rather than a defect.
- Heading detection is tuned to two filers, like the parser's Item detection before it (ADR-007). A filer who writes no sub-headings degrades to the fixed window, which is the correct fallback rather than a failure.
- Retrieval precision improved far more than retrieval recall, and answer accuracy followed precision: end to end, the share of answers stating the labelled figure rose from 75% to 93% while recall@5 did not move. That is worth remembering when reading any single retrieval number.
- The remaining ceiling is still real: recall@20 is 72.7%, so roughly a quarter of questions have evidence that neither ranking nor chunking currently reaches.
