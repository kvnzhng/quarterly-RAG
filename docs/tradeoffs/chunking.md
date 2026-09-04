# Chunking: fixed vs recursive vs section-aware vs parent-child

**Status:** decided (ADR-009; measured on the 33-question eval set, 2026-09-04)
**Ticket:** RAG-005, RAG-020

## Question

What is the unit of retrieval, and therefore the unit of evidence? RAG-009 turned this from a design question into a measured one: every retrieval strategy plateaued at the same recall, and pushing the best one to depth 100 still left ten of 33 questions with their evidence nowhere in the top 100 of 1,391 chunks. No ranking reaches a passage that far down. The ceiling was the chunks themselves.

Looking at those ten showed two boundary problems. A condensed financial statement is row labels and figures, sharing almost no words with a natural-language question. And an answering sentence often sat at the end of a chunk whose opening was about something else, so the chunk's embedding was about the wrong topic.

## Candidates

| Candidate | Idea |
|---|---|
| `fixed` | Pack whole lines to a word target inside one Item; never split a table |
| `recursive` | Split on the largest boundary that fits: lines, then sentences, then words. Knows nothing about SEC structure |
| `section-aware` | Cut at the filing's own sub-headings ("Gross Margin", "Segment Operating Performance"), then pack long blocks with the fixed window |
| `parent-child` | Small children for retrieval, the titled block around them handed to the generator |

## Criteria

| Criterion | How measured | Weight |
|---|---|---|
| recall@1 and MRR | Models attend less reliably to material deep in a prompt, so where the evidence lands matters | high |
| recall@5 | The generator receives five passages | high |
| Answer accuracy | Whether the answer states the labelled figure, end to end | high |
| Evidence integrity | A figure and the header naming its period and unit in the same chunk; no half tables | high |
| Chunk count | Index size and embedding cost | medium |

## Results

Same eval set, same embedding model, same hybrid retriever with inferred filtering. Only the chunker changes.

### Retrieval

| Strategy | Chunks | Median words | recall@1 | recall@3 | recall@5 | recall@10 | recall@20 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|---|---|---|
| `fixed` | 1,391 | 304 | 21.2% | 33.3% | 48.5% | 51.5% | 63.6% | 0.314 | 0.282 |
| `recursive` | 1,213 | 321 | 15.2% | 21.2% | 27.3% | 51.5% | 66.7% | 0.243 | 0.195 |
| **`section-aware`** | 2,891 | 96 | **39.4%** | **45.5%** | 48.5% | **57.6%** | **72.7%** | **0.449** | **0.406** |
| `parent-child` | 4,704 | 69 | 27.3% | 33.3% | 45.5% | 57.6% | 63.6% | 0.348 | 0.286 |

### Shape of the chunks

| Strategy | Median | p90 | Largest | Under 50 words | Holding a table | Holding **half** a table |
|---|---|---|---|---|---|---|
| `fixed` | 304 | 347 | 809 | 78 | 473 | 0 |
| `recursive` | 321 | 400 | 4,582 | 62 | 421 | **221** |
| `section-aware` | 96 | 312 | 809 | 1,148 | 545 | 0 |
| `parent-child` | 69 | 152 | 801 | 1,939 | 611 | 0 |

### End to end

`qwen3.8-27b-64k`, 23 lookup questions, retrieved passages, prompt v1.

| Chunker | Refused | Citations resolve | Fully grounded | States the gold figure |
|---|---|---|---|---|
| `fixed` | 30% | 100% | 100% | 75% |
| **`section-aware`** | 35% | 100% | 100% | **93%** |

Abstention over all 63 questions, same model:

| Chunker | Precision | Recall | F1 | Answerable coverage |
|---|---|---|---|---|
| `fixed` | 67.4% | 96.7% | 0.795 | 57.6% |
| **`section-aware`** | **70.0%** | 93.3% | **0.800** | **63.6%** |

Run record: commit `f2906a1`, `nomic-embed-text` with task prefixes, `context` embed variant, ChromaDB, hybrid retrieval with inferred filtering, pool 50, k=5, relevance = any overlap with a gold span.

## What the numbers say

**Cutting on the document's own headings nearly doubles recall@1** (21.2% to 39.4%) and lifts MRR by 43%. This is the ceiling RAG-009 found, and it moved as soon as the boundaries stopped being arbitrary. A filing puts a short title above each block of narrative, and that line is where a topic actually changes; splitting anywhere else guarantees that some chunks are about two things.

**recall@5 barely moves and everything above it does.** Three strategies tie at 48.5% for k=5 while differing by 18 points at k=1. Chunking is not deciding *whether* the evidence is retrievable so much as *how high* it lands, and where it lands is what the generator reads first.

**The structure-blind baseline is much worse, and cutting tables is why.** `recursive` scores 27.3% at k=5 against 48.5%, and it is the only strategy that leaves chunks holding half a table: 221 of them. A table row ends in a newline and a newline is the first boundary a recursive splitter reaches. This is the strategy behaving as designed, and it prices exactly what document awareness is worth.

**Parent-child is second, not first, and the reason is instructive.** Its children are the smallest and its embeddings the most focused, which should help. It scores below `section-aware` everywhere because a 69-word child cut by word count still crosses topics, while a titled block does not. Retrieving small and returning large is the right instinct; the gain comes from where the cut falls, not from how small the piece is.

**Answer accuracy follows retrieval precision, not retrieval recall.** Moving to `section-aware` left recall@5 unchanged and raised the share of answers stating the labelled figure from 75% to 93%. The generator was already receiving the evidence; it was receiving it fourth or fifth, among four passages about something else.

**More chunks is not a cost worth avoiding here.** `section-aware` doubles the index to 2,891 chunks and adds eight seconds to a full rebuild. 1,148 of them are under 50 words, which is honest: a filing has many short titled blocks, and one of them is often the whole answer.

## Decision

**`section-aware` is the default** (ADR-009). Best on every retrieval measure except recall@5, where it ties; best end to end on answer accuracy, abstention precision and coverage; and it never splits a table.

`parent-child` stays in the repo. Its case is a longer generation prompt, which is a question RAG-012 can answer once faithfulness is judged rather than proxied.

## Interview one-liner

Retrieval had a ceiling no reranker could lift, so the chunk boundaries were the problem; cutting on the filing's own sub-headings instead of on a word count nearly doubled top-one recall and took end-to-end answer accuracy from 75% to 93%, without changing recall at five at all.
