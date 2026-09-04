# Embeddings: which model, and what text to embed

**Status:** draft. The v1 embedder is measured and two variants are built (RAG-006); the model comparison is RAG-008.
**Ticket:** RAG-006, RAG-008

## Question

Two questions, not one. Which embedding model, and what text does it actually see? The second is easy to overlook and turned out to matter more than expected on this corpus, because a chunk of a financial table is row labels and figures: the words "Apple" and "third quarter of fiscal 2026" live in the chunk's provenance, not in its text.

## Candidates

| Candidate | Dimensions | Context | License | Runs locally? |
|---|---|---|---|---|
| `nomic-embed-text` (default) | 768 | 8192 tokens | Apache-2.0 | yes, ~274 MB on Ollama |
| `bge-m3` | 1024 | 8192 | MIT | yes, ~1.2 GB |
| `mxbai-embed-large` | 1024 | 512 | Apache-2.0 | yes, ~670 MB |
| `all-MiniLM-L6-v2` | 384 | 256 | Apache-2.0 | yes, via sentence-transformers |
| `text-embedding-3-large` | 3072 | 8191 | hosted, paid | no |

Embed-text variants, both built and both measurable:

| Variant | What the model sees |
|---|---|
| `raw` | the chunk exactly as chunked |
| `context` | a one-line header naming company, form, fiscal period and section, then the chunk |

## Criteria

| Criterion | How measured | Weight |
|---|---|---|
| recall@k | RAG-008 over the RAG-019 spans; a chunk counts when it overlaps a gold span | high |
| Handles tables | recall on the questions whose evidence is a financial table | high |
| Index build time | seconds for the full corpus on the serving machine | medium |
| Dimensions | index size and query cost | medium |
| Context window | must hold a chunk plus its header | low, all candidates clear it |

## Results

### Two findings from RAG-006, measured 2026-09-04

Dense retrieval only, top-k over 33 answerable eval questions, 1,391 chunks from the fixed chunker.

| Query and document text | recall@1 | recall@3 | recall@5 | recall@10 |
|---|---|---|---|---|
| raw chunk, no task prefix | 3.0% | 12.1% | 15.2% | 24.2% |
| raw chunk, nomic task prefixes | 9.1% | 15.2% | 18.2% | 24.2% |
| context header, no task prefix | 9.1% | 21.2% | 24.2% | 36.4% |
| **context header, nomic task prefixes** | **18.2%** | **30.3%** | **36.4%** | **45.5%** |

**The task prefixes were a bug, not a tuning knob.** `nomic-embed-text` is trained with `search_query:` on questions and `search_document:` on passages. Sending neither is a silent misuse: nothing errors, the vectors look fine, and recall@5 is a third lower. The fix lives in the `Embedder` interface, which now has `embed_documents` and `embed_query` rather than one `embed`, so a caller cannot forget which side it is on. The prefixes are settings, because they are specific to this model family.

**The context header roughly doubles recall at every k**, which is the result Anthropic's contextual retrieval work reports, reproduced here on a corpus where the effect has an obvious cause: a table chunk contains no company name and no fiscal period, so a question naming either has nothing to match. Both variants stay in the repo and both are rebuilt by `rag index build [--context]`, so RAG-008 keeps measuring the gap rather than inheriting this one number.

Build cost, full corpus, network Ollama server: 1,391 chunks in 13 seconds per variant, 768 dimensions, vectors already unit-normalised so cosine is the matching metric.

Run record: commit `7d9970b`, `openai_compatible/nomic-embed-text`, chunk strategy `fixed` (350 words, 60 overlap), store `chroma`, k as shown, eval set `data/eval/questions.jsonl` at 43 questions.

### What 36% recall@5 means

Low, and expected at this stage. This is dense retrieval alone against a set built with deliberate traps: two Nvidia revenue figures that differ by 258 million and answer different questions, four questions whose answer is absent from a corpus that discusses the topic, and financial tables that share almost every token with each other. BM25 is the obvious missing piece, because "Q3 FY2026" and "Total net sales" are exact-match terms, and hybrid retrieval plus reranking is RAG-009. The number here is the floor the rest of the retrieval work is measured against.

## Decision

Pending. `nomic-embed-text` with task prefixes and the `context` variant is the working default; the model comparison and the final variant choice are made against RAG-008 numbers and recorded in an ADR then.

## Interview one-liner

The embedding model was being misused in a way that raises no error and costs a third of recall, because nomic's models are trained with query and document prefixes; finding it took an eval set that existed before the index did.
