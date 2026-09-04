# Retrieval: dense vs BM25 vs hybrid vs hybrid with reranking

**Status:** decided (ADR-008; measured on the 33-question eval set, 2026-09-04)
**Ticket:** RAG-009

## Question

RAG-008 measured dense retrieval at 36.4% recall@5 and found it answering **zero of seven** quarterly questions. Everything downstream is capped by that: the generator refuses when the evidence is missing (RAG-011), so answer coverage cannot exceed retrieval. What raises it?

## Candidates

| Candidate | What it adds |
|---|---|
| `dense` | Embedding similarity alone, the RAG-006 baseline |
| `bm25` | Okapi BM25 over the chunk plus its provenance header, with the question expanded into the corpus's own spelling of any period or company it names |
| `hybrid` | Reciprocal rank fusion of both, pool 50, fusion constant 60 |
| `hybrid-filter` | Hybrid, plus a ticker filter inferred from the question |
| `hybrid-rerank` | Hybrid, then the configured chat model scores each of 10 candidates 0 to 10 |

## Criteria

| Criterion | How measured | Weight |
|---|---|---|
| recall@5 | The generator sees five passages, so this bounds answer coverage | high |
| recall@1 and MRR | Models attend less reliably to material deep in a prompt | medium |
| 10-Q recall | The failure RAG-008 found; a strategy that does not fix it has not fixed the problem | high |
| Latency | Per question, on the network Ollama server | medium |
| Dependencies | ADR-003 says the defaults fit a laptop | medium |

## Results

33 answerable questions, 1,391 chunks, `context` embed variant, `fixed` chunker.

| Strategy | recall@1 | recall@3 | recall@5 | recall@10 | MRR | 10-Q recall@5 | Seconds per question |
|---|---|---|---|---|---|---|---|
| `dense` | 18.2% | 30.3% | 36.4% | 45.5% | 0.266 | 0.0% | ~0.3 |
| `bm25` | 3.0% | 21.2% | 30.3% | 45.5% | 0.146 | **14.3%** | ~0.1 |
| **`hybrid`** | 21.2% | 33.3% | **45.5%** | 45.5% | 0.300 | 0.0% | ~0.4 |
| `hybrid-filter` | 21.2% | 33.3% | 45.5% | 45.5% | 0.295 | 0.0% | ~0.4 |
| `hybrid-rerank` (8B judge) | 27.3% | 33.3% | 39.4% | 45.5% | 0.325 | 0.0% | 3.0 |
| `hybrid-rerank` (20B judge) | **30.3%** | **36.4%** | 39.4% | 45.5% | **0.348** | 0.0% | 14.0 |

Run record: commit `7a9ee6e`, chunker `fixed` (350 words, 60 overlap), `nomic-embed-text` with task prefixes, `context` embed variant, ChromaDB, pool 50, fusion constant 60, rerank pool 10, relevance = any overlap with a gold span.

### Fusion is the win, and it is not the sum of its parts

Hybrid beats dense by 9 points at k=5 while BM25 alone is 6 points *worse* than dense. Neither retriever is better than the other; they fail on different questions, and reciprocal rank fusion keeps what they agree on. Widening the candidate pool from 20 to 50 moved recall@5 from 39.4% to 45.5%, because a passage both retrievers rank mid-list beats one only the leader ranks highly.

### The inferred filter buys nothing, as predicted

RAG-008's near-miss ladder showed retrieval already reaching the right filing 91% of the time, so there was little for a company filter to remove. It changes recall not at all and costs a little MRR. It stays available and off by default; it would matter on a corpus of fifty companies rather than two.

### Reranking trades recall for precision, and is not worth it here

Both judges move recall@1 up (18.2% to 30.3% with the 20B model) and recall@5 **down** (45.5% to 39.4%). The reranker is good at picking the best of a pool and, working from a pool of ten, it pushes some relevant passages out of the top five. Since the generator receives five passages, recall@5 is the number that bounds answers, so reranking as configured makes the end-to-end system worse while making the top of the list better. It also costs 3 to 14 seconds a question against 0.4.

A dedicated cross-encoder such as `bge-reranker-base` would be faster and probably better, and it would mean sentence-transformers and PyTorch, roughly two gigabytes, against ADR-003's promise that the defaults fit a laptop. These numbers say the gain would have to be large to justify that, and reranking's shape here suggests it would show up at k=1, not k=5. Revisit if the pipeline ever passes fewer passages to the generator.

### BM25 fixes the 10-Q failure and fusion loses it again

BM25 alone answers one of seven quarterly questions; dense and every fusion answer none. The lexical index is doing exactly what it was added for, and RRF dilutes it because the dense ranking outvotes it. That is the honest reading of a 14.3% against 0.0%: on seven questions the difference is one question, and the mechanism is real even if the sample is not.

### The ceiling is chunking, not ranking

Every strategy plateaus at 45.5% by k=10. Pushing hybrid deeper:

| Depth | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| recall | 45.5% | 45.5% | 54.5% | 66.7% | 69.7% |

**Ten of 33 questions have their gold chunk nowhere in the top 100 of 1,391.** No amount of reranking reaches a passage that ranks below the 93rd percentile. Those ten are mostly financial-statement tables: a chunk of row labels and figures shares almost no tokens with a natural-language question and sits far from it in embedding space. That is a chunking problem, and it is the argument RAG-020 exists to test, with parent-child chunking the obvious candidate since it would attach a section's prose to its tables.

## Decision

**`hybrid` is the default** (ADR-008): the best recall@5, which is what bounds the answers, at negligible cost and with no new heavyweight dependency. `hybrid-rerank` stays in the repo for when top-1 matters more than top-5.

## Interview one-liner

Fusing dense and keyword retrieval raised recall@5 from 36% to 46% even though the keyword index alone scored worse than dense, because the two fail on different questions; and pushing the same retriever to depth 100 showed that a third of the questions are unreachable at any rank, which made the next problem a chunking problem rather than a ranking one.
