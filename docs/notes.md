# Notes -- quarterly-RAG

> Claude's scratch pad for cross-session observations. Updated by `/project-update`.

## Observations

- 2026-09-03: Ollama is not installed yet; Docker is installed but was not running at init. Both are needed from RAG-002 / RAG-013 onward.
- 2026-09-04: Models run on Kevin's network Ollama server (v0.32.13; address in `.env` only). It holds an 8B to 32B ladder (llama3.1:8b, gpt-oss:20b, mistral-small, gemma4:26b, qwen3.6:27b, qwen3.8-27b, deepseek-r1:32b), so local comparisons are not laptop-bound. There is no Ollama on the laptop; `make models` pulls over the HTTP API. Warm `rag doctor`: chat 242 ms, embedding 38 ms.
- 2026-09-03: `jq` on PATH comes from Anaconda (`/opt/homebrew/anaconda3/bin/jq`); the ticket hook depends on it.

- 2026-09-03: Kevin runs a local AI server on the network; its address is given at RAG-002 start and lives in `.env` only. Provider choice is the user's (ADR-005).
- 2026-09-04: Eval set v0 (RAG-019) is 43 human-verified questions over 6 filings. Deliberate near-misses the corpus does *not* answer: iPhone unit sales (Apple stopped disclosing in 2018), Vision Pro revenue (inside Wearables), per-country headcount, executive compensation (proxy, incorporated by reference). Nvidia's segment revenue ($193,479M Compute & Networking) and Data Center end-market revenue ($193,737M) are close but different cuts, which is a good retrieval trap.
- 2026-09-04: Phase two closed except RAG-021. Findings in order of discovery: nomic prefixes were missing and cost a third of recall (RAG-006); the 8B model invents citations in half its answers and citation discipline is a model capability (RAG-010); the retrieval-score threshold is worse than useless as a refusal signal on this corpus (RAG-011); fusion beats both of its parts (RAG-009); the ticker filter buys nothing and the quarter filter fixes the 10-Q questions (RAG-026); chunking on the filing's own sub-headings nearly doubles recall@1 (RAG-020); FAISS is 16x faster and it is 3% of a retrieval (RAG-007); RAGAS is anti-correlated with local models (RAG-012). Remaining ceiling: recall@20 is 72.7%.
- 2026-09-04: Process failures this session: two commits whose messages described doc edits that had not applied, one ticket citing a discarded hash, one conclusion generalised beyond its experiment, one model measured at an unfair token budget. All caught and corrected; the rules are now in `project/conventions.md`.
- 2026-09-04: Corpus v0 on disk: 8 filings per company for the two-year default window (AAPL FY2024 10-K to FY2026 Q3; NVDA FY2025 Q3 to FY2027 Q2), 19 MB of inline-XBRL HTML, 0.7 to 2.1 MB each. EDGAR's `size` field is the whole submission, not the primary document. Two-year default `--since` drifts with the calendar; RAG-019 pins accession numbers.
- 2026-09-04: External review of the scaffold. Accepted: ship one thin end-to-end path before comparing tools; build the eval set right after parsing; label evidence as spans, not chunk ids; flag derived numbers and add calculation provenance; attach a run record to every number; make ticket enforcement portable; fix README tense and stale lines. Backlog reordered into phases, RAG-019 to RAG-023 added.

## Open Questions

- Apple and Nvidia are the starting tickers. Add a third company with a different fiscal calendar (e.g. Microsoft, June FY) to stress period filtering? Now more interesting: the quarter filter (RAG-026) and the scope check (RAG-011) both key on company words, and the near-miss ladder says filing-level retrieval is the solved part.
- The eval set concentrates its evidence in 6 of 16 filings and is 70% tables. Both were noted at RAG-019 and neither has been fixed. A second labelling round spread across quarters would sharpen every retrieval number.
- Recall@20 is 72.7%: roughly a quarter of questions have evidence neither ranking nor chunking reaches. Nobody has looked at which questions those are since RAG-020 changed the chunks.
- `q052` (which customers account for Nvidia's largest sales) leaks past every model's refusal. Is that a prompt problem or a genuinely unanswerable-looking-answerable question?
- ~~Should a sample of processed filings be committed for CI tests?~~ Resolved 2026-09-04 (RAG-004): no. Two truncated HTML fixtures of about 2 KB each under `tests/ingestion/fixtures/` cover the parser's rules; the real 16 filings are exercised by an integration test that skips when the corpus is absent. Eval sets are still committed under `data/eval/`.
