# Notes -- quarterly-RAG

> Claude's scratch pad for cross-session observations. Updated by `/project-update`.

## Observations

- 2026-09-03: Ollama is not installed yet; Docker is installed but was not running at init. Both are needed from RAG-002 / RAG-013 onward.
- 2026-09-04: Models run on Kevin's network Ollama server (v0.32.13; address in `.env` only). It holds an 8B to 32B ladder (llama3.1:8b, gpt-oss:20b, mistral-small, gemma4:26b, qwen3.6:27b, qwen3.8-27b, deepseek-r1:32b), so local comparisons are not laptop-bound. There is no Ollama on the laptop; `make models` pulls over the HTTP API. Warm `rag doctor`: chat 242 ms, embedding 38 ms.
- 2026-09-03: `jq` on PATH comes from Anaconda (`/opt/homebrew/anaconda3/bin/jq`); the ticket hook depends on it.

- 2026-09-03: Kevin runs a local AI server on the network; its address is given at RAG-002 start and lives in `.env` only. Provider choice is the user's (ADR-005).
- 2026-09-04: Corpus v0 on disk: 8 filings per company for the two-year default window (AAPL FY2024 10-K to FY2026 Q3; NVDA FY2025 Q3 to FY2027 Q2), 19 MB of inline-XBRL HTML, 0.7 to 2.1 MB each. EDGAR's `size` field is the whole submission, not the primary document. Two-year default `--since` drifts with the calendar; RAG-019 pins accession numbers.
- 2026-09-04: External review of the scaffold. Accepted: ship one thin end-to-end path before comparing tools; build the eval set right after parsing; label evidence as spans, not chunk ids; flag derived numbers and add calculation provenance; attach a run record to every number; make ticket enforcement portable; fix README tense and stale lines. Backlog reordered into phases, RAG-019 to RAG-023 added.

## Open Questions

- Apple and Nvidia are the starting tickers. Add a third company with a different fiscal calendar (e.g. Microsoft, June FY) to stress period filtering?
- Should the eval sets be committed under `data/eval/` (yes, planned) and should a small sample of processed filings be committed for CI tests (leaning yes, one 10-Q per company)?
