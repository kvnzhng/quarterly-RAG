# Notes -- rag_project

> Claude's scratch pad for cross-session observations. Updated by `/project-update`.

## Observations

- 2026-09-03: Ollama is not installed yet; Docker is installed but was not running at init. Both are needed from RAG-002 / RAG-013 onward.
- 2026-09-03: `jq` on PATH comes from Anaconda (`/opt/homebrew/anaconda3/bin/jq`); the ticket hook depends on it.

## Open Questions

- Apple and Nvidia are the starting tickers. Add a third company with a different fiscal calendar (e.g. Microsoft, June FY) to stress period filtering?
- Should the eval sets be committed under `data/eval/` (yes, planned) and should a small sample of processed filings be committed for CI tests (leaning yes, one 10-Q per company)?
