# Notes -- quarterly-RAG

> Claude's scratch pad for cross-session observations. Updated by `/project-update`.

## Observations

- 2026-09-03: Ollama is not installed yet; Docker is installed but was not running at init. Both are needed from RAG-002 / RAG-013 onward.
- 2026-09-03: `jq` on PATH comes from Anaconda (`/opt/homebrew/anaconda3/bin/jq`); the ticket hook depends on it.

- 2026-09-03: Kevin runs a local AI server on the network; its address is given at RAG-002 start and lives in `.env` only. Provider choice is the user's (ADR-005).
- 2026-09-04: External review of the scaffold. Accepted: ship one thin end-to-end path before comparing tools; build the eval set right after parsing; label evidence as spans, not chunk ids; flag derived numbers and add calculation provenance; attach a run record to every number; make ticket enforcement portable; fix README tense and stale lines. Backlog reordered into phases, RAG-019 to RAG-023 added.

## Open Questions

- Apple and Nvidia are the starting tickers. Add a third company with a different fiscal calendar (e.g. Microsoft, June FY) to stress period filtering?
- Should the eval sets be committed under `data/eval/` (yes, planned) and should a small sample of processed filings be committed for CI tests (leaning yes, one 10-Q per company)?
