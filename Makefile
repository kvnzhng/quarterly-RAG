.PHONY: setup lint fmt test test-all doctor models langfuse-up langfuse-down eval eval-accept

# Ollama helpers talk to the server's HTTP API, so no local `ollama` CLI is needed.
# OLLAMA_HOST wins if set; otherwise it is derived from LLM_BASE_URL in .env (minus /v1).
OLLAMA_HOST ?= $(shell sed -n 's|^LLM_BASE_URL=||p' .env 2>/dev/null | tr -d '"' | sed 's|/v1/*$$||;s|/$$||')
ifeq ($(OLLAMA_HOST),)
OLLAMA_HOST := http://localhost:11434
endif
LLM_MODEL ?= $(shell sed -n 's|^LLM_MODEL=||p' .env 2>/dev/null | tr -d '"')
EMBED_MODEL ?= $(shell sed -n 's|^EMBED_MODEL=||p' .env 2>/dev/null | tr -d '"')

setup:          ## Create venv, install deps, install git hooks (pre-commit + commit-msg)
	uv sync
	uv run pre-commit install --hook-type pre-commit --hook-type commit-msg

lint:           ## Lint and type-ish checks
	uv run ruff check .
	uv run ruff format --check .

fmt:            ## Auto-format
	uv run ruff format .
	uv run ruff check --fix .

test:           ## Unit tests (no Ollama / network)
	uv run pytest

test-all:       ## Unit + integration tests
	uv run pytest -m ""

doctor:         ## Check the configured model endpoints and data dirs
	uv run rag doctor

models:         ## Pull the configured chat + embedding models onto the Ollama at OLLAMA_HOST (from .env by default)
	@for m in $(or $(LLM_MODEL),llama3.1:8b) $(or $(EMBED_MODEL),nomic-embed-text); do \
	  printf 'pulling %s on %s ... ' "$$m" "$(OLLAMA_HOST)"; \
	  curl -sS --fail -X POST "$(OLLAMA_HOST)/api/pull" -d "{\"name\":\"$$m\",\"stream\":false}" || exit 1; echo; \
	done

langfuse-up:    ## Start self-hosted Langfuse (RAG-013)
	docker compose -f infra/docker-compose.langfuse.yml up -d

langfuse-down:
	docker compose -f infra/docker-compose.langfuse.yml down

eval:           ## Retrieval + generation + refusal evals against the committed baseline (~5 min, calls a model)
	uv run rag eval baseline

eval-accept:    ## Overwrite the baseline with the current numbers; the deliberate "I accept this" action
	uv run rag eval baseline --accept

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'
