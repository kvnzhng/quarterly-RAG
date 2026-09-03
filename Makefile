.PHONY: setup lint fmt test test-all models langfuse-up langfuse-down eval

setup:          ## Create venv, install deps, install git hooks
	uv sync
	uv run pre-commit install

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

models:         ## Pull local models (RAG-002)
	ollama pull $${LLM_MODEL:-llama3.1:8b}
	ollama pull $${EMBED_MODEL:-nomic-embed-text}

langfuse-up:    ## Start self-hosted Langfuse (RAG-013)
	docker compose -f infra/docker-compose.langfuse.yml up -d

langfuse-down:
	docker compose -f infra/docker-compose.langfuse.yml down

eval:           ## Run retrieval + generation evals (RAG-008+)
	uv run rag eval all

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'
