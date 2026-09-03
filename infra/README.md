# infra

Local infrastructure via Docker. Nothing here is required until RAG-013.

- `docker-compose.langfuse.yml` (added in RAG-013): self-hosted Langfuse. Langfuse v3 needs Postgres, ClickHouse, Redis, and MinIO; the compose file is taken from the official Langfuse repository and pinned to a version.
- `volumes/` is gitignored.

Usage: `make langfuse-up` / `make langfuse-down`.
