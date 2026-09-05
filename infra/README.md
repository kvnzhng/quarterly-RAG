# infra

Local infrastructure via Docker. Nothing here is required to answer a question: with Langfuse absent the pipeline uses a tracer that does nothing (ADR-011).

- `docker-compose.langfuse.yml`: self-hosted Langfuse, pinned to 4.30.0. Six services, because Langfuse v4 needs Postgres, ClickHouse, Redis and MinIO alongside its own web and worker. The file is upstream's, with pinned image tags, a healthcheck on the web service and telemetry off; the header lists every change, so re-pinning to a later release is a small diff.
- Data lives in Docker named volumes (`langfuse_postgres_data` and friends), not in this directory.

| Command | What it does |
|---|---|
| `make langfuse-up` | Start it and wait for the health endpoint. First boot runs database migrations and takes a couple of minutes. |
| `make langfuse-down` | Stop it, keeping every trace. |
| `make langfuse-logs` | Follow the web and worker logs. |
| `make langfuse-reset` | Stop it and delete the volumes, losing every trace. |

The API keys come from the `LANGFUSE_INIT_*` values in `.env` and are created on first boot, so there is no clicking through a UI to get a project. `rag doctor` reports whether the server is reachable and whether it accepts the keys.

Costs, measured on 2026-09-05: 6 containers, 2.6 GB of memory at idle, 5.6 GB of images. `docs/tradeoffs/observability.md` compares that with Arize Phoenix and MLflow.
