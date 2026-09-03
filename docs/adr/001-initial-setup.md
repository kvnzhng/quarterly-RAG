# ADR-001: Initial project setup and tooling

**Date:** 2026-09-03
**Status:** accepted
**Ticket:** RAG-001

## Context

The goal is to learn production RAG by building one and to be able to talk concretely about grounding, chunking, retrieval quality, hallucination control, and refusal. The repo is public-facing, so decisions need to be visible and traceable, not just the code.

## Decision

- Python 3.12 managed by `uv`, `src/` layout, `ruff` for lint and format, `pytest` for tests, GitHub Actions for CI.
- Ticket-based workflow: every change maps to a `RAG-NNN` ticket in `project/tickets.md`, enforced by a Claude Code PreToolUse hook and a git `commit-msg` hook.
- Decisions are recorded as ADRs under `docs/adr/`; tool comparisons live under `docs/tradeoffs/` and must contain measured numbers before they count.
- Learning notes per competency live under `docs/learning/` and double as interview preparation.

## Consequences

- All edits are tracked via tickets; the ticket list is also the project roadmap.
- Structured project management from day one; some ceremony for tiny changes, accepted for a portfolio project.
- Contributors (including Claude) must claim a ticket before editing non-meta files.
