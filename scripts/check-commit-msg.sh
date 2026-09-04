#!/usr/bin/env bash
# check-commit-msg.sh -- every commit message must reference a ticket (RAG-NNN).
#
# Usage:
#   scripts/check-commit-msg.sh <message-file>   # commit-msg hook (pre-commit installs it, see Makefile)
#   scripts/check-commit-msg.sh -                 # message on stdin
#   scripts/check-commit-msg.sh --range A..B      # every commit in a git range (CI)
set -euo pipefail

PREFIX="RAG"

check_message() {
  local msg="$1"
  # Merge and revert commits are exempt.
  if printf '%s\n' "$msg" | head -n 1 | grep -qE '^(Merge|Revert) '; then
    return 0
  fi
  # Comments and trailers do not count; the ticket id has to be in the message itself.
  local body
  body=$(printf '%s\n' "$msg" | grep -vE '^#' | grep -vE '^(Co-Authored-By|Claude-Session|Signed-off-by):' || true)
  if printf '%s\n' "$body" | grep -qE "${PREFIX}-[0-9]+"; then
    return 0
  fi
  cat >&2 <<MSG

ERROR: commit message must contain a ticket id (${PREFIX}-NNN)

  Format:  type(scope): description (${PREFIX}-NNN)
  Example: feat(ingestion): download 10-Q filings from EDGAR (${PREFIX}-003)

Your message was:
$(printf '%s\n' "$msg" | sed 's/^/  /')

MSG
  return 1
}

case "${1:-}" in
  --range)
    range="${2:?usage: $0 --range A..B}"
    status=0
    for sha in $(git rev-list --no-merges "$range"); do
      if ! check_message "$(git log -1 --format=%B "$sha")"; then
        echo "  in commit ${sha}" >&2
        status=1
      fi
    done
    exit "$status"
    ;;
  -)
    check_message "$(cat)"
    ;;
  "")
    echo "usage: $0 <message-file> | - | --range A..B" >&2
    exit 2
    ;;
  *)
    check_message "$(cat "$1")"
    ;;
esac
