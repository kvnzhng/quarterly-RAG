#!/usr/bin/env bash
# enforce-ticket.sh -- PreToolUse hook for Edit/Write
# Blocks edits to non-meta files when no active ticket is set.
# Installed by /project-init into .claude/hooks/

# Read hook input from stdin
INPUT=$(cat)

# Extract the file path being edited
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.filePath // ""')

# If we can't determine the file path, allow
if [ -z "$FILE_PATH" ]; then
  exit 0
fi

# Meta files are always allowed (no ticket required)
case "$FILE_PATH" in
  */project/tickets.md) exit 0 ;;
  */CLAUDE.md) exit 0 ;;
  */project/conventions.md) exit 0 ;;
  */docs/notes.md) exit 0 ;;
  */docs/adr/*) exit 0 ;;
  */.claude/active-ticket) exit 0 ;;
  */.claude/settings.json) exit 0 ;;
  */.claude/settings.local.json) exit 0 ;;
  */.gitignore) exit 0 ;;
esac

# Find the project root by looking for project/tickets.md
# Walk up from the file being edited
DIR=$(dirname "$FILE_PATH")
PROJECT_ROOT=""
while [ "$DIR" != "/" ] && [ "$DIR" != "." ]; do
  if [ -f "$DIR/project/tickets.md" ]; then
    PROJECT_ROOT="$DIR"
    break
  fi
  DIR=$(dirname "$DIR")
done

# If no project/tickets.md found, this isn't a project-init'd project -- allow
if [ -z "$PROJECT_ROOT" ]; then
  exit 0
fi

# Check if active-ticket file exists and is non-empty
ACTIVE_TICKET_FILE="$PROJECT_ROOT/.claude/active-ticket"
if [ -f "$ACTIVE_TICKET_FILE" ]; then
  TICKET_ID=$(cat "$ACTIVE_TICKET_FILE" | tr -d '[:space:]')
  if [ -n "$TICKET_ID" ]; then
    # Active ticket exists, allow the edit
    exit 0
  fi
fi

# No active ticket -- block the edit
cat <<'HOOK_OUTPUT'
{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"No active ticket. Before editing code:\n1. Find or create a ticket in project/tickets.md\n2. Move it to In Progress\n3. Run: echo \"PREFIX-NNN\" > .claude/active-ticket\nMeta files (tickets.md, CLAUDE.md, conventions.md, notes.md, ADRs) are exempt."}}
HOOK_OUTPUT
exit 0
