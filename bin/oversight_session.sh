#!/usr/bin/env bash
# Oversight Session — persistent interactive Claude session that auto-resumes.
#
# Usage:
#   tmux new-session -s oversight './scripts/oversight_session.sh'
#
# When the session hits rate limits or context exhaustion, it waits
# and restarts with --continue to pick up where it left off.
#
# Environment:
#   OVERSIGHT_COOLDOWN_SECS — seconds to wait before resume (default: 120)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COOLDOWN="${OVERSIGHT_COOLDOWN_SECS:-120}"
FIRST_RUN=true

cd "$PROJECT_ROOT"

echo "=== Oversight Session (auto-resume enabled) ==="
echo "  Project:  $PROJECT_ROOT"
echo "  Cooldown: ${COOLDOWN}s between restarts"
echo ""

while true; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    set +e
    if [ "$FIRST_RUN" = true ]; then
        echo "[$TIMESTAMP] Starting fresh session..."
        FIRST_RUN=false
        claude
    else
        echo "[$TIMESTAMP] Resuming previous session with --continue..."
        claude --continue
    fi
    EXIT_CODE=$?
    set -e

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo ""
    echo "[$TIMESTAMP] Session exited (code: $EXIT_CODE)"

    # Exit code 0 = user quit intentionally (Ctrl+C or /exit)
    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "Clean exit detected. Stop auto-resume? (y/n, auto-resumes in ${COOLDOWN}s)"
        read -t "$COOLDOWN" REPLY || REPLY="n"
        if [[ "$REPLY" =~ ^[Yy]$ ]]; then
            echo "Goodbye."
            break
        fi
    else
        echo "[$TIMESTAMP] Non-zero exit (likely rate limit or context exhaustion)"
        echo "[$TIMESTAMP] Waiting ${COOLDOWN}s before resuming..."
        sleep "$COOLDOWN"
    fi
done
