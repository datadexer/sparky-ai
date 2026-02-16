#!/usr/bin/env bash
# CEO Watchdog — keeps the CEO agent running across rate limits and context resets.
#
# Usage:
#   tmux new-session -d -s ceo './scripts/ceo_watchdog.sh'
#   tmux attach -t ceo          # to watch
#   tmux kill-session -t ceo    # to stop
#
# Environment:
#   CEO_COOLDOWN_SECS   — base cooldown between sessions (default: 300)
#   CEO_MAX_BUDGET_USD  — max spend per session (default: 15.00)
#   CEO_LOG_DIR         — log directory (default: logs/ceo_sessions)
#   CEO_MODEL           — model to use (default: sonnet)
#   CEO_PERMISSION_MODE — permission mode (default: bypassPermissions)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_COOLDOWN="${CEO_COOLDOWN_SECS:-300}"
MAX_BUDGET="${CEO_MAX_BUDGET_USD:-15.00}"
LOG_DIR="${CEO_LOG_DIR:-$PROJECT_ROOT/logs/ceo_sessions}"
MODEL="${CEO_MODEL:-sonnet}"
PERM_MODE="${CEO_PERMISSION_MODE:-bypassPermissions}"
SESSION_COUNT=0
TAIL_PID=""

mkdir -p "$LOG_DIR"

# Minimal prompt — CEO reads CLAUDE.md + STATE.yaml for full context
CEO_PROMPT='Continue work. Read CLAUDE.md, roadmap/00_STATE.yaml, coordination/TASK_CONTRACTS.md, check inbox (PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py startup ceo). Use GPU for all model training. Do not narrate — execute experiments and log results to files. Commit frequently.'

cleanup_tail() {
    if [ -n "$TAIL_PID" ] && kill -0 "$TAIL_PID" 2>/dev/null; then
        kill "$TAIL_PID" 2>/dev/null || true
        wait "$TAIL_PID" 2>/dev/null || true
        TAIL_PID=""
    fi
}

trap cleanup_tail EXIT

echo "=== CEO Watchdog started ==="
echo "  Project:    $PROJECT_ROOT"
echo "  Cooldown:   ${BASE_COOLDOWN}s + jitter"
echo "  Max budget: \$${MAX_BUDGET} per session"
echo "  Model:      $MODEL"
echo "  Perms:      $PERM_MODE"
echo "  Logs:       $LOG_DIR"
echo ""

while true; do
    SESSION_COUNT=$((SESSION_COUNT + 1))
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/ceo_session_${SESSION_COUNT}_${TIMESTAMP}.log"

    # === Health gate: wait until system is healthy before launching ===
    echo "[$TIMESTAMP] Checking system health before session #$SESSION_COUNT..."
    while true; do
        bash "$PROJECT_ROOT/scripts/system_health_check.sh" > /dev/null 2>&1 && break
        EXIT=$?
        if [ "$EXIT" -eq 2 ]; then
            echo "[$(date +%Y%m%d_%H%M%S)] System CRITICAL — waiting 60s before recheck..."
        else
            echo "[$(date +%Y%m%d_%H%M%S)] System DEGRADED — waiting 30s before recheck..."
        fi
        sleep $((EXIT == 2 ? 60 : 30))
    done

    echo "[$TIMESTAMP] Starting CEO session #$SESSION_COUNT (log: $LOG_FILE)"

    # Touch log file so tail -f has something to attach to
    touch "$LOG_FILE"

    # Start tail -f in background so tmux shows live output
    tail -f "$LOG_FILE" &
    TAIL_PID=$!

    # Launch claude CLI in print mode, write directly to log file
    set +e
    cd "$PROJECT_ROOT" && claude -p \
        --model "$MODEL" \
        --max-budget-usd "$MAX_BUDGET" \
        --permission-mode "$PERM_MODE" \
        --output-format stream-json \
        "$CEO_PROMPT" \
        2>"$LOG_FILE.stderr" \
        | while IFS= read -r line; do
            TYPE=$(echo "$line" | jq -r '.type // empty' 2>/dev/null)
            case "$TYPE" in
                assistant)
                    TEXT=$(echo "$line" | jq -r '.content // empty' 2>/dev/null)
                    [ -n "$TEXT" ] && printf '%s' "$TEXT" >> "$LOG_FILE"
                    ;;
                tool_use)
                    TOOL=$(echo "$line" | jq -r '.tool // empty' 2>/dev/null)
                    INPUT=$(echo "$line" | jq -r '.input // empty' 2>/dev/null | head -c 200)
                    echo "[TOOL: $TOOL] $INPUT" >> "$LOG_FILE"
                    ;;
                tool_result)
                    CONTENT=$(echo "$line" | jq -r '.content // empty' 2>/dev/null | head -c 500)
                    [ -n "$CONTENT" ] && echo "[RESULT] $(echo "$CONTENT" | head -3)" >> "$LOG_FILE"
                    ;;
                result)
                    COST=$(echo "$line" | jq -r '.cost_usd // "unknown"' 2>/dev/null)
                    DURATION=$(echo "$line" | jq -r '.duration_ms // "unknown"' 2>/dev/null)
                    echo "" >> "$LOG_FILE"
                    echo "[SESSION COMPLETE] Cost: \$$COST | Duration: ${DURATION}ms" >> "$LOG_FILE"
                    ;;
                *)
                    echo "$line" >> "$LOG_FILE.raw"
                    ;;
            esac
        done
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    # Kill the tail process
    cleanup_tail

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo ""
    echo "[$TIMESTAMP] CEO session #$SESSION_COUNT ended (exit code: $EXIT_CODE)"
    echo "[$TIMESTAMP] Log: $LOG_FILE"

    # Check for STOP file
    if [ -f "$LOG_DIR/STOP" ]; then
        echo "[$TIMESTAMP] STOP file detected. Shutting down watchdog."
        rm -f "$LOG_DIR/STOP"
        break
    fi

    # Cooldown with jitter (base + 0-120s random)
    JITTER=$((RANDOM % 120))
    COOLDOWN=$((BASE_COOLDOWN + JITTER))
    echo "[$TIMESTAMP] Cooling down ${COOLDOWN}s (${BASE_COOLDOWN}+${JITTER}s jitter)..."
    echo "  (touch $LOG_DIR/STOP to gracefully stop)"
    sleep "$COOLDOWN"
done

echo "=== CEO Watchdog stopped ==="
