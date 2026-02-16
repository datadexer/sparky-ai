#!/usr/bin/env bash
# CEO Watchdog — keeps the CEO agent running across rate limits and context resets.
#
# Usage:
#   ./scripts/ceo_watchdog.sh start   # start daemon (background)
#   ./scripts/ceo_watchdog.sh stop    # stop daemon
#   ./scripts/ceo_watchdog.sh logs    # tail -f latest session log
#   ./scripts/ceo_watchdog.sh status  # check if running
#
# Environment:
#   CEO_COOLDOWN_SECS   — base cooldown between sessions (default: 300)
#   CEO_MAX_BUDGET_USD  — max spend per session (default: 15.00)
#   CEO_MODEL           — model to use (default: sonnet)
#   CEO_PERMISSION_MODE — permission mode (default: bypassPermissions)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_COOLDOWN="${CEO_COOLDOWN_SECS:-60}"
MAX_BUDGET="${CEO_MAX_BUDGET_USD:-15.00}"
LOG_DIR="$PROJECT_ROOT/logs/ceo_sessions"
MODEL="${CEO_MODEL:-sonnet}"
PERM_MODE="${CEO_PERMISSION_MODE:-bypassPermissions}"
PID_FILE="$LOG_DIR/watchdog.pid"
LATEST_LINK="$LOG_DIR/latest.log"

mkdir -p "$LOG_DIR"

CEO_PROMPT='Continue work. Read CLAUDE.md, roadmap/00_STATE.yaml, coordination/TASK_CONTRACTS.md, check inbox (PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py startup ceo). Use GPU for all model training. Do not narrate — execute experiments and log results to files. Commit frequently.'

run_daemon() {
    # Unset Claude Code env vars so nested claude can launch
    unset CLAUDECODE CLAUDE_CODE_SSE_PORT CLAUDE_CODE_ENTRYPOINT
    echo $$ > "$PID_FILE"
    SESSION_COUNT=0

    while true; do
        SESSION_COUNT=$((SESSION_COUNT + 1))
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        LOG_FILE="$LOG_DIR/ceo_session_${SESSION_COUNT}_${TIMESTAMP}.log"
        touch "$LOG_FILE"
        ln -sf "$LOG_FILE" "$LATEST_LINK"

        # Health gate — only block on CRITICAL (exit 2), allow DEGRADED
        while true; do
            bash "$PROJECT_ROOT/scripts/system_health_check.sh" > /dev/null 2>&1
            HC_EXIT=$?
            [ "$HC_EXIT" -ne 2 ] && break
            echo "[$(date)] System CRITICAL — waiting 60s" >> "$LOG_FILE"
            sleep 60
        done

        echo "[$(date)] Starting CEO session #$SESSION_COUNT" >> "$LOG_FILE"

        # Run claude, all output to log file
        cd "$PROJECT_ROOT" && claude -p \
            --model "$MODEL" \
            --max-budget-usd "$MAX_BUDGET" \
            --permission-mode "$PERM_MODE" \
            "$CEO_PROMPT" \
            >> "$LOG_FILE" 2>&1 || true

        echo "[$(date)] Session #$SESSION_COUNT ended" >> "$LOG_FILE"

        # Check STOP file
        if [ -f "$LOG_DIR/STOP" ]; then
            rm -f "$LOG_DIR/STOP" "$PID_FILE"
            echo "[$(date)] Stopped by STOP file" >> "$LOG_FILE"
            exit 0
        fi

        # Cooldown with jitter
        JITTER=$((RANDOM % 30))
        COOLDOWN=$((BASE_COOLDOWN + JITTER))
        echo "[$(date)] Cooling down ${COOLDOWN}s" >> "$LOG_FILE"
        sleep "$COOLDOWN"
    done
}

case "${1:-help}" in
    start)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Already running (PID $(cat "$PID_FILE"))"
            exit 1
        fi
        echo "Starting CEO watchdog daemon..."
        nohup bash "$0" _daemon > /dev/null 2>&1 &
        sleep 1
        if [ -f "$PID_FILE" ]; then
            echo "Started (PID $(cat "$PID_FILE"))"
            echo "Logs: $0 logs"
        else
            echo "Failed to start"
            exit 1
        fi
        ;;
    _daemon)
        run_daemon
        ;;
    stop)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            # Kill the watchdog and its children (claude process)
            pkill -P "$PID" 2>/dev/null || true
            kill "$PID" 2>/dev/null || true
            rm -f "$PID_FILE"
            echo "Stopped (was PID $PID)"
        else
            echo "Not running"
        fi
        ;;
    logs)
        if [ -L "$LATEST_LINK" ] || [ -f "$LATEST_LINK" ]; then
            echo "=== Tailing $(readlink -f "$LATEST_LINK") ==="
            tail -f "$LATEST_LINK"
        else
            LATEST=$(ls -t "$LOG_DIR"/ceo_session_*.log 2>/dev/null | head -1)
            if [ -n "$LATEST" ]; then
                echo "=== Tailing $LATEST ==="
                tail -f "$LATEST"
            else
                echo "No logs yet"
            fi
        fi
        ;;
    status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Running (PID $(cat "$PID_FILE"))"
            LATEST=$(ls -t "$LOG_DIR"/ceo_session_*.log 2>/dev/null | head -1)
            [ -n "$LATEST" ] && echo "Latest log: $LATEST ($(wc -c < "$LATEST") bytes)"
        else
            echo "Not running"
            rm -f "$PID_FILE"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|logs|status}"
        ;;
esac
