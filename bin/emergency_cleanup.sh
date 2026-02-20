#!/bin/bash
# Emergency cleanup — kill runaway python processes if system is unresponsive
# ONLY run this if system_health_check.sh returns CRITICAL (exit 2)

set -euo pipefail

echo "=== EMERGENCY CLEANUP ==="
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# Log current state before cleanup
echo "Current python processes:"
ps aux | grep python3 | grep -v grep || echo "  (none)"

# Count processes
PYTHON_COUNT=$(pgrep -c python3 2>/dev/null || echo 0)
echo "Total python processes: $PYTHON_COUNT"

if [ "$PYTHON_COUNT" -gt 6 ]; then
    echo ""
    echo "WARNING: KILLING excess python processes (keeping up to 3 oldest)..."

    # Get PIDs sorted by start time (oldest first), skip the first 3
    PIDS_TO_KILL=$(ps -eo pid,etimes,comm --sort=etimes | grep python3 | awk 'NR>3{print $1}')

    for PID in $PIDS_TO_KILL; do
        CMD=$(ps -p "$PID" -o args= 2>/dev/null || echo "unknown")
        echo "  Killing PID $PID: $CMD"
        kill -TERM "$PID" 2>/dev/null || true
    done

    sleep 5

    # Force kill any that didn't terminate
    for PID in $PIDS_TO_KILL; do
        if kill -0 "$PID" 2>/dev/null; then
            echo "  Force killing PID $PID"
            kill -9 "$PID" 2>/dev/null || true
        fi
    done
fi

echo ""
echo "Post-cleanup state:"
ps aux | grep python3 | grep -v grep || echo "  (none)"
echo "=== CLEANUP COMPLETE ==="
