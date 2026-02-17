#!/usr/bin/env bash
# Quick status check for CEO watchdog.
# Usage: bash scripts/check_ceo_status.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/ceo_sessions"

echo "=== CEO Watchdog Status ==="
echo ""

# Check if tmux session exists
if tmux has-session -t ceo 2>/dev/null; then
    echo "tmux session 'ceo': RUNNING"
else
    echo "tmux session 'ceo': NOT RUNNING"
fi
echo ""

# Latest log file
LATEST_LOG=$(ls -t "$LOG_DIR"/ceo_session_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "Latest log: $LATEST_LOG"
    echo "Last 20 lines:"
    echo "---"
    tail -20 "$LATEST_LOG"
    echo "---"
else
    echo "No CEO session logs found yet."
fi
echo ""

# Recent git activity
echo "Recent commits (last 5):"
git -C "$PROJECT_ROOT" log --oneline -5 --all
echo ""

# Workflow status
echo "Workflow state:"
if [ -f "$PROJECT_ROOT/workflows/state/contract-004.json" ]; then
    python3 -c "import json; d=json.load(open('$PROJECT_ROOT/workflows/state/contract-004.json')); print(f'Step {d[\"current_step_index\"]+1}/{len(d[\"steps\"])}  Budget: {d[\"budget\"][\"hours_used\"]:.1f}/{d[\"budget\"][\"max_hours\"]}h')" 2>/dev/null || echo "(workflow state unavailable)"
else
    echo "(no workflow state)"
fi
