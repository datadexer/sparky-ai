#!/bin/bash
# Cron CEO guardrail — runs every 30 minutes via crontab.
# Pure bash, zero API cost.
#
# Checks:
# 1. Python process count (kills excess if >6)
# 2. Disk usage growth rate
# 3. CEO session still alive
# 4. Holdout data violations in session logs
# 5. Stall detection (>4 hours without commit)

set -uo pipefail

PROJECT_ROOT="/home/akamath/sparky-ai"
ALERT_DIR="$PROJECT_ROOT/logs/alerts"
CEO_STOP_FILE="$PROJECT_ROOT/logs/ceo_sessions/STOP"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VIOLATIONS=0

mkdir -p "$ALERT_DIR"

alert() {
    local severity="$1"
    local message="$2"
    local alert_file="$ALERT_DIR/guardrail_${severity}_${TIMESTAMP}.txt"
    {
        echo "GUARDRAIL ALERT: $severity"
        echo "Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo ""
        echo "$message"
    } >> "$alert_file"
    VIOLATIONS=$((VIOLATIONS + 1))
}

# === 1. Python process count ===
PYTHON_COUNT=$(pgrep -c python3 2>/dev/null || echo 0)
if [ "$PYTHON_COUNT" -gt 6 ]; then
    alert "WARNING" "Excess python processes: $PYTHON_COUNT (limit: 6). Killing newest excess processes."

    # Kill newest python processes beyond limit of 6
    # Sort by elapsed time ascending (newest first = smallest elapsed)
    PIDS_TO_KILL=$(ps -eo pid,etimes,comm --sort=etimes | grep python3 | head -n -6 | awk '{print $1}')
    for PID in $PIDS_TO_KILL; do
        CMD=$(ps -p "$PID" -o args= 2>/dev/null || echo "unknown")
        echo "  Killing PID $PID: $CMD" >> "$ALERT_DIR/guardrail_WARNING_${TIMESTAMP}.txt"
        kill -TERM "$PID" 2>/dev/null || true
    done
fi

# === 2. Disk usage growth ===
DISK_USED_KB=$(df -k / | awk 'NR==2{print $3}')
DISK_SNAPSHOT="$PROJECT_ROOT/logs/.disk_snapshot"
if [ -f "$DISK_SNAPSHOT" ]; then
    PREV_KB=$(cat "$DISK_SNAPSHOT")
    GROWTH_KB=$((DISK_USED_KB - PREV_KB))
    GROWTH_GB=$(echo "scale=2; $GROWTH_KB / 1048576" | bc 2>/dev/null || echo "0")
    # Alert if >5GB growth in 30 minutes
    if [ "$GROWTH_KB" -gt 5242880 ]; then
        alert "WARNING" "Disk grew ${GROWTH_GB}GB in last 30 minutes. Possible runaway data generation."
    fi
fi
echo "$DISK_USED_KB" > "$DISK_SNAPSHOT"

# === 3. CEO session alive check ===
CEO_ALIVE=false
if tmux has-session -t ceo 2>/dev/null; then
    CEO_ALIVE=true
fi
# Also check for claude process on pts/7 or similar
CEO_PID=$(pgrep -f "claude.*-p.*CEO" 2>/dev/null | head -1 || true)
if [ -z "$CEO_PID" ]; then
    # Check for any non-interactive claude in the ceo tmux
    CEO_PID=$(pgrep -f "scripts/ceo_watchdog" 2>/dev/null | head -1 || true)
fi

if [ "$CEO_ALIVE" = false ] && [ -z "$CEO_PID" ]; then
    alert "INFO" "CEO watchdog is not running. Start with: tmux new-session -d -s ceo '$PROJECT_ROOT/scripts/ceo_watchdog.sh'"
fi

# === 4. Holdout violation detection ===
# Scan recent CEO session logs for access to holdout period (2024-07 onwards)
HOLDOUT_PATTERNS="2024-0[7-9]|2024-1[0-2]|2025-|2026-"
LATEST_CEO_LOG=$(ls -t "$PROJECT_ROOT/logs/ceo_sessions"/ceo_session_*.log 2>/dev/null | head -1)

if [ -n "$LATEST_CEO_LOG" ] && [ -s "$LATEST_CEO_LOG" ]; then
    # Check last 500 lines for holdout period references in data operations
    # Look for patterns that suggest actual data access, not just mentions
    HOLDOUT_HITS=$(tail -500 "$LATEST_CEO_LOG" | grep -cE "(loc\[.*($HOLDOUT_PATTERNS)|test_start.*=.*($HOLDOUT_PATTERNS)|test_end.*=.*($HOLDOUT_PATTERNS)|\.loc\[\"($HOLDOUT_PATTERNS))" 2>/dev/null || echo 0)
    HOLDOUT_HITS=$(echo "$HOLDOUT_HITS" | tr -d '[:space:]')

    if [ "$HOLDOUT_HITS" -gt 0 ]; then
        alert "CRITICAL" "HOLDOUT VIOLATION DETECTED in CEO session log!
Found $HOLDOUT_HITS references to holdout period (2024-07+) in data access patterns.
Log: $LATEST_CEO_LOG
Matching lines:
$(tail -500 "$LATEST_CEO_LOG" | grep -E "(loc\[.*($HOLDOUT_PATTERNS)|test_start.*=.*($HOLDOUT_PATTERNS)|test_end.*=.*($HOLDOUT_PATTERNS)|\.loc\[\"($HOLDOUT_PATTERNS))" 2>/dev/null | head -10)"
        # Stop the CEO
        touch "$CEO_STOP_FILE"
    fi
fi

# Also check any python scripts the CEO may have created/modified recently
for f in "$PROJECT_ROOT"/scripts/*.py; do
    if [ -f "$f" ] && [ "$(find "$f" -mmin -35 2>/dev/null)" ]; then
        # Recently modified script — check for holdout access
        SCRIPT_HOLDOUT=$(grep -cE "test_end.*=.*(2024-0[7-9]|2024-1[0-2]|2025-|2026-)" "$f" 2>/dev/null || echo 0)
        SCRIPT_HOLDOUT=$(echo "$SCRIPT_HOLDOUT" | tr -d '[:space:]')
        if [ "$SCRIPT_HOLDOUT" -gt 0 ]; then
            alert "CRITICAL" "HOLDOUT VIOLATION in script: $f
Script accesses holdout period (post 2024-07-01).
Matching lines:
$(grep -nE "test_end.*=.*(2024-0[7-9]|2024-1[0-2]|2025-|2026-)" "$f" 2>/dev/null | head -5)"
            touch "$CEO_STOP_FILE"
        fi
    fi
done

# === 5. Stall detection (>4 hours without commit) ===
LAST_COMMIT_EPOCH=$(git -C "$PROJECT_ROOT" log -1 --format=%ct 2>/dev/null || echo 0)
NOW_EPOCH=$(date +%s)
HOURS_SINCE_COMMIT=$(( (NOW_EPOCH - LAST_COMMIT_EPOCH) / 3600 ))

if [ "$CEO_ALIVE" = true ] && [ "$HOURS_SINCE_COMMIT" -gt 4 ]; then
    alert "WARNING" "CEO stall detected: last commit was ${HOURS_SINCE_COMMIT} hours ago.
Last commit: $(git -C "$PROJECT_ROOT" log -1 --oneline 2>/dev/null)
The CEO may be stuck in a loop or hitting errors without committing progress."
fi

# === Summary ===
if [ "$VIOLATIONS" -eq 0 ]; then
    # All clear — silent exit
    exit 0
fi
