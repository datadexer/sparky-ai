#!/usr/bin/env bash
# CEO Watchdog — keeps the CEO agent running across rate limits and context resets.
#
# Usage:
#   tmux new-session -d -s ceo './scripts/ceo_watchdog.sh'
#   tmux attach -t ceo          # to watch
#   tmux kill-session -t ceo    # to stop
#
# Environment:
#   CEO_COOLDOWN_SECS   — seconds to wait between sessions (default: 300 = 5 min)
#   CEO_MAX_BUDGET_USD  — max spend per session (default: 5.00)
#   CEO_LOG_DIR         — log directory (default: logs/ceo_sessions)
#   CEO_MODEL           — model to use (default: sonnet)
#   CEO_PERMISSION_MODE — permission mode (default: bypassPermissions)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

COOLDOWN="${CEO_COOLDOWN_SECS:-300}"
MAX_BUDGET="${CEO_MAX_BUDGET_USD:-5.00}"
LOG_DIR="${CEO_LOG_DIR:-$PROJECT_ROOT/logs/ceo_sessions}"
MODEL="${CEO_MODEL:-sonnet}"
PERM_MODE="${CEO_PERMISSION_MODE:-bypassPermissions}"
SESSION_COUNT=0

mkdir -p "$LOG_DIR"

# The prompt the CEO gets on each fresh launch
read -r -d '' CEO_PROMPT << 'PROMPT_EOF' || true
You are the CEO agent of the Sparky AI autonomous trading research system. Resume work immediately:

1. Run coordination startup: PYTHONPATH=/home/akamath/sparky-ai python3 coordination/cli.py startup ceo
2. Read roadmap/00_STATE.yaml for current progress
3. Read roadmap/01_DECISIONS.md for pending decisions
4. Check git branch and status
5. Pick up the next unblocked task and execute it

KEY DIRECTIVES FROM AK (the human owner):
- Expand the feature set (more technical indicators, cross-timeframe, volume, macro)
- Expand model configurations (hyperparameter sweeps, LightGBM, CatBoost, ensembles)
- Be RIGOROUS about in-sample training only (2019-2023). You MUST ask oversight before any OOS evaluation on holdout data (2024+).
- Do NOT build paper trading until predictive models actually work.
- ALL timestamps must be timezone-aware UTC.
- Do not stop working. Keep executing experiments, logging results, and iterating.
- Commit work frequently. Push and create PRs when milestones are reached.

WORK CONTINUOUSLY. Do not present option menus. Do not declare failure after testing <5 configurations. Exhaust the search space methodically.
PROMPT_EOF

echo "=== CEO Watchdog started ==="
echo "  Project:    $PROJECT_ROOT"
echo "  Cooldown:   ${COOLDOWN}s between sessions"
echo "  Max budget: \$${MAX_BUDGET} per session"
echo "  Model:      $MODEL"
echo "  Perms:      $PERM_MODE"
echo "  Logs:       $LOG_DIR"
echo ""

while true; do
    SESSION_COUNT=$((SESSION_COUNT + 1))
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/ceo_session_${SESSION_COUNT}_${TIMESTAMP}.log"

    echo "[$TIMESTAMP] Starting CEO session #$SESSION_COUNT (log: $LOG_FILE)"

    # Launch claude CLI in non-interactive print mode
    # -p (--print): non-interactive, prints output and exits
    # --max-budget-usd: caps spend per session
    # --permission-mode: bypassPermissions for autonomous operation (sandboxed)
    # --model: use sonnet for cost efficiency
    set +e
    cd "$PROJECT_ROOT" && claude -p \
        --model "$MODEL" \
        --max-budget-usd "$MAX_BUDGET" \
        --permission-mode "$PERM_MODE" \
        "$CEO_PROMPT" \
        2>&1 | tee "$LOG_FILE"
    EXIT_CODE=$?
    set -e

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo ""
    echo "[$TIMESTAMP] CEO session #$SESSION_COUNT ended (exit code: $EXIT_CODE)"
    echo "[$TIMESTAMP] Log saved to: $LOG_FILE"

    # Check if we should stop (touch this file to gracefully stop the watchdog)
    if [ -f "$LOG_DIR/STOP" ]; then
        echo "[$TIMESTAMP] STOP file detected. Shutting down watchdog."
        rm -f "$LOG_DIR/STOP"
        break
    fi

    echo "[$TIMESTAMP] Cooling down for ${COOLDOWN}s before next session..."
    echo "  (touch $LOG_DIR/STOP to gracefully stop)"
    sleep "$COOLDOWN"
done

echo "=== CEO Watchdog stopped ==="
