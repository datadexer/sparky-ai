#!/bin/bash
# Cron RBM review — runs every 3 hours via crontab.
# Invokes Claude with RBM prompt in single-shot mode.
# Budget: capped at $0.50 per invocation.

set -uo pipefail

PROJECT_ROOT="/home/akamath/sparky-ai"
REVIEW_DIR="$PROJECT_ROOT/logs/rbm_reviews"
TIMESTAMP=$(date +%Y%m%d_%H%M)
REVIEW_FILE="$REVIEW_DIR/review_${TIMESTAMP}.md"

mkdir -p "$REVIEW_DIR"

# Gather context files for the RBM to review
HEALTH_OUTPUT=$(mktemp)
bash "$PROJECT_ROOT/scripts/system_health_check.sh" "$HEALTH_OUTPUT" 2>/dev/null || true

# Get latest CEO session log (last 200 lines)
LATEST_CEO_LOG=$(ls -t "$PROJECT_ROOT/logs/ceo_sessions"/ceo_session_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_CEO_LOG" ] && [ -s "$LATEST_CEO_LOG" ]; then
    CEO_LOG_TAIL=$(tail -200 "$LATEST_CEO_LOG")
else
    CEO_LOG_TAIL="(No CEO session log available or log is empty — CEO may still be running with buffered output)"
fi

# Get recent research log entries
RESEARCH_LOG=""
if [ -f "$PROJECT_ROOT/roadmap/02_RESEARCH_LOG.md" ]; then
    RESEARCH_LOG=$(tail -100 "$PROJECT_ROOT/roadmap/02_RESEARCH_LOG.md")
fi

# Get time tracking (last 50 entries)
TIME_TRACKING=""
if [ -f "$PROJECT_ROOT/logs/time_tracking.jsonl" ]; then
    TIME_TRACKING=$(tail -50 "$PROJECT_ROOT/logs/time_tracking.jsonl")
fi

# Get OOS evaluations
OOS_EVALS=""
if [ -f "$PROJECT_ROOT/results/oos_evaluations.jsonl" ]; then
    OOS_EVALS=$(cat "$PROJECT_ROOT/results/oos_evaluations.jsonl")
fi

# Get recent alerts
RECENT_ALERTS=""
if [ -d "$PROJECT_ROOT/logs/alerts" ]; then
    RECENT_ALERTS=$(ls -t "$PROJECT_ROOT/logs/alerts"/*.txt 2>/dev/null | head -5 | xargs cat 2>/dev/null || echo "(none)")
fi

# Get recent git activity
RECENT_COMMITS=$(git -C "$PROJECT_ROOT" log --oneline --all -20 2>/dev/null || echo "(git unavailable)")

# Get sweep results if available
SWEEP_RESULTS=""
for f in "$PROJECT_ROOT/results/validation/smart_sweep_intermediate.json" \
         "$PROJECT_ROOT/results/validation/smart_hyperparam_sweep.json" \
         "$PROJECT_ROOT/results/validation/hyperparam_sweep_comprehensive.json"; do
    if [ -f "$f" ]; then
        SWEEP_RESULTS="$SWEEP_RESULTS
=== $(basename "$f") ===
$(head -100 "$f")"
    fi
done

# Build RBM prompt
read -r -d '' RBM_PROMPT << 'PROMPT_EOF' || true
You are the Research Business Manager for Sparky AI. Produce a structured review of the CEO agent's recent work.

Read the context below and produce a review in this exact format:

# RBM Review — [DATE]

## System Health
[Summarize health check output. Flag any concerns.]

## CEO Activity Summary
[What has the CEO been doing? Is it productive or spinning?]

## Experiment Portfolio Assessment
- Active experiments: [list]
- Results so far: [summarize any sweep/validation results]
- Concerns: [overfitting? insufficient configs? wrong direction?]

## Validation Protocol Compliance
- In-sample discipline: [any holdout violations?]
- Statistical rigor: [bootstrap CIs? multi-seed? walk-forward?]
- Flags: [any preliminary results being treated as validated?]

## Recommendations
1. [Most important action item]
2. [Second priority]
3. [Third priority]

## Risk Flags
- [Any red flags requiring human attention, marked with severity: LOW/MEDIUM/HIGH/CRITICAL]
PROMPT_EOF

# Compose the full prompt with context
FULL_PROMPT="$RBM_PROMPT

=== CONTEXT ===

--- System Health Check ---
$(cat "$HEALTH_OUTPUT")

--- Recent CEO Session Log (last 200 lines) ---
$CEO_LOG_TAIL

--- Research Log (last 100 lines) ---
$RESEARCH_LOG

--- Recent Git Commits ---
$RECENT_COMMITS

--- Sweep Results ---
$SWEEP_RESULTS

--- OOS Evaluations ---
${OOS_EVALS:-(none)}

--- Time Tracking (last 50) ---
${TIME_TRACKING:-(none)}

--- Recent Alerts ---
$RECENT_ALERTS

Produce your structured RBM review now. Be concise and actionable."

# Invoke Claude in single-shot mode
cd "$PROJECT_ROOT" && claude -p \
    --model sonnet \
    --max-budget-usd 0.50 \
    --permission-mode bypassPermissions \
    "$FULL_PROMPT" \
    > "$REVIEW_FILE" 2>&1

# Append metadata footer
{
    echo ""
    echo "---"
    echo "_Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC') | Budget: \$0.50 cap | Model: sonnet_"
} >> "$REVIEW_FILE"

# Clean up
rm -f "$HEALTH_OUTPUT"
