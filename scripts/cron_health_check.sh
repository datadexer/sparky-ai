#!/bin/bash
# Cron health check — runs every 15 minutes via crontab.
# Pure bash, zero API cost.
#
# - HEALTHY: silent (no output, no log)
# - DEGRADED: logs warning to logs/alerts/
# - CRITICAL: runs emergency_cleanup, pauses CEO, writes alert

set -uo pipefail

PROJECT_ROOT="/home/akamath/sparky-ai"
ALERT_DIR="$PROJECT_ROOT/logs/alerts"
CEO_STOP_FILE="$PROJECT_ROOT/logs/ceo_sessions/STOP"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$ALERT_DIR"

# Run system health check
HEALTH_OUTPUT=$(mktemp)
"$PROJECT_ROOT/scripts/system_health_check.sh" "$HEALTH_OUTPUT" 2>/dev/null
EXIT_CODE=$?

case $EXIT_CODE in
    0)
        # HEALTHY — silent, clean up temp file
        rm -f "$HEALTH_OUTPUT"
        ;;
    1)
        # DEGRADED — log warning
        ALERT_FILE="$ALERT_DIR/degraded_${TIMESTAMP}.txt"
        {
            echo "ALERT: DEGRADED system status detected"
            echo "Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
            echo ""
            cat "$HEALTH_OUTPUT"
        } > "$ALERT_FILE"
        rm -f "$HEALTH_OUTPUT"
        ;;
    2)
        # CRITICAL — emergency response
        ALERT_FILE="$ALERT_DIR/CRITICAL_${TIMESTAMP}.txt"
        {
            echo "ALERT: CRITICAL system status — emergency actions taken"
            echo "Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
            echo ""
            echo "Actions taken:"
            echo "  1. Running emergency_cleanup.sh"
            echo "  2. Pausing CEO watchdog (STOP file created)"
            echo ""
            echo "=== Health Check Output ==="
            cat "$HEALTH_OUTPUT"
            echo ""
            echo "=== Emergency Cleanup Output ==="
            bash "$PROJECT_ROOT/scripts/emergency_cleanup.sh" 2>&1
        } > "$ALERT_FILE"

        # Pause CEO watchdog
        touch "$CEO_STOP_FILE"

        rm -f "$HEALTH_OUTPUT"
        ;;
esac
