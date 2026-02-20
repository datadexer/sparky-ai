#!/usr/bin/env bash
# alert.sh — Simple alerting: Slack webhook, desktop notification, log file.
#
# Usage: bin/alert.sh SEVERITY "Message text"
#   SEVERITY: INFO | WARN | ERROR | CRITICAL
#
# Slack: Set SPARKY_SLACK_WEBHOOK env var to enable.

set -euo pipefail

SEVERITY="${1:-INFO}"
MESSAGE="${2:-No message provided}"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LOG_FILE="logs/alerts.log"

mkdir -p "$(dirname "$LOG_FILE")"
echo "[$TIMESTAMP] [$SEVERITY] $MESSAGE" >> "$LOG_FILE"

# Desktop notification (non-blocking, best-effort)
if command -v notify-send &>/dev/null; then
    notify-send "Sparky AI [$SEVERITY]" "$MESSAGE" 2>/dev/null || true
fi

# Slack webhook (if configured)
if [[ -n "${SPARKY_SLACK_WEBHOOK:-}" ]]; then
    curl -s -X POST "$SPARKY_SLACK_WEBHOOK" \
        -H 'Content-type: application/json' \
        -d "{\"text\": \"[$SEVERITY] $MESSAGE\"}" \
        >/dev/null 2>&1 || true
fi

echo "[$SEVERITY] $MESSAGE"
