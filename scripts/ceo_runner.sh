#!/bin/bash
# CEO Agent Runner — streams output to log file in real-time
# Used by sparky-ceo.service (systemd)
#
# Lifecycle:
#   systemd starts this script → claude runs → exits (rate limit/completion)
#   → systemd restarts after RestartSec=120 → new session starts
#
# Logs: tail -f logs/ceo_sessions/latest.log
# Status: systemctl --user status sparky-ceo
# Stop: systemctl --user stop sparky-ceo
# Start: systemctl --user start sparky-ceo

set -uo pipefail

LOG_DIR="/home/akamath/sparky-ai/logs/ceo_sessions"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/ceo_session_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# Symlink for easy access
ln -sf "$LOG_FILE" "${LOG_DIR}/latest.log"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] CEO Agent starting" | tee "$LOG_FILE"

cd /home/akamath/sparky-ai
source .venv/bin/activate

# Unset Claude Code env vars so nested claude can launch
unset CLAUDECODE CLAUDE_CODE_SSE_PORT CLAUDE_CODE_ENTRYPOINT 2>/dev/null || true

# Health gate — skip session if system is CRITICAL
if bash scripts/system_health_check.sh /tmp/health.txt 2>/dev/null; then
    HC_EXIT=$?
else
    HC_EXIT=$?
fi
if [ "${HC_EXIT:-0}" -eq 2 ]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] System CRITICAL — skipping session, will retry after RestartSec" | tee -a "$LOG_FILE"
    exit 1
fi

# Run claude with stream-json output, parse to readable log
# stdbuf ensures line-buffered output for real-time streaming
# Exit code 0 on success, non-zero on rate limit/error → systemd will restart
stdbuf -oL claude -p \
  "Read CLAUDE.md. Check coordination inbox. Read roadmap/00_STATE.yaml and roadmap/02_RESEARCH_LOG.md. Then continue research — if CONTRACT #004 is active, execute it. If not, pick up the next unblocked task. Do NOT ask permission or present option menus. Commit frequently to a branch." \
  --model sonnet \
  --verbose \
  --output-format stream-json 2>>"${LOG_DIR}/ceo_systemd.err" | \
  stdbuf -oL python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
        if msg.get('type') == 'assistant':
            content = msg.get('message', {}).get('content', '')
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        print(block['text'], flush=True)
            elif isinstance(content, str):
                print(content, flush=True)
        elif msg.get('type') == 'result':
            content = msg.get('result', '')
            if content:
                print(content, flush=True)
    except json.JSONDecodeError:
        print(line, flush=True)
" >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] CEO Agent exited (code=$EXIT_CODE)" | tee -a "$LOG_FILE"
exit $EXIT_CODE
