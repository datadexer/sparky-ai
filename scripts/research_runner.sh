#!/bin/bash
# Research Agent Runner — streams output to log file in real-time
# Used by sparky-research.service (systemd)
#
# Lifecycle:
#   systemd starts this script → claude runs → exits (rate limit/completion)
#   → systemd restarts after RestartSec=120 → new session starts
#
# Logs: tail -f logs/research_sessions/latest.log
# Status: systemctl --user status sparky-research
# Stop: systemctl --user stop sparky-research
# Start: systemctl --user start sparky-research

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/research_sessions"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/research_session_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# Symlink for easy access
ln -sf "$LOG_FILE" "${LOG_DIR}/latest.log"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Research Agent starting" | tee "$LOG_FILE"

cd "$PROJECT_ROOT"
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
  "Read CLAUDE.md. You are continuing CONTRACT #004. Steps 1-2 are done (27 configs, TIER 4-5, no ML alpha from direct prediction). Steps 3-4 are NOT conditional on Step 2 success — they test a DIFFERENT HYPOTHESIS. Step 2 tested: Can ML predict price direction? Answer: No (AUC 0.5746). Step 3 tests: Can ML classify market REGIME (bull/bear/chop) so Donchian only trades in trending regimes? These are fundamentally different. Step 3 — do now: (1) Use scripts/train_regime_aware.py as scaffold, (2) Implement at least 2 regime detection methods (volatility threshold, ML classifier), (3) Donchian in trending/bull, flat in chop/bear, (4) Walk-forward validate combined system, (5) Log to wandb using log_sweep() for sweeps, use GPU, data loader, timeout per CLAUDE.md. Step 4: Only if Step 3 produces TIER 2+. Start immediately." \
  --model sonnet \
  --verbose \
  --output-format stream-json 2>>"${LOG_DIR}/research_systemd.err" | \
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
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Research Agent exited (code=$EXIT_CODE)" | tee -a "$LOG_FILE"
exit $EXIT_CODE
