#!/bin/bash
# System health check — run before and during agent sessions
# Returns exit code 0 if healthy, 1 if degraded, 2 if critical

set -uo pipefail

OUTPUT_FILE="${1:-}"
USE_STDOUT=false
if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE=$(mktemp)
    USE_STDOUT=true
fi

# Thresholds
WARN_CPU=70
CRIT_CPU=85
WARN_MEM=92
CRIT_MEM=97
WARN_DISK_GB=50
CRIT_DISK_GB=20
MAX_PYTHON_PROCS=8     # Hard ceiling on total python processes
MAX_CLAUDE_AGENTS=3    # Inferred from claude/task tool processes

STATUS="HEALTHY"

# CPU (1-second sample)
CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print int($2 + $4)}')
if [ "$CPU" -gt "$CRIT_CPU" ]; then STATUS="CRITICAL"; fi
if [ "$CPU" -gt "$WARN_CPU" ] && [ "$STATUS" = "HEALTHY" ]; then STATUS="DEGRADED"; fi

# Memory
MEM_TOTAL=$(free -g | awk '/^Mem:/{print $2}')
MEM_USED=$(free -g | awk '/^Mem:/{print $3}')
MEM_PCT=$((MEM_USED * 100 / MEM_TOTAL))
if [ "$MEM_PCT" -gt "$CRIT_MEM" ]; then STATUS="CRITICAL"; fi
if [ "$MEM_PCT" -gt "$WARN_MEM" ] && [ "$STATUS" = "HEALTHY" ]; then STATUS="DEGRADED"; fi

# Disk
DISK_FREE_GB=$(df -BG / | awk 'NR==2{print int($4)}')
if [ "$DISK_FREE_GB" -lt "$CRIT_DISK_GB" ]; then STATUS="CRITICAL"; fi
if [ "$DISK_FREE_GB" -lt "$WARN_DISK_GB" ] && [ "$STATUS" = "HEALTHY" ]; then STATUS="DEGRADED"; fi

# Python process count (proxy for agent count)
PYTHON_PROCS=$(pgrep -c python3 2>/dev/null || echo 0)
PYTHON_PROCS=$(echo "$PYTHON_PROCS" | head -1 | tr -d '[:space:]')

# Claude Code agent detection (look for node/claude processes)
CLAUDE_PROCS=$(pgrep -fc "claude" 2>/dev/null || echo 0)
CLAUDE_PROCS=$(echo "$CLAUDE_PROCS" | head -1 | tr -d '[:space:]')

# Top memory consumers
TOP_PROCS=$(ps aux --sort=-%mem | head -6 | tail -5 | awk '{printf "  %s %.1fGB %s\n", $1, $6/1048576, $11}')

# GPU status (DGX Spark)
GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "N/A")

# Build report
{
    echo "=== SYSTEM HEALTH CHECK ==="
    echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Status: $STATUS"
    echo ""
    echo "CPU: ${CPU}% (warn: ${WARN_CPU}%, crit: ${CRIT_CPU}%)"
    echo "Memory: ${MEM_USED}GB / ${MEM_TOTAL}GB (${MEM_PCT}%)"
    echo "Disk Free: ${DISK_FREE_GB}GB (warn: ${WARN_DISK_GB}GB, crit: ${CRIT_DISK_GB}GB)"
    echo "Python Processes: ${PYTHON_PROCS} (max: ${MAX_PYTHON_PROCS})"
    echo "Claude Processes: ${CLAUDE_PROCS} (max: ${MAX_CLAUDE_AGENTS})"
    echo "GPU: ${GPU_INFO}"
    echo ""
    echo "Top Memory Consumers:"
    echo "$TOP_PROCS"
    echo ""

    if [ "$PYTHON_PROCS" -gt "$MAX_PYTHON_PROCS" ]; then
        echo "WARNING: ${PYTHON_PROCS} python processes exceeds limit of ${MAX_PYTHON_PROCS}"
        echo "  Active python processes:"
        ps aux | grep python3 | grep -v grep | awk '{printf "  PID %s: %.1fGB %s\n", $2, $6/1048576, $11}'
        STATUS="DEGRADED"
    fi

    echo ""
    echo "=== END HEALTH CHECK (${STATUS}) ==="
} > "$OUTPUT_FILE"

if [ "$USE_STDOUT" = true ]; then
    cat "$OUTPUT_FILE"
    rm -f "$OUTPUT_FILE"
fi

# Exit codes
case "$STATUS" in
    HEALTHY)  exit 0 ;;
    DEGRADED) exit 1 ;;
    CRITICAL) exit 2 ;;
esac
