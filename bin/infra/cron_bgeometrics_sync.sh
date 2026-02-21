#!/usr/bin/env bash
# BGeometrics daily incremental sync — runs via cron until all metrics backfilled.
# Priority: STH-SOPR, STH-MVRV first, then remaining P002 metrics.
# Rate limit: 15 req/day server-side. Large metrics need 2-3 pages each.
set -euo pipefail

cd /home/akamath/sparky-ai
LOGDIR="logs/bgeometrics_sync"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/$(date -u +%Y%m%d_%H%M%S).log"

exec >> "$LOGFILE" 2>&1
echo "=== BGeometrics sync started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# Reset the rate limit counter (new day = fresh budget)
echo '{"request_count": 0, "last_request_time": null, "last_hour_reset": null}' > data/.bgeometrics_rate_limit.json

.venv/bin/python bin/infra/bgeometrics_daily_sync.py

echo "=== BGeometrics sync finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# Prune logs older than 30 days
find "$LOGDIR" -name "*.log" -mtime +30 -delete 2>/dev/null || true
