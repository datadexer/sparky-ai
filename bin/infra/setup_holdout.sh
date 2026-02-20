#!/usr/bin/env bash
# Setup OOS holdout data: create user, fetch hourly data, resample, lock down.
# Idempotent — safe to re-run. Skips steps that are already done.
#
# Usage:
#   sudo bin/infra/setup_holdout.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV="$PROJECT_ROOT/.venv"
HOLDOUT="$PROJECT_ROOT/data/holdout"
VAULT="$PROJECT_ROOT/data/.oos_vault"
FETCH_START="2023-11-01"  # 2 months before OOS boundary for Donchian warmup

echo "=== Sparky OOS Holdout Setup ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Must be root
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Run with sudo"
    exit 1
fi

# Detect the non-root user who owns the project
PROJECT_OWNER=$(stat -c '%U' "$PROJECT_ROOT")
echo "Project owner: $PROJECT_OWNER"

# Step 1: Create sparky-oos user
echo ""
echo "--- Step 1: System user ---"
if id sparky-oos &>/dev/null; then
    echo "  SKIP: sparky-oos user already exists"
else
    useradd -r -s /usr/sbin/nologin sparky-oos
    echo "  Created sparky-oos system user"
fi

# Step 2: Ensure venv is world-readable
echo ""
echo "--- Step 2: Venv permissions ---"
if [ ! -d "$VENV" ]; then
    echo "ERROR: $VENV not found"
    exit 1
fi
chmod -R o+rX "$VENV/lib/"
chmod -R o+rX "$VENV/bin/"
# sparky-oos needs to traverse the path to reach .venv/bin/python
# Add o+x to each directory in the chain from / to PROJECT_ROOT
DIR="$PROJECT_ROOT"
while [ "$DIR" != "/" ]; do
    PERMS=$(stat -c '%a' "$DIR")
    if [ $((PERMS % 10 & 1)) -eq 0 ]; then
        chmod o+x "$DIR"
        echo "  Added o+x to $DIR (was $PERMS)"
    fi
    DIR=$(dirname "$DIR")
done
echo "  .venv/lib/ and .venv/bin/ are now world-readable/executable"

# Step 2b: Ensure src/ and data/ (except holdout) are world-readable for imports + IS data
echo ""
echo "--- Step 2b: Source & data permissions ---"
chmod -R o+rX "$PROJECT_ROOT/src/"
# data/ dir itself (not recursive — avoid touching holdout)
chmod o+rX "$PROJECT_ROOT/data/" 2>/dev/null || true
# IS data subdirs only
for DDIR in "$PROJECT_ROOT/data/features" "$PROJECT_ROOT/data/processed" "$PROJECT_ROOT/data/raw"; do
    if [ -d "$DDIR" ]; then
        chmod -R o+rX "$DDIR"
    fi
done
echo "  src/ and data/{features,processed,raw} are now world-readable"

# Step 3: Fetch hourly data from OKX (skip if files already exist)
echo ""
echo "--- Step 3: Fetch hourly OHLCV from OKX ---"
mkdir -p "$HOLDOUT/btc" "$HOLDOUT/eth"

BTC_HOURLY="$HOLDOUT/btc/ohlcv_hourly_max_coverage.parquet"
ETH_HOURLY="$HOLDOUT/eth/ohlcv_hourly.parquet"

if [ -f "$BTC_HOURLY" ] && [ -f "$ETH_HOURLY" ]; then
    echo "  SKIP: Hourly parquets already exist"
else
    # Ensure dirs are writable by project owner for fetch
    chown -R "$PROJECT_OWNER":"$PROJECT_OWNER" "$HOLDOUT"

    sudo -u "$PROJECT_OWNER" "$VENV/bin/python" -c "
import ccxt
import pandas as pd
from pathlib import Path

exchange = ccxt.okx({'enableRateLimit': True})
FETCH_START = '$FETCH_START'

assets = [
    ('BTC/USDT', 'btc', '$BTC_HOURLY'),
    ('ETH/USDT', 'eth', '$ETH_HOURLY'),
]

for symbol, asset, out_path in assets:
    if Path(out_path).exists():
        print(f'  SKIP {asset}: {out_path} already exists')
        continue
    print(f'  Fetching {symbol} hourly from {FETCH_START}...')
    since = int(pd.Timestamp(FETCH_START, tz='UTC').timestamp() * 1000)
    end = int(pd.Timestamp.now('UTC').timestamp() * 1000)
    all_candles = []
    while since < end:
        candles = exchange.fetch_ohlcv(symbol, '1h', since=since, limit=300)
        if not candles:
            break
        all_candles.extend(candles)
        last_ts = candles[-1][0]
        if last_ts <= since:
            break
        since = last_ts + 3_600_000
        if len(all_candles) % 3000 == 0:
            print(f'    {len(all_candles)} candles...')

    df = pd.DataFrame(all_candles, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    df.to_parquet(out_path)
    print(f'  {asset}: {len(df)} candles ({df.index.min()} -> {df.index.max()})')
"
    echo "  Fetch complete"
fi

# Step 4: Resample to 8h (skip if 8h files already exist)
echo ""
echo "--- Step 4: Resample to 8h ---"
BTC_8H="$HOLDOUT/btc/ohlcv_8h.parquet"
ETH_8H="$HOLDOUT/eth/ohlcv_8h.parquet"

if [ -f "$BTC_8H" ] && [ -f "$ETH_8H" ]; then
    echo "  SKIP: 8h parquets already exist"
else
    chown -R sparky-oos:sparky-oos "$HOLDOUT"
    chmod -R 700 "$HOLDOUT"
    sudo -u sparky-oos "$VENV/bin/python" "$PROJECT_ROOT/scripts/build_holdout_resampled.py"
fi

# Step 5: Lock down
echo ""
echo "--- Step 5: Lock down permissions ---"
chown -R sparky-oos:sparky-oos "$HOLDOUT"
chmod 700 "$HOLDOUT"
chmod 700 "$HOLDOUT/btc" "$HOLDOUT/eth"
echo "  data/holdout/ owned by sparky-oos, mode 700"

if sudo -u "$PROJECT_OWNER" ls "$HOLDOUT/" &>/dev/null; then
    echo "  WARNING: $PROJECT_OWNER can still read data/holdout/ — check permissions"
else
    echo "  Verified: $PROJECT_OWNER cannot read data/holdout/"
fi

# Step 6: Remove legacy vault
echo ""
echo "--- Step 6: Remove legacy vault ---"
if [ -d "$VAULT" ]; then
    rm -rf "$VAULT"
    echo "  Removed data/.oos_vault/"
else
    echo "  SKIP: data/.oos_vault/ not found (already removed)"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run OOS evaluation:"
echo "  sudo -u sparky-oos SPARKY_OOS_ENABLED=1 $VENV/bin/python $PROJECT_ROOT/scripts/oos/oos_evaluate.py $PROJECT_ROOT/configs/oos/champion_btc82_eth83.yaml"
echo ""
echo "Report will be written to /tmp/sparky_oos_reports/. To copy into project:"
echo "  cp /tmp/sparky_oos_reports/oos_evaluation_*.md $PROJECT_ROOT/reports/oos/"
