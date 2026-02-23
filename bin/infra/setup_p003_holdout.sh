#!/usr/bin/env bash
# Split P003 data at holdout boundary and lock down permissions.
# Mirrors the pattern of bin/infra/setup_holdout.sh.
#
# Usage:
#   sudo bin/infra/setup_p003_holdout.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV="$PROJECT_ROOT/.venv"
P003_DATA="$PROJECT_ROOT/data/p003"
HOLDOUT_P003="$PROJECT_ROOT/data/holdout/p003"

echo "=== P003 Holdout Split & Lockdown ==="
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

# Verify P003 data exists
if [ ! -d "$P003_DATA" ]; then
    echo "ERROR: $P003_DATA does not exist. Download data first."
    exit 1
fi

# Step 1: Create holdout subdirectories
echo ""
echo "--- Step 1: Create holdout directories ---"
mkdir -p "$HOLDOUT_P003/binance_perps" "$HOLDOUT_P003/funding_rates" "$HOLDOUT_P003/dvol"
echo "  Created $HOLDOUT_P003/{binance_perps,funding_rates,dvol}"

# Step 2: Temporarily chown to project owner so Python can write
echo ""
echo "--- Step 2: Set temp ownership for split ---"
chown -R "$PROJECT_OWNER":"$PROJECT_OWNER" "$HOLDOUT_P003"
echo "  Holdout P003 dir owned by $PROJECT_OWNER (temporary)"

# Step 3: Run split as project owner
echo ""
echo "--- Step 3: Split P003 data ---"
sudo -u "$PROJECT_OWNER" "$VENV/bin/python" "$PROJECT_ROOT/bin/infra/split_p003_holdout.py"
SPLIT_EXIT=$?
if [ $SPLIT_EXIT -ne 0 ]; then
    echo "ERROR: Split script failed (exit $SPLIT_EXIT)"
    exit 1
fi

# Step 4: Lock down holdout permissions
echo ""
echo "--- Step 4: Lock down permissions ---"
chown -R sparky-oos:sparky-oos "$HOLDOUT_P003"
chmod 700 "$HOLDOUT_P003"
chmod 700 "$HOLDOUT_P003/binance_perps" "$HOLDOUT_P003/funding_rates" "$HOLDOUT_P003/dvol"
echo "  data/holdout/p003/ owned by sparky-oos, mode 700"

# Step 5: Verify project owner cannot read
echo ""
echo "--- Step 5: Verify permissions ---"
if sudo -u "$PROJECT_OWNER" ls "$HOLDOUT_P003/" &>/dev/null; then
    echo "  WARNING: $PROJECT_OWNER can still read data/holdout/p003/ — check permissions"
else
    echo "  Verified: $PROJECT_OWNER cannot read data/holdout/p003/"
fi

# Step 6: Run scanner to confirm clean state
echo ""
echo "--- Step 6: Final validation ---"
sudo -u "$PROJECT_OWNER" "$VENV/bin/python" "$PROJECT_ROOT/bin/infra/scan_data_holdout.py"

echo ""
echo "=== P003 holdout setup complete ==="
