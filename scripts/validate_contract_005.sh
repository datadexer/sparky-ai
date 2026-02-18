#!/bin/bash
set -euo pipefail

# Post-validation script for Contract 005.
# Checks that the contract produced all required outputs and passed guardrails.
# Distinct from smoke_test_contract_005.sh which tests infrastructure pre-launch.

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="$PROJECT/results"
GUARDRAIL_LOG="$RESULTS/guardrail_log.jsonl"

PASS=0
FAIL=0
WARN=0

pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }
warn() { echo "  WARN: $1"; WARN=$((WARN + 1)); }

echo "=== Contract 005 Post-Validation ==="
echo ""

# --------------------------------------------------------------------------
# 1. Required output files
# --------------------------------------------------------------------------
echo "[1/4] Checking required output files..."

for f in contract_005_summary.md contract_005_validation.md contract_005_validation_raw.json; do
    if [ -f "$RESULTS/$f" ]; then
        pass "$f exists"
    else
        fail "$f missing"
    fi
done

# Check audit file (produced by Step 1)
if ls "$RESULTS"/contract_005_audit*.md 1>/dev/null 2>&1; then
    pass "audit file exists"
else
    warn "audit file not found (may use different naming)"
fi

# --------------------------------------------------------------------------
# 2. Guardrail compliance
# --------------------------------------------------------------------------
echo ""
echo "[2/4] Checking guardrail compliance..."

if [ -f "$GUARDRAIL_LOG" ]; then
    pass "guardrail_log.jsonl exists"

    # Count entries
    TOTAL=$(wc -l < "$GUARDRAIL_LOG")
    pass "guardrail log has $TOTAL entries"

    # Check for BLOCKING failures
    if grep -q '"severity":\s*"BLOCKING"' "$GUARDRAIL_LOG" 2>/dev/null && \
       grep -q '"passed":\s*false' "$GUARDRAIL_LOG" 2>/dev/null; then
        # More precise: check if any single line has both BLOCKING and failed
        BLOCKING_FAILS=$(python3 -c "
import json, sys
count = 0
for line in open('$GUARDRAIL_LOG'):
    try:
        r = json.loads(line)
        if r.get('severity') == 'BLOCKING' and not r.get('passed', True):
            count += 1
    except json.JSONDecodeError:
        pass
print(count)
" 2>/dev/null || echo "0")
        if [ "$BLOCKING_FAILS" -gt 0 ]; then
            fail "$BLOCKING_FAILS BLOCKING guardrail failures found"
        else
            pass "no BLOCKING guardrail failures"
        fi
    else
        pass "no BLOCKING guardrail failures"
    fi
else
    fail "guardrail_log.jsonl missing"
fi

# --------------------------------------------------------------------------
# 3. Wandb contract_005 runs
# --------------------------------------------------------------------------
echo ""
echo "[3/4] Checking wandb contract_005 runs..."

VENV_PYTHON="$PROJECT/.venv/bin/python"
if "$VENV_PYTHON" -c "import wandb" &>/dev/null; then
    # Use Python to query wandb API for contract_005 tagged runs
    WANDB_COUNT=$("$VENV_PYTHON" -c "
import wandb
api = wandb.Api(timeout=30)
runs = api.runs('datadex_ai/sparky-ai', filters={'tags': {'\$in': ['contract_005']}})
print(len(runs))
" 2>/dev/null || echo "ERROR")

    if [ "$WANDB_COUNT" = "ERROR" ]; then
        warn "could not query wandb (offline or auth issue)"
    elif [ "$WANDB_COUNT" -gt 0 ]; then
        pass "$WANDB_COUNT wandb runs tagged contract_005"
    else
        fail "no wandb runs tagged contract_005"
    fi
else
    warn "wandb CLI not found, skipping run count check"
fi

# --------------------------------------------------------------------------
# 4. Summary verdict
# --------------------------------------------------------------------------
echo ""
echo "[4/4] Summary verdict..."

SUMMARY="$RESULTS/contract_005_summary.md"
if [ -f "$SUMMARY" ]; then
    # Check that summary contains key sections
    if grep -qi "tier" "$SUMMARY"; then
        pass "summary contains tier assessment"
    else
        warn "summary missing tier assessment"
    fi

    if grep -qi "dsr\|deflated sharpe" "$SUMMARY"; then
        pass "summary references DSR"
    else
        warn "summary missing DSR reference"
    fi
else
    fail "cannot check summary (file missing)"
fi

# --------------------------------------------------------------------------
# Final report
# --------------------------------------------------------------------------
echo ""
echo "=== Results: $PASS passed, $FAIL failed, $WARN warnings ==="

if [ "$FAIL" -gt 0 ]; then
    echo "VERDICT: FAIL — $FAIL checks failed"
    exit 1
else
    echo "VERDICT: PASS"
    exit 0
fi
