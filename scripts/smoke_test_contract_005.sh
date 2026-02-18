#!/bin/bash
set -euo pipefail

VENV="/home/akamath/sparky-ai/.venv/bin"
PROJECT="/home/akamath/sparky-ai"

echo "=== Contract 005 Smoke Test ==="

# 1. Unit tests pass
echo "[1/7] Running unit tests..."
"$VENV/pytest" tests/test_guardrails.py tests/test_metrics.py tests/test_manager_log.py -v --tb=short

# 2. Integration tests pass
echo "[2/7] Running integration tests..."
"$VENV/pytest" tests/test_integration_contract005.py -v --tb=short

# 3. No CEO references remain in src/
echo "[3/7] Checking CEO references..."
if grep -rn 'agent_id="ceo"' src/ scripts/ 2>/dev/null; then
    echo "FAIL: CEO agent_id refs found"
    exit 1
else
    echo "PASS: no CEO agent_id refs"
fi
if grep -rn '"ceo_sessions"' src/ scripts/ 2>/dev/null; then
    echo "FAIL: ceo_sessions refs found"
    exit 1
else
    echo "PASS: no ceo_sessions refs"
fi

# 4. All new modules import cleanly
echo "[4/7] Import checks..."
"$VENV/python" -c "from sparky.tracking.guardrails import run_pre_checks, run_post_checks, has_blocking_failure"
"$VENV/python" -c "from sparky.tracking.manager_log import ManagerLog, ManagerSession"
"$VENV/python" -c "from sparky.tracking.metrics import compute_all_metrics, deflated_sharpe_ratio"
"$VENV/python" -c "from sparky.interfaces import StrategyProtocol, BacktesterProtocol, DataFeedProtocol"
echo "PASS: all imports succeeded"

# 5. Workflow file loads correctly
echo "[5/7] Workflow load check..."
"$VENV/python" -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('c005', 'workflows/contract_005_statistical_audit.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
wf = mod.build_workflow()
assert wf.name == 'contract-005-statistical-audit', f'Bad name: {wf.name}'
assert len(wf.steps) == 3, f'Expected 3 steps, got {len(wf.steps)}'
print(f'Workflow loaded: {wf.name}, {len(wf.steps)} steps, budget={wf.max_hours}h')
print('PASS')
"

# 6. Systemd service file exists
echo "[6/7] Systemd check..."
if test -f ~/.config/systemd/user/sparky-research.service; then
    echo "PASS: service file exists"
else
    echo "WARN: sparky-research.service not found (non-blocking)"
fi

# 7. Sparky CLI responds
echo "[7/7] CLI check..."
if scripts/sparky status 2>/dev/null; then
    echo "PASS: sparky status succeeded"
else
    echo "WARN: sparky status failed (service may not be running — OK for smoke test)"
fi

echo ""
echo "=== All smoke tests passed ==="
