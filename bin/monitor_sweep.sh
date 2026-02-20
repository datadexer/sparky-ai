#!/bin/bash
# Monitor the running hyperparameter sweep

echo "=== Hyperparameter Sweep Monitor ==="
echo "Time: $(date)"
echo

# Check if process is running
PID=$(pgrep -f "python3 scripts/smart_hyperparam_sweep.py" | head -1)
if [ -n "$PID" ]; then
    echo "✓ Sweep is RUNNING (PID: $PID)"
    echo "  CPU: $(ps -p $PID -o %cpu= 2>/dev/null | tail -1 | tr -d ' ')%"
    echo "  Memory: $(ps -p $PID -o %mem= 2>/dev/null | tail -1 | tr -d ' ')%"
    echo "  Runtime: $(ps -p $PID -o etime= 2>/dev/null | tail -1 | tr -d ' ')"
else
    echo "✗ Sweep is NOT running"
fi
echo

# Check for intermediate results
if [ -f "results/validation/smart_sweep_intermediate.json" ]; then
    CONFIGS_DONE=$(python3 -c "import json; print(len(json.load(open('results/validation/smart_sweep_intermediate.json'))['configs']))" 2>/dev/null || echo "0")
    echo "✓ Intermediate results found: $CONFIGS_DONE/54 configs completed"

    # Check for configs beating baseline
    BEATS_BASELINE=$(python3 -c "
import json
data = json.load(open('results/validation/smart_sweep_intermediate.json'))
baseline = data.get('baseline', 1.062)
count = sum(1 for c in data['configs'] if c['mean_sharpe'] > baseline)
print(count)
" 2>/dev/null || echo "?")
    echo "  Configs beating baseline (${1:-1.062}): $BEATS_BASELINE"
else
    echo "⏳ Waiting for first results..."
fi
echo

# Check final results
if [ -f "results/validation/smart_hyperparam_sweep.json" ]; then
    echo "✅ SWEEP COMPLETE!"
    TOTAL=$(python3 -c "import json; print(json.load(open('results/validation/smart_hyperparam_sweep.json'))['total_configs'])" 2>/dev/null)
    BEATS=$(python3 -c "import json; print(json.load(open('results/validation/smart_hyperparam_sweep.json'))['configs_beat_baseline'])" 2>/dev/null)
    echo "  Total configs: $TOTAL"
    echo "  Configs beating baseline: $BEATS"
fi
