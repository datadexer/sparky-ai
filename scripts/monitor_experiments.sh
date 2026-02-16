#!/bin/bash
# Monitor progress of comprehensive experiments

OUTPUT_FILE="/tmp/claude-1000/-home-akamath-sparky-ai/tasks/bd823e0.output"

echo "==================================="
echo "Experiment Progress Monitor"
echo "==================================="
echo ""

# Check if output file exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "❌ Output file not found: $OUTPUT_FILE"
    exit 1
fi

# Show current experiment
echo "📊 Current experiment:"
grep -E "\[[0-9]+/15\] Starting:" "$OUTPUT_FILE" | tail -n 1
echo ""

# Show recent completions
echo "✅ Recently completed:"
grep "✓ Experiment complete:" "$OUTPUT_FILE" | tail -n 5
echo ""

# Show any leakage detections
LEAKAGE_COUNT=$(grep -c "✗ LEAKAGE DETECTED" "$OUTPUT_FILE")
if [ $LEAKAGE_COUNT -gt 0 ]; then
    echo "⚠️  Leakage detected in $LEAKAGE_COUNT experiments:"
    grep "✗ LEAKAGE DETECTED" "$OUTPUT_FILE"
    echo ""
fi

# Show failures
FAILURE_COUNT=$(grep -c "✗ Experiment failed" "$OUTPUT_FILE")
if [ $FAILURE_COUNT -gt 0 ]; then
    echo "❌ Failed experiments: $FAILURE_COUNT"
    echo ""
fi

# Show total lines (rough progress indicator)
TOTAL_LINES=$(wc -l < "$OUTPUT_FILE")
echo "📈 Total output lines: $TOTAL_LINES"

# Show recent Sharpe results
echo ""
echo "📉 Recent Sharpe results:"
grep -E "Sharpe: [0-9\-\.]+" "$OUTPUT_FILE" | tail -n 10

echo ""
echo "==================================="
echo "Run 'tail -f $OUTPUT_FILE' to watch live"
