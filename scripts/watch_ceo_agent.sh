#!/bin/bash
# Live monitoring script for CEO agent
# Usage: ./scripts/watch_ceo_agent.sh

WATCH_INTERVAL=5  # seconds

echo "════════════════════════════════════════════════════════════════════════════"
echo "CEO AGENT LIVE MONITOR"
echo "════════════════════════════════════════════════════════════════════════════"
echo "Watching activity logs and MLflow runs every ${WATCH_INTERVAL}s"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "CEO AGENT LIVE MONITOR - $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""

    # 1. Recent activity
    echo "📋 RECENT ACTIVITY (Last 5 events)"
    echo "────────────────────────────────────────────────────────────────────────────"
    if [ -f "logs/agent_activity/ceo_$(date -u +%Y-%m-%d).jsonl" ]; then
        tail -5 "logs/agent_activity/ceo_$(date -u +%Y-%m-%d).jsonl" | jq -r '"[" + .timestamp[11:19] + "] " + .action_type + " → " + (.task // .phase // "N/A")'

        # Show last event details
        echo ""
        echo "🔍 LAST EVENT DETAILS"
        echo "────────────────────────────────────────────────────────────────────────────"
        tail -1 "logs/agent_activity/ceo_$(date -u +%Y-%m-%d).jsonl" | jq -r '
            "Time: " + .timestamp[11:19] +
            "\nAction: " + .action_type +
            "\nTask: " + (.task // .phase // "N/A") +
            "\nDescription: " + (.description // .conclusion // "N/A")
        '
    else
        echo "No activity log found for today"
    fi

    echo ""
    echo "📊 DATA STATUS"
    echo "────────────────────────────────────────────────────────────────────────────"
    if [ -f "data/processed/feature_matrix_btc.parquet" ]; then
        stat -c "Feature matrix: %s bytes, modified %y" data/processed/feature_matrix_btc.parquet
    else
        echo "❌ No feature matrix found"
    fi

    echo ""
    echo "🔄 MLFLOW RUNS (Last 5)"
    echo "────────────────────────────────────────────────────────────────────────────"
    find mlruns -name "meta.yaml" -type f -mmin -120 | head -5 | while read meta; do
        echo "  $(stat -c '%y' "$meta" | cut -d' ' -f2 | cut -d'.' -f1) - $(dirname "$meta" | rev | cut -d'/' -f1 | rev | cut -c1-8)"
    done

    echo ""
    echo "⏱️  IDLE TIME"
    echo "────────────────────────────────────────────────────────────────────────────"
    if [ -f "logs/agent_activity/ceo_$(date -u +%Y-%m-%d).jsonl" ]; then
        LAST_TS=$(tail -1 "logs/agent_activity/ceo_$(date -u +%Y-%m-%d).jsonl" | jq -r '.timestamp')
        LAST_EPOCH=$(date -d "$LAST_TS" +%s 2>/dev/null || echo "0")
        NOW_EPOCH=$(date -u +%s)
        IDLE_SEC=$((NOW_EPOCH - LAST_EPOCH))

        if [ $IDLE_SEC -lt 60 ]; then
            echo "✅ Active (${IDLE_SEC}s ago)"
        elif [ $IDLE_SEC -lt 300 ]; then
            echo "⚠️  Low activity (${IDLE_SEC}s ago)"
        else
            IDLE_MIN=$((IDLE_SEC / 60))
            echo "⚠️  IDLE (${IDLE_MIN}m ${$((IDLE_SEC % 60))}s ago)"
        fi
    fi

    echo ""
    echo "────────────────────────────────────────────────────────────────────────────"
    echo "Next refresh in ${WATCH_INTERVAL}s... (Ctrl+C to stop)"

    sleep $WATCH_INTERVAL
done
