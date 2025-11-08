#!/bin/bash

# Quick status check for 143-sample run

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║       FESTA 143 SAMPLES - PROGRESS MONITOR                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if process is running
if ps aux | grep "festa_with_apis.py" | grep -v grep > /dev/null; then
    PID=$(ps aux | grep "festa_with_apis.py" | grep -v grep | awk '{print $2}')
    CPU=$(ps aux | grep "festa_with_apis.py" | grep -v grep | awk '{print $3}')
    MEM=$(ps aux | grep "festa_with_apis.py" | grep -v grep | awk '{print $4}')

    echo "✅ Status: RUNNING"
    echo "   PID: $PID | CPU: ${CPU}% | MEM: ${MEM}%"
else
    echo "❌ Status: NOT RUNNING"
    echo ""
    echo "Check if evaluation completed or check logs for errors."
    exit 1
fi

echo ""

# Get current sample
LATEST_LOG=$(ls -t output/api_run/logs/festa_143_samples_*.log 2>/dev/null | head -1)

if [ -n "$LATEST_LOG" ]; then
    echo "📊 Current Progress:"
    CURRENT=$(grep "PROCESSING SAMPLE" "$LATEST_LOG" | tail -1 | grep -oP '\d+/\d+' | head -1)
    if [ -n "$CURRENT" ]; then
        CURR_NUM=$(echo $CURRENT | cut -d'/' -f1)
        TOTAL=$(echo $CURRENT | cut -d'/' -f2)
        PERCENT=$((CURR_NUM * 100 / TOTAL))
        echo "   Sample: $CURRENT ($PERCENT%)"

        # Estimate time remaining
        ELAPSED_MIN=$((CURR_NUM * 7))
        REMAINING_MIN=$(((TOTAL - CURR_NUM) * 7))
        REMAINING_HR=$((REMAINING_MIN / 60))
        REMAINING_MIN_MOD=$((REMAINING_MIN % 60))

        echo "   Estimated time remaining: ${REMAINING_HR}h ${REMAINING_MIN_MOD}m"
    fi

    echo ""
    echo "📁 Generated Files:"
    FILE_COUNT=$(ls -1 output/api_run/generated_samples/ 2>/dev/null | wc -l)
    echo "   Total: $FILE_COUNT files"

    echo ""
    echo "🖥️  GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | \
        awk -F', ' '{printf "   GPU: %s | VRAM: %s / %s\n", $1, $2, $3}' || echo "   GPU info unavailable"

    echo ""
    echo "📝 Recent Activity:"
    tail -5 "$LATEST_LOG" | sed 's/^/   /'

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Log file: $LATEST_LOG"
    echo "To view full log: tail -f $LATEST_LOG"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "⚠️  No log file found"
fi

echo ""

