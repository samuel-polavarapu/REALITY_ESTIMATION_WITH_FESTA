#!/bin/bash

# Monitor FESTA 143 samples run progress

echo "FESTA 143 Samples Progress Monitor"
echo "=================================="
echo ""

# Check if process is running
PROCESS_COUNT=$(ps aux | grep "festa_with_apis.py" | grep -v grep | wc -l)

if [ $PROCESS_COUNT -eq 0 ]; then
    echo "❌ Process not running"
    echo ""
    echo "Last log entries:"
    if [ -d "output/api_run/logs" ]; then
        LATEST_LOG=$(ls -t output/api_run/logs/*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            echo "Log: $LATEST_LOG"
            tail -20 "$LATEST_LOG"
        fi
    fi
    exit 1
fi

echo "✓ Process is running (PID: $(ps aux | grep 'festa_with_apis.py' | grep -v grep | awk '{print $2}'))"
echo ""

# Check latest log file
LATEST_LOG=$(ls -t output/api_run/logs/*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "Latest log: $LATEST_LOG"
    echo ""

    # Extract progress info
    PROCESSING_SAMPLE=$(grep "PROCESSING SAMPLE" "$LATEST_LOG" | tail -1)
    if [ -n "$PROCESSING_SAMPLE" ]; then
        echo "Current progress:"
        echo "  $PROCESSING_SAMPLE"
    fi
    echo ""

    # Show last few log lines
    echo "Recent activity (last 15 lines):"
    echo "-----------------------------------"
    tail -15 "$LATEST_LOG"
    echo "-----------------------------------"
fi

echo ""
echo "Generated samples so far:"
SAMPLE_COUNT=$(ls output/api_run/generated_samples/ 2>/dev/null | wc -l)
echo "  Total files: $SAMPLE_COUNT"

echo ""
echo "GPU Usage:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free --format=csv,noheader 2>/dev/null || echo "  GPU info not available"

echo ""
echo "To check full log: tail -f output/api_run/logs/*.log"
echo "To stop process: pkill -f festa_with_apis.py"

