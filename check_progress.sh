#!/bin/bash

# Quick progress check for FESTA probability test

echo "FESTA Probability Test - Progress Check"
echo "========================================"
echo ""

# Check if process is running
if ps aux | grep "festa_with_apis.py" | grep -v grep > /dev/null; then
    echo "✓ Process is running"
    PID=$(ps aux | grep "festa_with_apis.py" | grep -v grep | awk '{print $2}')
    echo "  PID: $PID"

    # Show memory usage
    MEM=$(ps aux | grep "festa_with_apis.py" | grep -v grep | awk '{print $6/1024 " MB"}')
    echo "  Memory: $MEM"
else
    echo "❌ Process not running"
fi

echo ""
echo "Latest log entries:"
echo "-------------------"
tail -15 output/api_run/logs/festa_probability_test_*.log 2>/dev/null || echo "No log file found"

echo ""
echo "Generated files count:"
ls -1 output/api_run/generated_samples/ 2>/dev/null | wc -l | xargs echo "  Samples:"

echo ""

