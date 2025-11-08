#!/bin/bash

# Monitor the progress of the 143 sample FESTA evaluation
LOG_FILE=$(ls -t /data/sam/Kaggle/code/LLAVA-V5-2/output/api_run/festa_all_143_samples_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "No active log file found!"
    exit 1
fi

echo "============================================================"
echo "FESTA Evaluation Progress Monitor"
echo "============================================================"
echo "Log file: $LOG_FILE"
echo ""

# Check if process is still running
if ps aux | grep "festa_with_apis.py" | grep -v grep > /dev/null; then
    echo "✓ Process is RUNNING"
    PID=$(ps aux | grep "festa_with_apis.py" | grep -v grep | awk '{print $2}')
    echo "  PID: $PID"
else
    echo "✗ Process is NOT running"
fi

echo ""
echo "Current Progress:"
echo "----------------"
# Get current sample being processed
CURRENT_SAMPLE=$(grep "PROCESSING SAMPLE" "$LOG_FILE" | tail -1)
echo "$CURRENT_SAMPLE"

# Count completed samples
COMPLETED=$(grep -c "Sample Summary:" "$LOG_FILE")
echo "Completed samples: $COMPLETED / 143"

# Calculate percentage
if [ $COMPLETED -gt 0 ]; then
    PERCENT=$((COMPLETED * 100 / 143))
    echo "Progress: $PERCENT%"
fi

echo ""
echo "Recent Activity (last 10 lines):"
echo "---------------------------------"
tail -10 "$LOG_FILE"

echo ""
echo "GPU Status:"
echo "-----------"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader

echo ""
echo "To view live log: tail -f $LOG_FILE"
echo "To check full report: cat reports/festa_report_*.json | python3 -m json.tool | grep metrics -A 15"

