#!/bin/bash
# Monitor FESTA 143 sample run progress
LOG_FILE=$(ls -t output/api_run/festa_all_143_samples_*.log 2>/dev/null | head -1)
if [ -z "$LOG_FILE" ]; then
    echo "No log file found"
    exit 1
fi
echo "================================================================================"
echo "FESTA 143 SAMPLES - PROGRESS MONITOR"
echo "================================================================================"
echo "Log file: $LOG_FILE"
echo ""
# Get current sample being processed
CURRENT=$(grep -oP "PROCESSING SAMPLE \K\d+/143" "$LOG_FILE" | tail -1)
echo "Current Progress: $CURRENT"
# Count completed samples
COMPLETED=$(grep -c "Sample Summary:" "$LOG_FILE")
echo "Completed Samples: $COMPLETED / 143"
# Calculate percentage
if [ "$COMPLETED" -gt 0 ]; then
    PERCENT=$(echo "scale=2; $COMPLETED * 100 / 143" | bc)
    echo "Percentage Complete: ${PERCENT}%"
fi
# Show recent activity
echo ""
echo "Recent Activity (last 10 lines):"
echo "--------------------------------------------------------------------------------"
tail -10 "$LOG_FILE"
echo "================================================================================"
