#!/bin/bash

# Continue run with nohup to prevent termination
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="output/festa_143_continue_${TIMESTAMP}.log"

echo "Starting continued run from sample 37 with nohup..."
echo "Log: $LOG_FILE"

# Update .env to ensure correct values
cat > .env << 'EOF'


# Number of samples to process
NUM_SAMPLES=107
SKIP_SAMPLES=36
EOF

# Run with nohup
nohup python -u src/festa_with_apis.py > "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > output/current_run.pid
echo "Started with PID: $PID"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check status:"
echo "  ps aux | grep $PID"

