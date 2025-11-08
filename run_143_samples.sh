#!/bin/bash
# Run FESTA evaluation for all 143 samples

cd /data/sam/Kaggle/code/LLAVA-V5-2

echo "========================================="
echo "FESTA 143-Sample Evaluation Starting"
echo "========================================="
echo "Start Time: $(date)"
echo ""

# Set log file
LOGFILE="output/api_run/festa_143_samples_$(date +%Y%m%d_%H%M%S).log"

echo "Log file: $LOGFILE"
echo "Running evaluation..."
echo ""

# Run the evaluation
python3 -u src/festa_with_apis.py 2>&1 | tee "$LOGFILE"

echo ""
echo "========================================="
echo "Evaluation Complete!"
echo "End Time: $(date)"
echo "========================================="
echo ""
echo "Results location:"
echo "  - Visualizations: output/api_run/visualizations/"
echo "  - Metrics: output/api_run/comprehensive_metrics.json"
echo "  - Report: reports/festa_report_*.json"
echo "  - Log: $LOGFILE"

