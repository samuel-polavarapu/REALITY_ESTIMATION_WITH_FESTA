#!/bin/bash

# Test FESTA evaluation with probability-based prompts for 2 samples

set -e  # Exit on error

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="output/api_run/logs"
LOG_FILE="${LOG_DIR}/festa_probability_test_${TIMESTAMP}.log"

echo "================================================================================"
echo "FESTA PROBABILITY-BASED PROMPT TEST - 2 SAMPLES"
echo "================================================================================"
echo "Start Time: $(date)"
echo "Test: Probability-based top-k inference with k=4, n_samples=5"
echo "Log file: ${LOG_FILE}"
echo "================================================================================"
echo ""

# Ensure output directories exist
mkdir -p output/api_run
mkdir -p ${LOG_DIR}
mkdir -p output/api_run/visualizations
mkdir -p output/api_run/generated_samples
mkdir -p reports

# Verify NUM_SAMPLES is set to 2
echo "Verifying configuration..."
grep "NUM_SAMPLES" .env
echo ""

# Check GPU availability
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
    echo ""
fi

# Run the evaluation with probability-based prompts
echo "Starting FESTA evaluation with probability-based prompts..."
echo "This will process 2 samples for testing."
echo ""

# Run with GPU support and tee output to log file
CUDA_VISIBLE_DEVICES=0 python3 src/festa_with_apis.py 2>&1 | tee "${LOG_FILE}"

# Check if the run completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "TEST COMPLETED SUCCESSFULLY"
    echo "================================================================================"
    echo "End Time: $(date)"
    echo ""
    echo "Generated files:"
    echo ""

    # Show latest reports
    echo "Latest JSON Report:"
    ls -lth reports/*.json 2>/dev/null | head -1 || echo "  No JSON reports found"
    echo ""

    echo "Latest CSV Files:"
    ls -lth output/api_run/*.csv 2>/dev/null | head -5 || echo "  No CSV files found"
    echo ""

    echo "Visualizations:"
    ls -1 output/api_run/visualizations/*.png 2>/dev/null | wc -l | xargs echo "  Total PNG files:"
    echo ""

    echo "Generated Samples:"
    ls -1 output/api_run/generated_samples/* 2>/dev/null | wc -l | xargs echo "  Total sample files:"
    echo ""

    echo "Log file: ${LOG_FILE}"
    echo ""

    # Show key metrics from the latest report
    echo "Key Metrics:"
    LATEST_REPORT=$(ls -t reports/*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_REPORT" ]; then
        python3 -c "
import json
with open('$LATEST_REPORT') as f:
    data = json.load(f)
    metrics = data.get('metrics', {})
    print(f\"  AUROC: {metrics.get('auroc', 'N/A')}\")
    print(f\"  Accuracy: {metrics.get('accuracy', 'N/A')}\")
    print(f\"  F1-Score: {metrics.get('f1_score', 'N/A')}\")
    print(f\"  Total predictions: {metrics.get('total_predictions', 'N/A')}\")
" 2>/dev/null || echo "  Could not parse metrics"
    fi

    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "TEST FAILED"
    echo "================================================================================"
    echo "Check log file for details: ${LOG_FILE}"
    echo "================================================================================"
    exit 1
fi

