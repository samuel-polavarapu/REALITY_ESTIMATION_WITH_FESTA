#!/bin/bash

# Run FESTA evaluation for all 143 samples with GPU
# This script ensures proper logging and captures all output

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="output/api_run/festa_all_143_samples_${TIMESTAMP}.log"

echo "Starting FESTA evaluation for all 143 samples..."
echo "Using GPU: NVIDIA GeForce RTX 5090"
echo "Log file: ${LOG_FILE}"
echo ""

# Ensure output directory exists
mkdir -p output/api_run

# Run the evaluation with GPU and log everything
python3 src/festa_with_apis.py 2>&1 | tee "${LOG_FILE}"

echo ""
echo "Evaluation complete! Log saved to: ${LOG_FILE}"
echo ""
echo "Check reports folder for detailed JSON report"
ls -lh reports/*.json | tail -1

