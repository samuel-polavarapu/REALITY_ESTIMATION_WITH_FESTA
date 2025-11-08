#!/bin/bash

# Run FESTA evaluation for all 143 samples with GPU
# Ensures proper logging, GPU usage, and comprehensive metrics generation

set -e  # Exit on error

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="output/api_run/logs"
LOG_FILE="${LOG_DIR}/festa_143_samples_${TIMESTAMP}.log"

echo "================================================================================"
echo "FESTA EVALUATION - ALL 143 SAMPLES WITH GPU"
echo "================================================================================"
echo "Start Time: $(date)"
echo "GPU: NVIDIA GeForce RTX 5090"
echo "Log file: ${LOG_FILE}"
echo "================================================================================"
echo ""

# Ensure output directories exist
mkdir -p output/api_run
mkdir -p ${LOG_DIR}
mkdir -p output/api_run/visualizations
mkdir -p output/api_run/generated_samples
mkdir -p reports

# Check GPU availability
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
    echo ""
fi

# Verify Python environment
echo "Python environment:"
python3 --version
echo ""

# Check for required environment variables
echo "Checking environment configuration..."
if [ -f .env ]; then
    echo "✓ .env file found"
    # Verify NUM_SAMPLES is set to 143
    grep "NUM_SAMPLES" .env || echo "NUM_SAMPLES not set in .env"
    grep "SKIP_SAMPLES" .env || echo "SKIP_SAMPLES not set in .env"
else
    echo "ERROR: .env file not found!"
    exit 1
fi
echo ""

# Run the evaluation with GPU and comprehensive logging
echo "Starting FESTA evaluation..."
echo "This will process all 143 samples and may take several hours."
echo "Progress will be logged to: ${LOG_FILE}"
echo ""

# Run with GPU support and tee output to log file
CUDA_VISIBLE_DEVICES=0 python3 src/festa_with_apis.py 2>&1 | tee "${LOG_FILE}"

# Check if the run completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "EVALUATION COMPLETED SUCCESSFULLY"
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
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "EVALUATION FAILED"
    echo "================================================================================"
    echo "Check log file for details: ${LOG_FILE}"
    echo "================================================================================"
    exit 1
fi

