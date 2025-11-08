#!/bin/bash

# Run FESTA for 2 samples and generate CSV reports ONLY

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="output/api_run/logs"
LOG_FILE="${LOG_DIR}/festa_csv_only_${TIMESTAMP}.log"

echo "================================================================================"
echo "FESTA EVALUATION - 2 SAMPLES - CSV OUTPUT ONLY"
echo "================================================================================"
echo "Start Time: $(date)"
echo "Log file: ${LOG_FILE}"
echo "================================================================================"
echo ""

# Ensure directories exist
mkdir -p output/api_run
mkdir -p ${LOG_DIR}
mkdir -p output/api_run/csv_reports
mkdir -p reports

# Verify configuration
echo "Configuration:"
grep "NUM_SAMPLES" .env || echo "NUM_SAMPLES=2" >> .env
grep "SKIP_SAMPLES" .env || echo "SKIP_SAMPLES=0" >> .env
echo ""

# Run evaluation
echo "Running FESTA evaluation..."
CUDA_VISIBLE_DEVICES=0 python3 src/festa_with_apis.py 2>&1 | tee "${LOG_FILE}"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "EVALUATION COMPLETED - GENERATING CSV REPORTS"
    echo "================================================================================"

    # Find the latest JSON report
    LATEST_JSON=$(ls -t reports/festa_report_*.json 2>/dev/null | head -1)

    if [ -n "$LATEST_JSON" ]; then
        echo "Processing results from: $LATEST_JSON"

        # Generate calibrated charts and extract CSV data
        python3 src/analyze_results.py "$LATEST_JSON" "output/api_run/csv_reports"

        # List generated CSV files
        echo ""
        echo "CSV Files Generated:"
        ls -lh output/api_run/*.csv 2>/dev/null | tail -10
        echo ""

        echo "Calibrated Charts:"
        ls -lh output/api_run/csv_reports/*.png 2>/dev/null | wc -l | xargs echo "  Total charts:"

        echo ""
        echo "================================================================================"
        echo "CSV REPORTS SUMMARY"
        echo "================================================================================"

        # Display metrics summary
        LATEST_METRICS_CSV=$(ls -t output/api_run/metrics_summary_*.csv 2>/dev/null | head -1)
        if [ -n "$LATEST_METRICS_CSV" ]; then
            echo "Metrics Summary ($LATEST_METRICS_CSV):"
            echo ""
            cat "$LATEST_METRICS_CSV"
            echo ""
        fi

        # Display master metrics
        LATEST_MASTER_CSV=$(ls -t output/api_run/master_metrics_*.csv 2>/dev/null | head -1)
        if [ -n "$LATEST_MASTER_CSV" ]; then
            echo "Master Metrics ($LATEST_MASTER_CSV):"
            echo ""
            cat "$LATEST_MASTER_CSV"
            echo ""
        fi

        echo "================================================================================"
        echo "ALL CSV FILES AVAILABLE IN: output/api_run/"
        echo "================================================================================"
    else
        echo "WARNING: No JSON report found, CSV generation skipped"
    fi

    echo "Completed: $(date)"
else
    echo ""
    echo "ERROR: Evaluation failed"
    exit 1
fi

