#!/bin/bash

# Run FESTA for all 143 samples with GPU - CSV output focused
# Estimated time: 4-6 hours

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="output/api_run/logs"
LOG_FILE="${LOG_DIR}/festa_143_samples_${TIMESTAMP}.log"

echo "================================================================================"
echo "FESTA EVALUATION - ALL 143 SAMPLES WITH PROBABILITY-BASED PROMPTS"
echo "================================================================================"
echo "Start Time: $(date)"
echo "Configuration: Probability-based inference (k=4, n_samples=5)"
echo "GPU: NVIDIA GeForce RTX 5090"
echo "Log file: ${LOG_FILE}"
echo "================================================================================"
echo ""

# Create necessary directories
mkdir -p output/api_run
mkdir -p ${LOG_DIR}
mkdir -p output/api_run/csv_final
mkdir -p reports

# Verify GPU
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  GPU check skipped"
echo ""

# Verify configuration
echo "Configuration:"
grep "NUM_SAMPLES" .env
echo ""

# Start evaluation
echo "Starting FESTA evaluation for 143 samples..."
echo "This will take approximately 4-6 hours"
echo ""

# Run with GPU
CUDA_VISIBLE_DEVICES=0 python3 src/festa_with_apis.py 2>&1 | tee "${LOG_FILE}"

# Check completion status
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "EVALUATION COMPLETED SUCCESSFULLY"
    echo "================================================================================"
    echo "End Time: $(date)"
    echo ""

    # Find latest report
    LATEST_JSON=$(ls -t output/api_run/api_evaluation_results.json 2>/dev/null | head -1)

    if [ -n "$LATEST_JSON" ]; then
        echo "Generating CSV reports..."
        python3 generate_csv_only.py "$LATEST_JSON" output/api_run/csv_final

        echo ""
        echo "Generating calibrated charts..."
        python3 src/analyze_results.py "$LATEST_JSON" output/api_run/calibrated_charts

        echo ""
        echo "================================================================================"
        echo "RESULTS SUMMARY"
        echo "================================================================================"

        # Show metrics
        python3 << EOF
import json
with open('$LATEST_JSON', 'r') as f:
    data = json.load(f)
    print(f"Samples Processed: {data['samples_processed']}")
    print(f"Total Predictions: {data['metrics']['total_predictions']}")
    print(f"AUROC: {data['metrics']['auroc']:.4f}")
    print(f"Accuracy: {data['metrics']['accuracy']:.4f}")
    print(f"F1-Score: {data['metrics']['f1_score']:.4f}")
    print(f"\nGenerated Samples:")
    print(f"  FES Text: {data['generated_samples_summary']['fes_text_total']}")
    print(f"  FES Image: {data['generated_samples_summary']['fes_image_total']}")
    print(f"  FCS Text: {data['generated_samples_summary']['fcs_text_total']}")
    print(f"  FCS Image: {data['generated_samples_summary']['fcs_image_total']}")
EOF

        echo ""
        echo "CSV Files:"
        ls -lh output/api_run/csv_final/*.csv 2>/dev/null

        echo ""
        echo "Charts:"
        ls -1 output/api_run/calibrated_charts/*.png 2>/dev/null | wc -l | xargs echo "  Total visualizations:"
    fi

    echo ""
    echo "================================================================================"
    echo "ALL FILES SAVED IN: output/api_run/"
    echo "Log: ${LOG_FILE}"
    echo "================================================================================"
else
    echo ""
    echo "ERROR: Evaluation failed - check log: ${LOG_FILE}"
    exit 1
fi

