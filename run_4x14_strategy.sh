#!/bin/bash

# Run FESTA with 4x14 combination strategy for 3 samples - GPU Optimized
echo "=========================================="
echo "FESTA 4x14 Strategy - GPU Optimized"
echo "=========================================="
echo "RTX 5090 24GB - Maximum Utilization"
echo "Processing 3 samples"
echo "FES: 4 text × 14 image = 56 combinations"
echo "FCS: 4 text × 14 image = 56 combinations"
echo "Total per sample: 112 combinations"
echo "Documentation: DISABLED"
echo "=========================================="

python src/festa_4x14_gpu.py 2>&1 | tee logs/festa_4x14_gpu_$(date +%Y%m%d_%H%M%S).log

