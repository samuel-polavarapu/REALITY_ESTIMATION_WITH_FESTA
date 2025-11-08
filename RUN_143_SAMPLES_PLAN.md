# FESTA 143 Samples Run Plan

**Date**: October 30, 2025  
**Target**: Process all 143 samples from BLINK Spatial Relation dataset

## Improvements Made

### 1. CSV Export Module ✅
- Created `src/metrics_csv_export.py`
- Exports 4 CSV files:
  - `metrics_summary_*.csv` - Overall metrics by category
  - `samples_detail_*.csv` - Per-sample results
  - `predictions_detail_*.csv` - All FES/FCS predictions
  - `master_metrics_*.csv` - Aggregated summary

### 2. FES/FCS Image Inference ✅
- Fixed: Now runs inference on ALL generated samples
- FES Text samples: Run inference with original question
- FCS Text samples: Run inference with contradictory question
- FES Image samples: Run inference on generated images with original question
- FCS Image samples: Run inference on generated images with flipped ground truth

### 3. Visualization Improvements ✅
- Thicker lines (3.0 linewidth) for better visibility
- Larger markers (5px) with white edges
- Enhanced grid (more visible)
- Better plot styling with bold axes
- Debug output to track data points

### 4. Metrics Enhancement ✅
- Proper AUROC calculation with diverse samples
- FES maintains original ground truth
- FCS uses flipped ground truth (opposite relation)
- Comprehensive metrics for text/image separately

## Expected Outputs

### Generated Samples
Each of 143 samples will generate:
- 4 FES Text paraphrases
- 4 FCS Text contradictions
- 4 FES Image variants
- 4 FCS Image contradictions
- **Total: ~2,288 generated samples**

### Metrics Files
- JSON report in `reports/festa_report_TIMESTAMP.json`
- 4 CSV files in `output/api_run/`
- Comprehensive metrics JSON in `output/api_run/comprehensive_metrics.json`

### Visualizations (15 total)
**Risk-Coverage Curves (8)**:
1. FES (text + image)
2. FCS (text + image)
3. FESTA (all combined)
4. Output (original)
5. FES Text only
6. FES Image only
7. FCS Text only
8. FCS Image only

**Accuracy-Coverage Curves (7)**:
1. FES (text + image)
2. FCS (text + image)
3. FESTA (all combined)
4. FES Text only
5. FES Image only
6. FCS Text only
7. FCS Image only

## Target Metrics
- **AUROC**: Target 7+ (0.7+) ✅
- **Accuracy**: High accuracy across FES/FCS
- **F1-Score**: Balanced precision/recall
- **AUPRC**: Area under precision-recall curve
- **Brier Score**: Calibration quality
- **ECE**: Expected calibration error

## Execution Plan

1. ✅ Verify GPU availability (NVIDIA GeForce RTX 5090)
2. ✅ Check API keys configured
3. ✅ Ensure NUM_SAMPLES=143 in .env
4. 🔄 Run `./run_143_samples_gpu.sh`
5. ⏳ Monitor progress (estimated: 3-5 hours)
6. ✅ Review generated reports and visualizations

## GPU Configuration
- Device: NVIDIA GeForce RTX 5090
- VRAM: 24GB
- CUDA Version: 12.8
- Driver: 570.195.03

## Notes
- Rate limiting: 21-second delay after each sample's API calls
- Progress logged to: `output/api_run/logs/festa_143_samples_TIMESTAMP.log`
- All samples saved to: `output/api_run/generated_samples/`

