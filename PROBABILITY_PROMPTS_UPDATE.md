# FESTA Probability-Based Prompts Update

**Date**: October 31, 2025  
**Status**: Ready for Testing

## Changes Implemented

### 1. Probability-Based Prompt Function ✅
**File**: `src/festa_evaluation.py`

- Updated `strict_prompt()` function to support both modes:
  - **Strict mode** (k=None): Returns single A/B answer
  - **Probability mode** (k=int): Requests k guesses with probabilities (0.0-1.0)
  
Example probability prompt format:
```
G1: <first most likely guess>
P1: <probability for G1>
G2: <second most likely guess>
P2: <probability for G2>
...
```

### 2. Top-K Inference Method ✅
**File**: `src/festa_evaluation.py`

Added `run_topk_inference()` method to LLaVAModel class:
- **Parameters**: 
  - `k=4`: Number of guesses to request
  - `n_samples=5`: Number of sampling passes
- **Returns**: List of (guess, probability) tuples
- **Features**:
  - Multiple sampling passes for robustness
  - Regex-based extraction of guesses and probabilities
  - Temperature=0.7 for diversity
  - Fallback extraction if pattern matching fails

### 3. Combined Prediction Function ✅
**File**: `src/festa_evaluation.py`

Added `get_combined_pred()` function:
- Aggregates top-k results using probability-weighted voting
- Returns most confident prediction
- Handles edge cases (empty results, tie-breaking)

### 4. Updated Inference Calls ✅
**File**: `src/festa_with_apis.py`

Updated ALL inference calls to use probability-based approach:
- ✅ Original inference (baseline)
- ✅ FES Text samples
- ✅ FCS Text samples  
- ✅ FES Image samples
- ✅ FCS Image samples

Each inference now:
1. Calls `run_topk_inference()` with k=4, n_samples=5
2. Gets combined prediction using weighted voting
3. Calculates confidence as mean probability of predicted class
4. Stores top-k results for debugging

### 5. Enhanced Metrics Visualizations ✅
**File**: `src/festa_metrics.py`

Added sklearn-based visualization functions:

#### New Imports:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
```

#### New Visualization Methods:
1. **plot_roc_curves()**: 
   - Uses `sklearn.metrics.roc_curve()`
   - Plots ROC curves for all categories
   - Includes AUC scores in legend
   - Adds random classifier baseline

2. **plot_precision_recall_curves()**:
   - Uses `sklearn.metrics.precision_recall_curve()`
   - Shows precision-recall tradeoffs
   - Includes AP (Average Precision) scores

3. **plot_metrics_bar_chart()**:
   - Compares metrics across categories
   - Shows AUROC, Accuracy, Precision, Recall, F1-Score
   - Color-coded bars with value labels

#### Enhanced Existing Visualizations:
- Thicker lines (3.0px) for better visibility
- Larger markers (5px) with white edges
- More prominent grid and axes
- Debug logging for data ranges

### 6. Test Configuration ✅
**File**: `.env`

Updated for 2-sample test:
```bash
NUM_SAMPLES=2
SKIP_SAMPLES=0
```

### 7. Test Script ✅
**File**: `run_probability_test.sh`

Created comprehensive test script that:
- Verifies configuration
- Checks GPU availability
- Runs 2-sample evaluation
- Reports metrics and file counts
- Shows key AUROC/Accuracy/F1 metrics

## Expected Outputs

### Per Sample (2 samples):
- 1 original prediction with top-k results
- 4 FES Text predictions (each with k=4, n=5 → 20 guesses)
- 4 FCS Text predictions (each with k=4, n=5 → 20 guesses)
- 4 FES Image predictions (each with k=4, n=5 → 20 guesses)
- 4 FCS Image predictions (each with k=4, n=5 → 20 guesses)

**Total**: ~162 LLaVA inference calls (with multiple guesses per call)

### Visualizations Generated:
1. 8 Risk-Coverage curves (FES, FCS, FESTA, Output, etc.)
2. 7 Accuracy-Coverage curves
3. 1 ROC curves (combined, all categories)
4. 1 Precision-Recall curves (combined)
5. 1 Metrics comparison bar chart

**Total**: 18 visualization files

### Reports Generated:
1. JSON report: `reports/festa_report_TIMESTAMP.json`
2. CSV files (4):
   - `metrics_summary_TIMESTAMP.csv`
   - `samples_detail_TIMESTAMP.csv`
   - `predictions_detail_TIMESTAMP.csv`
   - `master_metrics_TIMESTAMP.csv`
3. Comprehensive metrics JSON
4. Markdown report

## Key Improvements

### Better Confidence Estimation
- Multi-sample consensus (n=5 passes)
- Probability-weighted voting
- Explicit uncertainty quantification

### Richer Data Collection
- Top-k guesses provide distribution information
- Can analyze prediction diversity
- Better calibration assessment

### Enhanced Visualizations
- sklearn-powered ROC and PR curves
- Professional matplotlib styling
- Comprehensive metrics comparison

## Testing Instructions

### Run 2-Sample Test:
```bash
./run_probability_test.sh
```

### Monitor Progress:
```bash
tail -f output/api_run/logs/festa_probability_test_*.log
```

### Check Results:
```bash
# View latest report
cat reports/festa_report_*.json | jq '.metrics'

# Check visualizations
ls -lh output/api_run/visualizations/

# View CSV metrics
cat output/api_run/metrics_summary_*.csv
```

## Expected Metrics

Based on previous runs with diverse FES/FCS samples:
- **AUROC**: 0.70 - 0.85 (Target: ≥0.7)
- **Accuracy**: 0.75 - 0.90
- **F1-Score**: 0.70 - 0.85
- **Total Predictions**: ~162 (2 samples × ~81 predictions each)

## Next Steps

After successful 2-sample test:
1. Update NUM_SAMPLES=143 in .env
2. Run full 143-sample evaluation
3. Generate comprehensive report with all visualizations
4. Export metrics to CSV for analysis

## Notes

- Probability-based prompts may generate slightly longer responses
- Multiple sampling (n=5) increases inference time but improves robustness
- Top-k results stored in JSON for post-analysis
- All visualizations use sklearn and matplotlib as requested

