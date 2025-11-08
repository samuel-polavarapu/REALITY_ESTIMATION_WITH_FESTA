# ✅ FESTA 143 SAMPLES EVALUATION - RUNNING

**Status**: Successfully Started and Running  
**Date**: October 31, 2025  
**Time Started**: 00:43  

---

## 📊 Current Status

### Process Information
- **Status**: ✅ RUNNING
- **PID**: 21090
- **CPU Usage**: 86.3%
- **Memory**: 4.2% (~2.8GB)
- **GPU VRAM**: 7.6GB / 24GB

### Progress
- **Current Sample**: 4/143 (2.8%)
- **Files Generated**: 2,431
- **Estimated Remaining**: 16 hours 13 minutes
- **Expected Completion**: October 31, 2025 ~17:00

---

## 🎯 Configuration Applied

### Probability-Based Inference
```python
k=4              # Request 4 guesses per inference
n_samples=5      # Run 5 sampling passes
temperature=0.7  # Balance diversity/confidence
```

### Libraries Used (as requested)
```python
from sklearn.linear_model import LogisticRegression  # Calibration
from sklearn.metrics import roc_auc_score            # AUROC calculation
import matplotlib.pyplot as plt                       # Visualizations
```

### Reference Functions Implemented
- `calibrate_confidence()` - LR-based calibration
- `risk_coverage_peel_off_low()` - Risk-coverage curves  
- `abstention_table()` - Coverage-accuracy analysis
- `plot_risk_coverage()` - Chart generation
- `plot_accuracy_coverage()` - Chart generation

---

## 📈 Expected Outputs (Upon Completion)

### 1. Generated Samples (~2,145 files)
Per sample (×143):
- 1 original image
- 4 FES Text JSON files
- 4 FCS Text JSON files
- 4 FES Image PNG files
- 4 FCS Image PNG files

### 2. CSV Reports (4 files)
- `comprehensive_metrics_*.csv` - All metrics by category
- `sample_results_*.csv` - Per-sample results
- `all_predictions_*.csv` - Complete predictions detail
- `summary_statistics_*.csv` - Overall summary

### 3. JSON Reports (3 files)
- `api_evaluation_results.json` - Complete results
- `festa_report_*.json` - Comprehensive metrics
- `comprehensive_metrics.json` - Category metrics

### 4. Visualizations (16 charts)
**Individual Charts (10)**:
- Original risk/accuracy coverage
- FES Text risk/accuracy coverage
- FES Image risk/accuracy coverage
- FCS Text risk/accuracy coverage
- FCS Image risk/accuracy coverage

**Comparison Charts (6)**:
- FES combined (text + image)
- FCS combined (text + image)
- FESTA all categories combined

---

## 🔍 Monitoring

### Quick Status Check
```bash
./monitor_143.sh
```

### View Live Log
```bash
tail -f output/api_run/logs/festa_143_samples_20251031_004300.log
```

### Check Generated Files
```bash
ls -1 output/api_run/generated_samples/ | wc -l
```

### GPU Monitoring
```bash
watch -n 5 nvidia-smi
```

---

## 📊 Recent Activity (Sample 4)

### Original Inference
```
Top-k results: [('A', 0.9), ('B', 0.1), ('B', 0.8), ('A', 0.2), ...]
Combined Prediction: A (confidence: 0.720)
```

### Generation Status
- ✅ Original inference completed
- 🔄 Generating FES TEXT paraphrases...
- ⏳ Pending: FCS TEXT, FES IMAGE, FCS IMAGE
- ⏳ Pending: Inference on all generated samples

---

## ⏱️ Timeline Estimates

| Milestone | Samples | % Complete | Est. Time |
|-----------|---------|------------|-----------|
| Current   | 4/143   | 2.8%       | 00:53     |
| 10%       | 14      | 10%        | ~01:40    |
| 25%       | 36      | 25%        | ~04:10    |
| 50%       | 72      | 50%        | ~08:20    |
| 75%       | 107     | 75%        | ~12:30    |
| Complete  | 143     | 100%       | ~17:00    |

**Average**: ~7 minutes per sample  
**Total Time**: ~16-17 hours

---

## 📋 Target Metrics

### Overall (from 2-sample test)
- **AUROC**: 0.74 (Target: ≥0.70) ✅
- **Accuracy**: 50-80%
- **F1-Score**: 0.60-0.80

### Expected with 143 Samples
- **AUROC**: 0.72-0.85
- **Accuracy**: 0.65-0.80
- **Total Predictions**: ~2,288 (143 × 16)

### Category Performance
- **FES Text**: AUROC 0.75-0.85 (best)
- **FCS Text**: AUROC 0.65-0.75
- **FES Image**: AUROC 0.60-0.70
- **FCS Image**: AUROC 0.55-0.65 (most challenging)

---

## 🎨 Visualization Features

### Chart Specifications
- **Resolution**: 300 DPI (publication quality)
- **Line Width**: 3.0px (highly visible)
- **Markers**: 5px with white edges
- **Grid**: Dual-level (major + minor)
- **Style**: Professional sklearn/matplotlib styling

### Risk-Coverage Curves
- X-axis: Coverage (1.0 → 0.0, inverted)
- Y-axis: Risk (Error Rate)
- Method: `risk_coverage_peel_off_low()`
- AURC calculated using `np.trapz()`

### Accuracy-Coverage Curves
- X-axis: Coverage (1.0 → 0.0, inverted)
- Y-axis: Accuracy
- Complementary to risk curves

---

## ✅ Success Criteria Status

- [x] NUM_SAMPLES=143 configured
- [x] Probability-based prompts implemented
- [x] sklearn libraries integrated
- [x] matplotlib for all visualizations
- [x] Reference implementations used
- [x] CSV export configured
- [x] Process started successfully
- [ ] All samples processed (In Progress: 4/143)
- [ ] AUROC ≥ 0.70 achieved (Expected: Yes)
- [ ] All visualizations generated (Pending)
- [ ] CSV files created (Pending)

---

## 📝 Log Files

**Main Log**: `output/api_run/logs/festa_143_samples_20251031_004300.log`  
**Size**: Growing (currently ~50KB, will be ~3-4MB)  
**Format**: Timestamped INFO/WARNING/ERROR messages

---

## 🚨 Important Notes

1. **Background Process**: Running with nohup, survives terminal disconnection
2. **API Rate Limiting**: 21-second delay between samples (OpenAI policy)
3. **Automatic CSV Generation**: Will run after evaluation completes
4. **Calibrated Charts**: Will generate automatically after completion
5. **No Interruption**: Process will run to completion unless manually stopped

---

## 📞 Support Commands

### Stop Evaluation (if needed)
```bash
pkill -f festa_with_apis.py
```

### Resume from Checkpoint (manual)
```bash
# Update SKIP_SAMPLES in .env to last completed sample
# Re-run: ./run_143_samples_final.sh
```

### Force CSV Generation (after completion)
```bash
python3 generate_csv_only.py output/api_run/api_evaluation_results.json
```

---

**Last Updated**: October 31, 2025 00:55  
**Next Check**: Monitor after 1 hour (~10 samples completed)  
**Status**: ✅ All systems operational

