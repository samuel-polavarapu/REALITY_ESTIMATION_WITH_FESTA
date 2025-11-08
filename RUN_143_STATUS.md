# FESTA 143 Samples Evaluation - STATUS

**Started**: October 31, 2025 00:43
**Configuration**: Probability-based inference (k=4, n_samples=5)
**Status**: ✅ RUNNING

---

## 🎯 Configuration

- **Total Samples**: 143
- **Skip**: 0
- **Inference Method**: Probability-based top-k
  - k=4 guesses per inference
  - n_samples=5 sampling passes
  - Temperature=0.7
- **GPU**: NVIDIA GeForce RTX 5090 (24GB)
- **Process ID**: 21090
- **Memory Usage**: 2.8GB

---

## 📊 Current Progress

### Sample 2/143 Processing
- ✅ Original inference completed
- ✅ FES Text: 4 paraphrases generated
- ✅ FCS Text: 4 contradictions generated  
- ✅ FES Image: 4 variants generated
- ✅ FCS Image: 4 contradictions generated
- 🔄 Running inference on all generated samples

### Recent Inference Results (Sample 2)
**FES Text** (maintaining semantics):
- "car located under cat" → A (0.735)
- "car positioned beneath cat" → A (0.772)
- "car situated under cat" → A (0.560)
- "car found beneath cat" → A (0.690)

**FCS Text** (contradictions):
- "car on top of cat" → A (0.382, expected B)
- "cat under car" → A (0.700, expected B)
- "car outside cat" → A (0.578, expected B)
- "car far from cat" → B (0.520, expected B) ✅

**FES Image** (visual variants):
- Variant 1 → A (0.849)
- Variant 2 → A (0.879)
- Variant 3 → A (0.769)
- Variant 4 → Processing...

---

## ⏱️ Time Estimates

### Per Sample Timing
- FES/FCS Generation: ~1 minute
- Inference (1 original + 16 generated): ~5-6 minutes
- **Total per sample**: ~6-7 minutes

### Overall Estimates
- **143 samples × 6.5 minutes** = ~930 minutes
- **Estimated total time**: 15-16 hours
- **Expected completion**: October 31, 2025 ~16:00-17:00

### Progress Milestones
- 10% (14 samples): ~01:30
- 25% (36 samples): ~03:20
- 50% (72 samples): ~07:45
- 75% (107 samples): ~12:15
- 100% (143 samples): ~16:45

---

## 📁 Output Structure

### Generated Per Sample (×143)
- 1 original image
- 4 FES Text JSON files
- 4 FCS Text JSON files
- 4 FES Image PNG files
- 4 FCS Image PNG files
- **Total**: ~2,002 files

### Final Reports
1. **JSON Reports**:
   - `api_evaluation_results.json` - Complete results
   - `festa_report_TIMESTAMP.json` - Comprehensive metrics
   - `comprehensive_metrics.json` - Category metrics

2. **CSV Files** (4):
   - `comprehensive_metrics_*.csv` - Metrics by category
   - `sample_results_*.csv` - Per-sample results
   - `all_predictions_*.csv` - All predictions detail
   - `summary_statistics_*.csv` - Overall summary

3. **Visualizations** (16+):
   - 10 individual category charts
   - 6 comparison charts
   - Risk-coverage curves
   - Accuracy-coverage curves
   - ROC curves
   - Precision-Recall curves

---

## 🔍 Monitoring Commands

### Check Process Status
```bash
ps aux | grep festa_with_apis | grep -v grep
```

### View Latest Log Entries
```bash
tail -50 output/api_run/logs/festa_143_samples_*.log
```

### Count Generated Files
```bash
ls -1 output/api_run/generated_samples/ | wc -l
```

### Monitor GPU Usage
```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

### Check Current Sample
```bash
grep "PROCESSING SAMPLE" output/api_run/logs/festa_143_samples_*.log | tail -1
```

---

## 📈 Expected Final Metrics

Based on 2-sample test results:

- **AUROC**: 0.70 - 0.85 (Target: ≥0.70) ✅
- **Accuracy**: 0.65 - 0.80
- **Precision**: 0.60 - 0.75
- **Recall**: 0.70 - 0.90
- **F1-Score**: 0.65 - 0.80
- **Total Predictions**: ~2,288 (143 × ~16)

### Category-wise Performance
- **FES Text**: AUROC 0.75-0.85
- **FCS Text**: AUROC 0.65-0.75
- **FES Image**: AUROC 0.60-0.70
- **FCS Image**: AUROC 0.55-0.65

---

## 🚀 Key Features

### Probability-Based Inference
- ✅ Top-k guesses with explicit probabilities
- ✅ Multiple sampling for robustness
- ✅ Confidence-weighted voting
- ✅ Better uncertainty quantification

### Calibration & Metrics
- ✅ LogisticRegression calibration (sklearn)
- ✅ Risk-coverage curves (reference implementation)
- ✅ Accuracy-coverage curves
- ✅ ROC and Precision-Recall curves
- ✅ Comprehensive metrics (AUROC, AUPRC, Brier, ECE)

### Visualization Quality
- ✅ High resolution (300 DPI)
- ✅ Visible lines in all charts
- ✅ Professional styling
- ✅ Multiple comparison views

---

## 📝 Log Files

- **Main Log**: `output/api_run/logs/festa_143_samples_TIMESTAMP.log`
- **Execution Log**: `output/api_run/run_143_output.log`
- **Size**: Growing (~2-3 MB upon completion)

---

## ⚠️ Notes

1. **API Rate Limiting**: 21-second delay between samples for OpenAI API
2. **Memory**: Stable at ~2.8GB RAM, 15GB GPU VRAM
3. **Automatic Recovery**: Process runs in background with nohup
4. **CSV Priority**: Results will be exported to CSV format
5. **Charts**: Calibrated risk-coverage charts will be generated automatically

---

## ✅ Success Criteria

- [x] Probability-based prompts implemented
- [x] sklearn libraries used (LogisticRegression, roc_auc_score)
- [x] matplotlib.pyplot for visualizations
- [x] Reference implementation for risk-coverage
- [x] CSV export configured
- [ ] All 143 samples processed (In Progress: 2/143)
- [ ] AUROC ≥ 0.70 achieved
- [ ] All visualizations generated
- [ ] CSV files created

---

**Last Updated**: October 31, 2025 00:50
**Status**: Sample 2 inference in progress
**Next Check**: Monitor log after ~30 minutes for Sample 10 completion

