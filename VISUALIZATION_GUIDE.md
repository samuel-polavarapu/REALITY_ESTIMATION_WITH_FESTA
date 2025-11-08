# FESTA Visualization Guide

## Understanding Risk-Coverage and Accuracy-Coverage Curves

### What Are These Curves?

These curves help you understand model performance at different confidence thresholds. They answer: **"How accurate is my model when I only keep predictions it's confident about?"**

---

## Risk-Coverage Curves

### What They Show:
- **X-axis (Coverage):** Proportion of predictions kept (0 to 1)
- **Y-axis (Risk):** Error rate on kept predictions (0 to 1)
- **Lower is better** - you want low risk maintained as coverage increases

### How to Read:
```
Risk
1.0 |     X (bad - high risk)
    |    /
0.5 |   /
    |  /__________ (good - low risk maintained)
0.0 |_____________
    0.0    0.5    1.0  Coverage
```

### Interpretation:
- **Ideal:** Flat line at low risk (model is consistently accurate)
- **Bad:** Rising line (model makes more errors on low-confidence predictions)
- **Use Case:** Set confidence threshold where risk becomes acceptable

### Generated Charts:
1. **FES Risk-Coverage** - Risk for Functionally Equivalent Samples
   - Should show LOW risk (FES should maintain same answer)
   
2. **FCS Risk-Coverage** - Risk for Functionally Contradictory Samples
   - Should show risk changes (FCS flips spatial relations)
   
3. **FESTA Risk-Coverage** - Combined FES + FCS
   - Overall system reliability
   
4. **Output Risk-Coverage** - Original predictions only
   - Baseline model performance
   
5. **FES Text Risk-Coverage** - Text paraphrases only
   
6. **FES Image Risk-Coverage** - Image perturbations only
   
7. **FCS Text Risk-Coverage** - Text contradictions only
   
8. **FCS Image Risk-Coverage** - Image flips only

---

## Accuracy-Coverage Curves

### What They Show:
- **X-axis (Coverage):** Proportion of predictions kept (0 to 1)
- **Y-axis (Accuracy):** Accuracy on kept predictions (0 to 1)
- **Higher is better** - you want high accuracy maintained as coverage increases

### How to Read:
```
Accuracy
1.0 |_____________ (good - high accuracy maintained)
    |
0.5 |      \
    |       \
0.0 |        X (bad - low accuracy)
    |_____________
    0.0    0.5    1.0  Coverage
```

### Interpretation:
- **Ideal:** Flat line at high accuracy
- **Bad:** Dropping line (accuracy decreases as you include more predictions)
- **Use Case:** Find optimal coverage for target accuracy

### Generated Charts:
1. **FES Accuracy-Coverage** - Accuracy on FES samples
   - Should show HIGH accuracy (semantic equivalence)
   
2. **FCS Accuracy-Coverage** - Accuracy on FCS samples
   - May show lower accuracy (flipped relations)
   
3. **FESTA Accuracy-Coverage** - Combined performance
   
4. **Output Accuracy-Coverage** - Baseline model
   
5. **FES Text Accuracy-Coverage** - Text paraphrases
   
6. **FES Image Accuracy-Coverage** - Image variants
   
7. **FCS Text Accuracy-Coverage** - Text contradictions
   
8. **FCS Image Accuracy-Coverage** - Image flips

---

## Example Analysis

### Scenario 1: Well-Calibrated Model
```
FES Accuracy-Coverage:
1.0 |________________  (flat at 100%)
    |
0.0 |________________
    0.0           1.0

FCS Accuracy-Coverage:
1.0 |
    |     _________    (high for confident predictions)
0.5 |    /
    |___/
0.0 |________________
    0.0           1.0
```

**Interpretation:**
- FES: Model consistently correct on equivalent samples ✅
- FCS: Model can detect contradictions when confident ✅
- **Action:** Deploy with confidence threshold at 0.7

### Scenario 2: Poorly-Calibrated Model
```
FES Accuracy-Coverage:
1.0 |
    |     \________    (drops quickly)
0.5 |      \
    |       \
0.0 |________\_______
    0.0           1.0

FCS Accuracy-Coverage:
1.0 |
    |                  (random performance)
0.5 |~~~~~~~~~~~~~~~~
    |
0.0 |________________
    0.0           1.0
```

**Interpretation:**
- FES: Model not consistent on equivalent samples ❌
- FCS: Cannot distinguish contradictions ❌
- **Action:** Retrain model or use different architecture

---

## Practical Use Cases

### 1. Setting Confidence Thresholds

**Goal:** Maintain 90% accuracy

**Steps:**
1. Look at Accuracy-Coverage curve
2. Find where accuracy = 0.9 on Y-axis
3. Draw horizontal line to curve
4. Drop vertical line to X-axis
5. That's your required coverage (and confidence threshold)

### 2. Comparing Text vs Image Performance

**Compare:**
- FES Text Accuracy-Coverage
- FES Image Accuracy-Coverage

**If Text > Image:**
- Model better at handling text variations
- May need better image augmentation

**If Image > Text:**
- Model better at visual robustness
- May need better text paraphrasing

### 3. Evaluating FESTA Approach

**Look at FESTA curves (combined FES + FCS):**

**Good FESTA:**
- High accuracy on FES (equivalence preserved)
- Clear separation on FCS (contradictions detected)
- Low risk overall

**Bad FESTA:**
- Mixed performance on FES
- Cannot distinguish FCS
- High risk

---

## Metrics Reference

### Coverage
**Definition:** Proportion of predictions kept after applying confidence threshold

**Formula:** `coverage = (predictions above threshold) / (total predictions)`

**Example:** 
- 100 predictions, threshold = 0.8
- 70 predictions have confidence ≥ 0.8
- Coverage = 70/100 = 0.7

### Risk
**Definition:** Error rate on kept predictions

**Formula:** `risk = (incorrect predictions) / (total kept predictions)`

**Example:**
- 70 predictions kept
- 10 are incorrect
- Risk = 10/70 = 0.143 (14.3%)

### Accuracy
**Definition:** Correct rate on kept predictions

**Formula:** `accuracy = (correct predictions) / (total kept predictions)`

**Example:**
- 70 predictions kept
- 60 are correct
- Accuracy = 60/70 = 0.857 (85.7%)

**Note:** accuracy = 1 - risk

---

## Visualization Files

All charts saved in: `output/api_run/visualizations/`

**Format:** PNG, 300 DPI, 10x6 inches
**Style:** White grid background, clear labels
**Legend:** Shows data series names

### File Naming Convention:
```
{category}_{metric}_coverage.png

Examples:
- fes_risk_coverage.png
- festa_accuracy_coverage.png
- fcs_text_risk_coverage.png
```

---

## Advanced Analysis

### 1. Area Under Curve (AUC)

Calculate area under Risk-Coverage or Accuracy-Coverage curves:
- Lower AUC for Risk = Better (less risk overall)
- Higher AUC for Accuracy = Better (more accuracy overall)

### 2. Selective Prediction

Use curves to implement selective prediction:
```python
# Reject low-confidence predictions
if confidence < threshold:
    return "UNSURE"
else:
    return prediction
```

Set threshold where Risk-Coverage curve shows acceptable risk.

### 3. Model Comparison

Generate curves for multiple models and overlay:
- Better model = higher accuracy, lower risk
- Compare at same coverage levels

---

## Troubleshooting

### Issue: Flat Lines at 0 or 1
**Cause:** All predictions have same confidence
**Solution:** Check model calibration, may need temperature scaling

### Issue: Jagged/Noisy Curves
**Cause:** Too few samples
**Solution:** Run with more samples (use all 143)

### Issue: Missing Curves
**Cause:** No predictions for that category (e.g., FCS Image)
**Solution:** Ensure inference runs on image samples, not just text

---

## Summary

### Key Takeaways:

1. **Risk-Coverage:** Lower is better, shows error rate
2. **Accuracy-Coverage:** Higher is better, shows correct rate
3. **Coverage:** Trade-off between quantity and quality
4. **Use:** Set confidence thresholds for deployment
5. **Compare:** Text vs Image, FES vs FCS performance

### Best Practices:

- ✅ Always check FES accuracy (should be high)
- ✅ Compare text and image separately
- ✅ Use risk curves for safety-critical applications
- ✅ Use accuracy curves for performance optimization
- ✅ Set thresholds based on application requirements

---

## References

- **FESTA Paper:** Functionally Equivalent Samples and Testing Approach
- **Calibration:** Temperature scaling for better confidence estimates
- **Selective Prediction:** Reject low-confidence predictions
- **Risk-Coverage Trade-off:** More coverage = potentially more risk

---

**All visualizations available in:** `output/api_run/visualizations/`

**Comprehensive metrics in:** `output/api_run/comprehensive_metrics.json`

