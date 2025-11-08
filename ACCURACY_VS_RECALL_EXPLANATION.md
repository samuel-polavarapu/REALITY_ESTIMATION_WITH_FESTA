# Accuracy vs Recall Discrepancy Explanation

## Question: Why is Accuracy = 0.6 but Recall = 0.95?

---

## Executive Summary

The model demonstrates **HIGH RECALL (95.5%)** but **LOWER ACCURACY (60.7%)** because it is **BIASED towards predicting positive/yes answers (option "A")**. This means it catches most positive cases but creates many false alarms.

---

## Detailed Analysis from Previous Run

### Dataset: FES TEXT (Functionally Equivalent Samples - Text Perturbations)
- **Total Predictions**: 588 samples
- **Ground Truth Distribution**: 
  - Positive (A): 312 samples (53.1%)
  - Negative (B): 276 samples (46.9%)

---

## Confusion Matrix

```
                    PREDICTED
                  A           B
              ┌─────────┬─────────┐
            A │   298   │   14    │  312  (Actual Positives)
ACTUAL        │   TP    │   FN    │
              ├─────────┼─────────┤
            B │   217   │   59    │  276  (Actual Negatives)
              │   FP    │   TN    │
              └─────────┴─────────┘
                515       73       588
           (Pred Pos) (Pred Neg)
```

### Confusion Matrix Breakdown:
- **True Positives (TP)**: 298 - Correctly predicted A when answer is A
- **False Positives (FP)**: 217 - Incorrectly predicted A when answer is B ⚠️
- **True Negatives (TN)**: 59 - Correctly predicted B when answer is B
- **False Negatives (FN)**: 14 - Incorrectly predicted B when answer is A

---

## Metrics Calculation

### 1. **Accuracy = 60.71%**
```
Accuracy = (TP + TN) / Total
         = (298 + 59) / 588
         = 357 / 588
         = 0.6071 or 60.71%
```

**What it means**: The model is correct 60.7% of the time overall.

---

### 2. **Recall = 95.51%**
```
Recall = TP / (TP + FN)
       = 298 / (298 + 14)
       = 298 / 312
       = 0.9551 or 95.51%
```

**What it means**: Of all actual positive cases, the model correctly identifies 95.5% of them.

---

### 3. **Precision = 57.86%**
```
Precision = TP / (TP + FP)
          = 298 / (298 + 217)
          = 298 / 515
          = 0.5786 or 57.86%
```

**What it means**: When the model predicts A, it's only correct 58% of the time.

---

### 4. **F1-Score = 0.7207**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.5786 × 0.9551) / (0.5786 + 0.9551)
   = 0.7207
```

---

## Why the Discrepancy?

### The Root Cause: **Positive Class Bias**

The model predicts "A" (positive/yes) **87.6%** of the time (515 out of 588 predictions), even though the actual positive rate is only **53.1%** (312 out of 588).

### Visual Comparison:

```
Prediction Distribution:
█████████████████████████████████████████████████████ A (87.6%)
█████████ B (12.4%)

Ground Truth Distribution:
█████████████████████████████ A (53.1%)
████████████████████████ B (46.9%)
```

---

## The Impact

### ✅ **Why Recall is High (95.5%)**
- The model predicts A so frequently that it catches almost all actual A cases
- Only 14 out of 312 actual positives are missed (4.5% miss rate)
- **This is good for not missing positive cases**

### ❌ **Why Accuracy is Lower (60.7%)**
- The over-prediction of A causes 217 FALSE POSITIVES
- These false alarms bring down overall accuracy
- Out of 276 actual B cases, 217 are incorrectly classified as A (78.6% error rate on negatives!)

### ❌ **Why Precision is Low (57.9%)**
- When model says A, it's wrong 42% of the time
- **Low confidence in positive predictions**

---

## Mathematical Breakdown

| Metric | Formula | Calculation | Result |
|--------|---------|-------------|--------|
| **Accuracy** | (TP+TN)/Total | (298+59)/588 | **60.71%** |
| **Precision** | TP/(TP+FP) | 298/(298+217) | **57.86%** |
| **Recall** | TP/(TP+FN) | 298/(298+14) | **95.51%** |
| **F1-Score** | 2×P×R/(P+R) | Harmonic mean | **72.07%** |

---

## Real-World Analogy

Think of this like a **smoke detector**:

- **High Recall**: The detector goes off for almost every actual fire (good!)
- **Low Precision**: But it also goes off for burnt toast, steam, etc. (false alarms)
- **Low Accuracy**: Many alerts are false, so overall reliability is lower

In medical terms:
- **High Recall**: Good for screening tests (don't want to miss sick patients)
- **Low Precision**: But many healthy people get flagged (false positives)

---

## Why Does This Happen?

### Possible Causes:

1. **Imbalanced Training**: Model may have been trained on more positive examples
2. **Loss Function**: May prioritize not missing positives over avoiding false positives
3. **Threshold Issue**: Classification threshold may be set too low for predicting "A"
4. **Model Uncertainty**: When uncertain, model defaults to predicting "A"
5. **Prompt Design**: The prompt structure may encourage affirmative responses

---

## Solutions to Consider

### 1. **Adjust Classification Threshold**
- Currently using default threshold (likely 0.5)
- Could increase to 0.6 or 0.7 to reduce false positives
- Would trade off some recall for better precision

### 2. **Rebalance Training Data**
- Ensure equal representation of A and B in training

### 3. **Modify Loss Function**
- Use weighted cross-entropy to penalize false positives more

### 4. **Prompt Engineering**
- Revise prompts to be more neutral
- Avoid language that biases toward affirmative responses

### 5. **Calibration**
- Use probability calibration techniques (Platt scaling, isotonic regression)
- Better align predicted probabilities with actual frequencies

---

## Summary Table

| Aspect | Value | Interpretation |
|--------|-------|----------------|
| Total Samples | 588 | FES TEXT perturbations |
| Actual Positives | 312 (53.1%) | True distribution |
| Predicted Positives | 515 (87.6%) | ⚠️ Over-predicting |
| True Positives | 298 | Correct positive predictions |
| False Positives | 217 | ⚠️ Major issue |
| False Negatives | 14 | Very low |
| **Accuracy** | **60.71%** | Overall correctness |
| **Precision** | **57.86%** | Confidence in A predictions |
| **Recall** | **95.51%** | Catches most positives |
| **F1-Score** | **72.07%** | Balanced metric |

---

## Key Takeaway

> **The model is a "yes-sayer"** - it predicts positive/yes (option A) far too often, which means it catches most positive cases (high recall) but creates many false alarms (low precision and accuracy).

This is a classic example of the **precision-recall tradeoff**. The model has been tuned (intentionally or unintentionally) to favor recall at the expense of precision and overall accuracy.

---

## Recommendation

For the FESTA framework evaluating spatial reasoning:
- The current metrics suggest the model lacks confidence in its predictions
- Consider implementing probability-based prompting with calibration
- Monitor the precision-recall balance across different perturbation types
- Aim for more balanced performance (target: accuracy, precision, recall all above 70%)

---

*Analysis Date: November 7, 2025*  
*Report Based On: festa_report_20251031_075953.json*  
*Dataset: 143 samples with FES and FCS perturbations (Text & Image)*

