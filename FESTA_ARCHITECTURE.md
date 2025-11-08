# FESTA Evaluation System - Architecture Documentation

## Executive Summary

The FESTA (Factuality Evaluation via Semantic Transformation Analysis) system evaluates Vision-Language Models (VLMs) on spatial reasoning tasks by generating semantic transformations (paraphrases) and contradictions. The system uses multiple AI models and APIs to create robust test cases and calculate comprehensive evaluation metrics.

**Key Metrics Target**: AUROC ≥ 0.7 (70% discrimination capability)

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FESTA EVALUATION SYSTEM                      │
│                                                                       │
│  ┌────────────────┐      ┌──────────────────┐     ┌──────────────┐ │
│  │   Data Input   │─────▶│  Sample Generator │────▶│  Inference   │ │
│  │  BLINK Dataset │      │  (FES/FCS)        │     │   Engine     │ │
│  └────────────────┘      └──────────────────┘     └──────────────┘ │
│          │                        │                        │         │
│          │                        ▼                        ▼         │
│          │               ┌──────────────────┐    ┌──────────────┐  │
│          │               │  External APIs   │    │    LLaVA     │  │
│          │               │ (OpenAI/Gemini)  │    │    Model     │  │
│          │               └──────────────────┘    └──────────────┘  │
│          │                                                │         │
│          └────────────────────────┬───────────────────────┘         │
│                                   ▼                                 │
│                        ┌──────────────────┐                        │
│                        │ Metrics & Reports │                        │
│                        │   Visualization   │                        │
│                        └──────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. **Core Models & APIs**

```
┌───────────────────────────────────────────────────────────────────────┐
│                          MODEL ECOSYSTEM                              │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ PRIMARY MODEL: LLaVA (Vision-Language Model)                 │    │
│  │ ─────────────────────────────────────────────────────────    │    │
│  │ Model: llava-hf/llava-v1.6-mistral-7b-hf                    │    │
│  │ Purpose: Spatial reasoning evaluation                         │    │
│  │ Method: Probability-based top-k inference                     │    │
│  │                                                               │    │
│  │ Input:  Image + Question                                      │    │
│  │ Output: Prediction (A/B) + Confidence Scores                  │    │
│  │                                                               │    │
│  │ Inference Strategy:                                           │    │
│  │  • Generates k=4 predictions per sample                       │    │
│  │  • Samples n=5 times per prediction position                  │    │
│  │  • Uses temperature=0.7 for diversity                         │    │
│  │  • Combines via weighted voting                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ EXTERNAL API #1: OpenAI GPT-4o-mini                          │    │
│  │ ────────────────────────────────────────────────────────     │    │
│  │ Purpose: Text perturbation generation                         │    │
│  │                                                               │    │
│  │ FES Text (Functional Equivalent Samples):                     │    │
│  │  • Generates 5 semantic paraphrases                           │    │
│  │  • Maintains original meaning                                 │    │
│  │  • Example: "Is A left of B?" →                              │    │
│  │            "Is A positioned to the left of B?"               │    │
│  │                                                               │    │
│  │ FCS Text (Functional Complementary Samples):                  │    │
│  │  • Generates 5 contradictory questions                        │    │
│  │  • Flips spatial relations                                    │    │
│  │  • Example: "Is A left of B?" → "Is A right of B?"          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ EXTERNAL API #2: Google Gemini Pro (Configured, Not Used)    │    │
│  │ ────────────────────────────────────────────────────────     │    │
│  │ Purpose: Image analysis (reserved for future use)             │    │
│  │                                                               │    │
│  │ Status: API key configured but NOT actively called            │    │
│  │ Note: The GeminiImageTransformer class exists in code but     │    │
│  │       actual image transformations use PIL locally            │    │
│  │                                                               │    │
│  │ Actual Implementation: PIL handles all transformations        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ IMAGE PROCESSOR: PIL (Python Imaging Library)                │    │
│  │ ────────────────────────────────────────────────────────     │    │
│  │ Purpose: Execute image transformations locally                │    │
│  │                                                               │    │
│  │ FES Image Transformations (meaning-preserving):               │    │
│  │  • Rotation, flipping                                         │    │
│  │  • Color adjustments (brightness, contrast, saturation)       │    │
│  │  • Cropping, resizing                                         │    │
│  │  • Maintains spatial relationships                            │    │
│  │                                                               │    │
│  │ FCS Image Transformations (meaning-altering):                 │    │
│  │  • Object position swapping                                   │    │
│  │  • Spatial arrangement changes                                │    │
│  │  • Mirroring that alters relationships                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      FESTA PROCESSING PIPELINE                        │
└──────────────────────────────────────────────────────────────────────┘

PHASE 1: INITIALIZATION
════════════════════════
┌────────────┐
│ Load .env  │ → API Keys: OPENAI_API_KEY, GOOGLE_API_KEY, HF_TOKEN
└─────┬──────┘
      │
      ▼
┌────────────────────┐
│ Initialize APIs    │ → ComplementGenerator(OpenAI for text + PIL for images)
└─────┬──────────────┘
      │
      ▼
┌────────────────────┐
│ Load LLaVA Model   │ → llava-hf/llava-v1.6-mistral-7b-hf (GPU)
└─────┬──────────────┘
      │
      ▼
┌────────────────────┐
│ Load BLINK Dataset │ → Spatial_Relation (143 samples total)
└─────┬──────────────┘
      │
      ▼

PHASE 2: SAMPLE PROCESSING LOOP (for each of N samples)
════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────────┐
│ Sample N (e.g., N=31)                                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│ STEP 1: Original Inference                                         │
│ ─────────────────────────                                         │
│   Input: Original Image + Question                                │
│          ↓                                                         │
│   [LLaVA Model]                                                   │
│    • run_topk_inference(k=4, n_samples=5)                        │
│    • Generates 4 positions × 5 samples = 20 predictions          │
│    • Uses probability-based prompts                              │
│          ↓                                                         │
│   Output: Combined Prediction + Confidence Score                  │
│          ↓                                                         │
│   Save: original_prediction, original_confidence                  │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│ STEP 2: Generate FES Text (Paraphrases)                          │
│ ────────────────────────────────────────                          │
│   Input: Original Question                                        │
│          ↓                                                         │
│   [OpenAI GPT-4o-mini API]                                       │
│    • Prompt: Generate 5 semantic paraphrases                     │
│    • Maintain original meaning                                    │
│          ↓                                                         │
│   Output: 5 paraphrased questions                                │
│          ↓                                                         │
│   Save: sample_id_fes_text_1.json ... _5.json                   │
│          ↓                                                         │
│   [LLaVA Inference on each paraphrase]                           │
│    • Run inference on Original Image + Paraphrased Question      │
│    • Expected: Same prediction as original                       │
│          ↓                                                         │
│   Save: predictions + confidence scores                           │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│ STEP 3: Generate FCS Text (Contradictions)                       │
│ ─────────────────────────────────────────                         │
│   Input: Original Question                                        │
│          ↓                                                         │
│   [OpenAI GPT-4o-mini API]                                       │
│    • Prompt: Generate 5 contradictory questions                  │
│    • Flip spatial relations (left→right, above→below)           │
│          ↓                                                         │
│   Output: 5 contradictory questions                              │
│          ↓                                                         │
│   Save: sample_id_fcs_text_1.json ... _5.json                   │
│          ↓                                                         │
│   [LLaVA Inference on each contradiction]                        │
│    • Run inference on Original Image + Contradictory Question    │
│    • Expected: Opposite prediction (A→B or B→A)                 │
│          ↓                                                         │
│   Save: predictions + confidence scores                           │
│                                                                    │
│   [21-second delay to respect API rate limits]                   │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│ STEP 4: Generate FES Image (Meaning-Preserving)                  │
│ ────────────────────────────────────────────────                  │
│   Input: Original Image + Question                                │
│          ↓                                                         │
│   [Gemini Pro API]                                                │
│    • Analyze image content                                        │
│    • Suggest meaning-preserving transformations                  │
│          ↓                                                         │
│   [PIL Image Processing]                                          │
│    • Apply transformations:                                       │
│      - Rotation (maintain relationships)                          │
│      - Color adjustments                                          │
│      - Cropping/resizing                                          │
│          ↓                                                         │
│   Output: 5 transformed images                                    │
│          ↓                                                         │
│   Save: sample_id_fes_image_1.png ... _5.png                    │
│          ↓                                                         │
│   [LLaVA Inference on each image]                                │
│    • Run inference on Transformed Image + Original Question      │
│    • Expected: Same prediction as original                       │
│          ↓                                                         │
│   Save: predictions + confidence scores                           │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│ STEP 5: Generate FCS Image (Meaning-Altering)                    │
│ ──────────────────────────────────────────────                    │
│   Input: Original Image + Question                                │
│          ↓                                                         │
│   [PIL Image Processing - LOCAL]                                  │
│    • Apply relation-flipping transformations directly:            │
│      - Horizontal flip/mirror (left↔right)                       │
│      - Vertical flip (above↔below)                               │
│      - Rotation 180° (both axes)                                 │
│      - Plus subtle photometric adjustments for uniqueness        │
│          ↓                                                         │
│   Output: 3 transformed images                                    │
│          ↓                                                         │
│   Save: sample_id_fcs_image_1.png ... _3.png                    │
│          ↓                                                         │
│   [LLaVA Inference on each image]                                │
│    • Run inference on Transformed Image + Original Question      │
│    • Expected: Opposite prediction (A→B or B→A)                 │
│          ↓                                                         │
│   Save: predictions + confidence scores                           │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
      │
      ▼

PHASE 3: METRICS CALCULATION & REPORTING
═══════════════════════════════════════════

┌────────────────────────────────────────────────────────────────────┐
│ STEP 1: Aggregate All Predictions                                  │
│ ─────────────────────────────────                                 │
│   • Original predictions (N samples)                              │
│   • FES text predictions (N × 5 samples)                         │
│   • FCS text predictions (N × 5 samples)                         │
│   • FES image predictions (N × 5 samples)                        │
│   • FCS image predictions (N × 3 samples)                        │
│                                                                    │
│   Total predictions: N × (1 + 5 + 5 + 5 + 3) = N × 19           │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│ STEP 2: Calculate Core Metrics                                     │
│ ──────────────────────────────                                    │
│   [Collect arrays: y_true_all, y_pred_all, y_scores_all]        │
│                                                                    │
│   • AUROC (Area Under ROC Curve)                                 │
│     - Requires: Multiple classes in ground truth                  │
│     - Method: sklearn.metrics.roc_auc_score()                    │
│     - Input: y_true_all, y_scores_all                            │
│     - Target: ≥ 0.7 (70% discrimination)                         │
│                                                                    │
│   • Accuracy                                                      │
│     - Original samples only                                       │
│     - All samples (original + generated)                         │
│                                                                    │
│   • Precision, Recall, F1-Score                                  │
│     - sklearn.metrics.precision_recall_fscore_support()          │
│     - Binary classification metrics                              │
│                                                                    │
│   • AUPRC (Area Under Precision-Recall Curve)                   │
│     - sklearn.metrics.average_precision_score()                  │
│                                                                    │
│   • Brier Score                                                   │
│     - Calibration metric                                          │
│     - sklearn.metrics.brier_score_loss()                         │
│                                                                    │
│   • ECE (Expected Calibration Error)                             │
│     - Custom calculation with n_bins=10                          │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│ STEP 3: FESTA-Specific Metrics                                    │
│ ──────────────────────────────                                    │
│   • FES Consistency                                               │
│     - Measures: Agreement between original and paraphrases       │
│     - Formula: # matching predictions / # total FES samples      │
│                                                                    │
│   • FCS Discrimination                                            │
│     - Measures: Ability to detect contradictions                 │
│     - Formula: # opposite predictions / # total FCS samples      │
│                                                                    │
│   • Sample Type Breakdown                                         │
│     - Separate metrics for:                                       │
│       * Original samples                                          │
│       * FES text samples                                          │
│       * FES image samples                                         │
│       * FCS text samples                                          │
│       * FCS image samples                                         │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│ STEP 4: Generate Visualizations                                   │
│ ───────────────────────────────                                   │
│   [FESTAVisualizer class]                                         │
│                                                                    │
│   • ROC Curve                                                     │
│     - X-axis: False Positive Rate                                │
│     - Y-axis: True Positive Rate                                 │
│     - Shows discrimination capability                            │
│                                                                    │
│   • Precision-Recall Curve                                       │
│     - X-axis: Recall                                              │
│     - Y-axis: Precision                                           │
│     - Shows trade-off between precision/recall                   │
│                                                                    │
│   • Risk-Coverage Curve                                          │
│     - X-axis: Coverage (fraction of samples retained)            │
│     - Y-axis: Risk (error rate)                                  │
│     - Shows selective prediction performance                     │
│                                                                    │
│   • Accuracy-Coverage Curve                                      │
│     - X-axis: Coverage                                            │
│     - Y-axis: Accuracy                                            │
│     - Shows accuracy when abstaining from uncertain predictions  │
│                                                                    │
│   • Confidence Distribution                                       │
│     - Histogram of confidence scores                             │
│     - Separate by correct/incorrect predictions                  │
│                                                                    │
│   • Calibration Plot                                              │
│     - Expected vs Observed Accuracy                              │
│     - Shows model calibration quality                            │
│                                                                    │
│   • Bar Charts                                                    │
│     - Accuracy by sample type (FES/FCS, text/image)             │
│     - Metrics comparison (Precision, Recall, F1)                │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│ STEP 5: Generate Reports                                          │
│ ────────────────────────                                          │
│   • JSON Results                                                  │
│     - output/api_run/api_evaluation_results.json                │
│     - Complete detailed results                                  │
│                                                                    │
│   • Timestamped Report                                            │
│     - reports/festa_report_YYYYMMDD_HHMMSS.json                 │
│     - Comprehensive metrics + metadata                           │
│                                                                    │
│   • Markdown Summary                                              │
│     - output/api_run/api_evaluation_report.md                   │
│     - Human-readable summary                                     │
│                                                                    │
│   • CSV Export (if module available)                             │
│     - Tabular format for external analysis                       │
│     - Per-sample and aggregate metrics                           │
│                                                                    │
│   • Visualization PNGs                                            │
│     - All charts saved as PNG files                              │
│     - Embedded in HTML reports                                   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Metrics Explained

### 1. **AUROC (Area Under ROC Curve)** [Target: ≥ 0.7]

**What it measures**: Model's ability to discriminate between positive and negative cases across all confidence thresholds.

**How it's calculated**:
```python
from sklearn.metrics import roc_auc_score
auroc = roc_auc_score(y_true, y_confidence_scores)
```

**Interpretation**:
- **1.0**: Perfect discrimination
- **≥ 0.7**: Good discrimination (FESTA target)
- **0.5**: Random guessing
- **< 0.5**: Worse than random

**Why it matters**: AUROC evaluates the model across ALL possible thresholds, not just at a single decision boundary. It's the gold standard for binary classification evaluation.

**FESTA enhancement**: By including FES and FCS samples, we create prediction diversity that enables meaningful AUROC calculation even from small sample sets.

---

### 2. **Accuracy**

**What it measures**: Percentage of correct predictions.

**Formula**:
```
Accuracy = (True Positives + True Negatives) / Total Samples
```

**Two versions**:
- **Original Accuracy**: Only from original samples
- **All Accuracy**: Including generated FES/FCS samples

**Interpretation**:
- **100%**: All predictions correct
- **50%**: Random guessing (for binary)
- **0%**: All predictions wrong

---

### 3. **Precision**

**What it measures**: Of all positive predictions, how many were actually correct?

**Formula**:
```
Precision = True Positives / (True Positives + False Positives)
```

**When it matters**: When false positives are costly.

---

### 4. **Recall (Sensitivity)**

**What it measures**: Of all actual positives, how many did we correctly identify?

**Formula**:
```
Recall = True Positives / (True Positives + False Negatives)
```

**When it matters**: When false negatives are costly.

---

### 5. **F1-Score**

**What it measures**: Harmonic mean of Precision and Recall.

**Formula**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Interpretation**: Balanced metric when you care equally about precision and recall.

---

### 6. **AUPRC (Area Under Precision-Recall Curve)**

**What it measures**: Model performance across different precision-recall trade-offs.

**How it's calculated**:
```python
from sklearn.metrics import average_precision_score
auprc = average_precision_score(y_true, y_confidence_scores)
```

**When it's better than AUROC**: For imbalanced datasets.

---

### 7. **Brier Score**

**What it measures**: Mean squared difference between predicted probabilities and actual outcomes.

**Formula**:
```
Brier Score = (1/N) × Σ(predicted_prob - actual_outcome)²
```

**Interpretation**:
- **0.0**: Perfect calibration
- **Lower is better**

**Why it matters**: Evaluates both discrimination AND calibration.

---

### 8. **ECE (Expected Calibration Error)**

**What it measures**: Average difference between confidence and accuracy across probability bins.

**Method**:
1. Divide predictions into 10 bins by confidence (0-0.1, 0.1-0.2, ..., 0.9-1.0)
2. For each bin, calculate: |average_confidence - accuracy|
3. Weight by bin size and sum

**Interpretation**:
- **0.0**: Perfectly calibrated
- **Lower is better**

**Why it matters**: A well-calibrated model's confidence scores match actual accuracy (e.g., when it says 80% confident, it's correct 80% of the time).

---

### 9. **FES Consistency** (FESTA-specific)

**What it measures**: Agreement between original and semantically equivalent samples.

**Formula**:
```
FES Consistency = # matching predictions / # total FES samples
```

**Interpretation**:
- **100%**: Perfect robustness to paraphrasing
- **High values expected**: FES samples should yield same predictions

---

### 10. **FCS Discrimination** (FESTA-specific)

**What it measures**: Ability to detect contradictions.

**Formula**:
```
FCS Discrimination = # opposite predictions / # total FCS samples
```

**Interpretation**:
- **100%**: Perfect contradiction detection
- **50%**: Random/confused
- **High values desired**: Model should detect contradictory spatial relations

---

## Key Design Decisions

### 1. **Why Probability-Based Prompts?**

```python
def run_topk_inference(model, processor, image, question, options, k=4, n_samples=5):
    prompt = f"""
    Provide your {k} best guesses and the probability that each is correct (0.0 to 1.0).
    G1: <first most likely guess>
    P1: <probability for G1>
    ...
    G{k}: <{k}-th most likely guess>
    P{k}: <probability for G{k}>
    """
```

**Rationale**:
- **Uncertainty quantification**: Captures model's confidence
- **Diversity**: Multiple samples reveal prediction stability
- **AUROC calculation**: Requires probability scores, not just binary predictions

### 2. **Why External APIs (OpenAI + Gemini)?**

**Problem**: Manual generation of high-quality paraphrases and contradictions is:
- Time-consuming
- Prone to bias
- Not scalable

**Solution**: Leverage state-of-the-art LLMs:
- **OpenAI GPT-4o-mini**: Expert at text transformations
- **Gemini Pro**: Multimodal understanding for image analysis

### 3. **Why 21-Second Delays?**

**API Rate Limits**:
- OpenAI: 3 requests per minute (free tier)
- Gemini: Similar restrictions

**Calculation**: 60 seconds / 3 requests = 20 seconds minimum
**Safety margin**: Add 1 second → 21 seconds

### 4. **Why FES + FCS?**

**FESTA Philosophy**:
- **FES**: Tests semantic robustness (should maintain predictions)
- **FCS**: Tests logical consistency (should flip predictions)
- **Together**: Create prediction diversity needed for comprehensive evaluation

---

## Performance Optimization

### GPU Utilization
```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_4BIT_QUANTIZATION = True  # Fits in 25GB GPU
```

### Batch Processing
```python
NUM_SAMPLES = 143  # Full dataset
SKIP_SAMPLES = 0   # Offset for parallel runs
```

### Memory Management
- 4-bit quantization reduces model size
- Automatic garbage collection after each sample
- Image transformations done locally (PIL) to reduce API calls

---

## Output Structure

```
output/
├── api_run/
│   ├── api_evaluation_results.json     # Complete results
│   ├── api_evaluation_report.md        # Human-readable summary
│   └── generated_samples/
│       ├── sample_31_original.png
│       ├── sample_31_fes_text_1.json
│       ├── sample_31_fes_text_2.json
│       ├── ...
│       ├── sample_31_fes_image_1.png
│       ├── ...
│       ├── sample_31_fcs_text_1.json
│       └── sample_31_fcs_image_1.png
│
└── metrics/
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── risk_coverage_curve.png
    ├── accuracy_coverage_curve.png
    ├── confidence_distribution.png
    └── calibration_plot.png

reports/
└── festa_report_20251031_HHMMSS.json  # Timestamped comprehensive report
```

---

## Summary

### **System Purpose**
Evaluate LLaVA's spatial reasoning and robustness through systematic semantic transformations.

### **Key Innovation**
Uses OpenAI GPT-4o-mini API for text generation (paraphrases and contradictions) and PIL (Python Imaging Library) for local image transformations, enabling scalable evaluation without manual sample creation. Note: Google Gemini Pro API is configured but not actively used in the current implementation.

### **Core Workflow**
1. Load sample from BLINK dataset
2. Generate FES/FCS variations (5 text + 5 image + 5 text + 3 image = 18 variants)
3. Run LLaVA inference on original + all variants
4. Calculate comprehensive metrics (10+ metrics)
5. Generate visualizations (6+ charts)
6. Export to multiple formats (JSON, Markdown, CSV)

### **Primary Metric**
**AUROC ≥ 0.7** (measures discrimination capability across all thresholds)

### **Scalability**
- Configurable: 2 samples for testing → 143 samples for full evaluation
- Parallel execution support via `SKIP_SAMPLES` offset
- GPU-optimized with 4-bit quantization

### **Output**
- Comprehensive JSON reports with all predictions
- Timestamped reports for tracking runs
- Visual charts for analysis
- CSV export for external tools

---

## Configuration

### Environment Variables
```bash
# API Keys
OPENAI_API_KEY=your-key-here
GOOGLE_API_KEY=your-key-here
HF_TOKEN=your-huggingface-token

# Sample Configuration
NUM_SAMPLES=2          # Number of samples to process
SKIP_SAMPLES=0         # Offset for parallel runs

# Model Configuration
MODEL_ID_MLLM=llava-hf/llava-v1.6-mistral-7b-hf
OPENAI_TEXT_MODEL=gpt-4o-mini
GOOGLE_GEMINI_MODEL=gemini-pro-latest

# FESTA Parameters
FES_TEXT_SAMPLES=5
FES_IMAGE_SAMPLES=5
FCS_TEXT_SAMPLES=5
FCS_IMAGE_SAMPLES=3
```

---

## Conclusion

The FESTA system provides a comprehensive, automated framework for evaluating VLMs on spatial reasoning tasks. By leveraging multiple AI models and APIs, it generates diverse test cases that enable robust evaluation with metrics that meet research standards (AUROC ≥ 0.7).

The architecture is designed for scalability, reproducibility, and comprehensive analysis, making it suitable for both research and production evaluation pipelines.

