# FESTA Evaluation System - Quick Reference Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Component Summary](#component-summary)
3. [Metrics Quick Reference](#metrics-quick-reference)
4. [API Usage](#api-usage)
5. [Running the System](#running-the-system)

---

## System Overview

**FESTA** = **F**actuality **E**valuation via **S**emantic **T**ransformation **A**nalysis

### Purpose
Evaluate Vision-Language Models (VLMs) on spatial reasoning tasks through systematic semantic transformations.

### Core Concept
```
Original Sample
    ↓
Generate Variants (FES + FCS)
    ↓
Run Inference on All Variants
    ↓
Calculate Comprehensive Metrics
    ↓
Visualize Results
```

### Key Innovation
Uses external AI APIs (OpenAI + Gemini) to automatically generate high-quality test variations, eliminating manual sample creation.

---

## Component Summary

### 1. Models

| Model | Purpose | Location | Usage |
|-------|---------|----------|-------|
| **LLaVA v1.6** | Primary evaluation target | GPU (local) | Spatial reasoning inference |
| **OpenAI GPT-4o-mini** | Text generation | Cloud API (Active) | FES/FCS text samples |
| **Google Gemini Pro** | Reserved (not used) | Cloud API (Configured only) | Not actively called |
| **PIL (Pillow)** | Image transformations | Local (CPU) | ALL FES/FCS image generation |

### 2. Sample Types

| Type | Count | Purpose | Expected Result |
|------|-------|---------|-----------------|
| **Original** | 1 | Baseline | N/A |
| **FES Text** | 5 | Semantic paraphrases | Same prediction |
| **FCS Text** | 5 | Contradictions | Opposite prediction |
| **FES Image** | 5 | Meaning-preserving transforms | Same prediction |
| **FCS Image** | 3 | Meaning-altering transforms | Opposite prediction |
| **TOTAL** | **19** per sample | Comprehensive evaluation | Diverse predictions |

### 3. Data Flow

```
BLINK Dataset (143 samples)
    ↓
Select N samples (configurable)
    ↓
For each sample:
    1. Run original inference
    2. Generate FES text (OpenAI)
    3. Generate FCS text (OpenAI)
    4. Generate FES image (Gemini + PIL)
    5. Generate FCS image (Gemini + PIL)
    6. Run inference on all variants
    ↓
Aggregate predictions
    ↓
Calculate metrics
    ↓
Generate visualizations
    ↓
Export reports (JSON, MD, CSV)
```

---

## Metrics Quick Reference

### Core Metrics

#### 1. AUROC (Area Under ROC Curve) [PRIMARY METRIC]
- **Target**: ≥ 0.7 (70% discrimination)
- **Range**: 0.0 - 1.0
- **Interpretation**:
  - 1.0 = Perfect discrimination
  - 0.7-0.9 = Good discrimination ✓
  - 0.5 = Random guessing
  - < 0.5 = Worse than random
- **Requires**: At least 2 classes in ground truth
- **Why it matters**: Evaluates model across all thresholds, not just one decision point

#### 2. Accuracy
- **Formula**: Correct predictions / Total predictions
- **Two versions**:
  - Original: Only baseline samples
  - All: Including generated samples
- **Range**: 0% - 100%
- **Interpretation**: Higher is better

#### 3. Precision
- **Formula**: True Positives / (True Positives + False Positives)
- **Range**: 0.0 - 1.0
- **When it matters**: When false positives are costly
- **Example**: 0.95 = Of all "yes" predictions, 95% were correct

#### 4. Recall (Sensitivity)
- **Formula**: True Positives / (True Positives + False Negatives)
- **Range**: 0.0 - 1.0
- **When it matters**: When false negatives are costly
- **Example**: 0.90 = Found 90% of all actual "yes" cases

#### 5. F1-Score
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Range**: 0.0 - 1.0
- **Purpose**: Balanced metric combining precision and recall
- **Interpretation**: Higher is better

#### 6. AUPRC (Area Under Precision-Recall Curve)
- **Similar to**: AUROC but for precision-recall trade-off
- **Range**: 0.0 - 1.0
- **Better than AUROC when**: Dataset is imbalanced
- **Interpretation**: Higher is better

#### 7. Brier Score
- **Formula**: Mean squared error of probability predictions
- **Range**: 0.0 - 1.0
- **Interpretation**: Lower is better (0.0 = perfect calibration)
- **Measures**: Both discrimination AND calibration

#### 8. ECE (Expected Calibration Error)
- **Method**: Binned average of |confidence - accuracy|
- **Range**: 0.0 - 1.0
- **Interpretation**: Lower is better (0.0 = perfectly calibrated)
- **Purpose**: Check if confidence scores match actual accuracy
- **Example**: ECE=0.05 means model is well-calibrated

### FESTA-Specific Metrics

#### 9. FES Consistency
- **Formula**: Matching predictions / Total FES samples
- **Range**: 0% - 100%
- **Expected**: High (ideally 100%)
- **Measures**: Robustness to semantic paraphrasing
- **Example**: If original pred=A, all FES should also pred=A

#### 10. FCS Discrimination
- **Formula**: Opposite predictions / Total FCS samples
- **Range**: 0% - 100%
- **Expected**: High (ideally 100%)
- **Measures**: Ability to detect contradictions
- **Example**: If original pred=A, all FCS should pred=B

---

## API Usage

### OpenAI GPT-4o-mini

**Purpose**: Text generation (paraphrases and contradictions)

**Rate Limit**: 3 requests per minute (free tier)

**Usage Pattern**:
```python
# FES Text (Paraphrases)
prompt = """
Generate 5 semantic paraphrases of this question that maintain the same meaning:
Question: {question}
"""

# FCS Text (Contradictions)
prompt = """
Generate 5 contradictory questions by flipping spatial relations:
Question: {question}
Example: "Is A left of B?" → "Is A right of B?"
"""
```

**Cost Estimate** (2 samples):
- FES text requests: 2
- FCS text requests: 2
- Total: 4 requests (~$0.01)

**Cost Estimate** (143 samples):
- Total: 286 requests (~$1.50)

**Note**: No image generation API costs since PIL is used locally

**Purpose**: Image analysis (reserved for future use)

**Status**: **CONFIGURED BUT NOT ACTIVELY USED**

**Current Implementation**: 
The code contains a `GeminiImageTransformer` class that initializes the Gemini Pro API, but in practice, **ALL image transformations are performed locally using PIL (Python Imaging Library)**. No actual API calls are made to Gemini for image generation.

**Actual Image Generation**:
The code contains a `GeminiImageTransformer` class that initializes the Gemini Pro API, but in practice, **ALL image transformations are performed locally using PIL (Python Imaging Library)**. No actual API calls are made to Gemini for image generation.
# PIL performs all transformations locally (CPU-based)
# FES: Gaussian noise, blur, contrast, brightness, grayscale, dotted masking
# FCS: Horizontal flip, vertical flip, rotation 180°
# No external API calls involved
- Full control over transformation parameters

**Why PIL Instead of API**:
- Faster execution (no network latency)
- No API costs
- Deterministic results with seed control
- Full control over transformation parameters
- Deterministic results with seed control
**Clarification**: While the codebase contains an `OpenAIImageTransformer` class that references DALL-E 2, **this class is NOT instantiated or used in the actual workflow**. All image generation is handled by PIL locally.

### Rate Limiting Strategy

**Problem**: OpenAI API limits = 3 requests/minute (text generation only)

**Solution**: 21-second delays between API calls
- 60 seconds / 3 requests = 20 seconds
- +1 second safety margin = 21 seconds

**Impact on Runtime**:
- 2 samples: ~5-10 minutes (mainly for text generation)
- 143 samples: ~8-10 hours (text generation + local PIL processing)

**Note**: No rate limiting needed for image transformations since PIL runs locally

**Optimization**: Use parallel execution with multiple API keys for text generation

---

## Running the System

### Prerequisites

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
#   OPENAI_API_KEY=sk-...
#   GOOGLE_API_KEY=AIza...
#   HF_TOKEN=hf_...

# 3. Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Test Run (2 samples)

```bash
# Set configuration
export NUM_SAMPLES=2
export SKIP_SAMPLES=0

# Run evaluation
python src/festa_with_apis.py

# Expected output:
# - 38 predictions (2 samples × 19 variants)
# - AUROC ≥ 0.7
# - Runtime: ~5-10 minutes
```

### Full Run (143 samples)

```bash
# Set configuration
export NUM_SAMPLES=143
export SKIP_SAMPLES=0

# Run evaluation (takes 8-12 hours)
python src/festa_with_apis.py

# Expected output:
# - 2,717 predictions (143 × 19)
# - Comprehensive metrics
# - Runtime: ~8-12 hours
```

### Parallel Execution (Recommended for full dataset)

**Terminal 1** (Samples 1-50):
```bash
export NUM_SAMPLES=50
export SKIP_SAMPLES=0
python src/festa_with_apis.py > logs/run1.log 2>&1 &
```

**Terminal 2** (Samples 51-100):
```bash
export NUM_SAMPLES=50
export SKIP_SAMPLES=50
python src/festa_with_apis.py > logs/run2.log 2>&1 &
```

**Terminal 3** (Samples 101-143):
```bash
export NUM_SAMPLES=43
export SKIP_SAMPLES=100
python src/festa_with_apis.py > logs/run3.log 2>&1 &
```

**Monitor Progress**:
```bash
tail -f logs/run1.log logs/run2.log logs/run3.log
```

**Benefits**:
- Reduced runtime: ~3-4 hours
- Parallel API usage
- Independent failure domains

### Monitoring Progress

```bash
# Check live progress
tail -f output/logs/festa_evaluation.log

# Count completed samples
ls -1 output/api_run/generated_samples/*_original.png | wc -l

# Check GPU usage
nvidia-smi -l 1

# Monitor API calls
grep "→ Generating" output/logs/festa_evaluation.log | tail -20
```

---

## Output Structure

```
output/
├── api_run/
│   ├── api_evaluation_results.json       # Complete results
│   ├── api_evaluation_report.md          # Human-readable summary
│   └── generated_samples/
│       ├── sample_31_original.png
│       ├── sample_31_fes_text_1.json
│       ├── sample_31_fes_image_1.png
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
└── festa_report_20251031_HHMMSS.json     # Timestamped report
```

---

## Troubleshooting

### Issue: "AUROC calculation skipped: only 1 class present"

**Cause**: All samples have the same ground truth label

**Solution**: 
- Need samples with different labels (both A and B)
- With FES/FCS samples, this is typically resolved
- Check if predictions are too consistent

### Issue: "Rate limit exceeded"

**Cause**: Too many API calls too quickly

**Solution**:
- Ensure 21-second delays are enabled
- Check if parallel runs are using same API key
- Consider using different API keys for parallel runs

### Issue: "CUDA out of memory"

**Cause**: GPU memory insufficient

**Solution**:
```bash
# Enable 4-bit quantization
export USE_4BIT_QUANTIZATION=true

# Reduce batch size
export BATCH_SIZE=1

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

### Issue: "Generated images don't look correct"

**Cause**: PIL transformations may be too aggressive

**Solution**:
- Review transformation parameters in `complement_generator.py`
- Check Gemini's analysis output
- Verify image paths are correct

---

## Quick Command Reference

```bash
# Test run (2 samples)
NUM_SAMPLES=2 SKIP_SAMPLES=0 python src/festa_with_apis.py

# Full run (143 samples)
NUM_SAMPLES=143 SKIP_SAMPLES=0 python src/festa_with_apis.py

# Monitor GPU
watch -n 1 nvidia-smi

# Count generated samples
find output/api_run/generated_samples -name "*.png" | wc -l

# View latest report
cat output/api_run/api_evaluation_report.md

# Check AUROC
grep "AUROC" output/api_run/api_evaluation_report.md

# Export to CSV
python src/metrics_csv_export.py

# Generate visualizations only
python test_visualizations.py
```

---

## Performance Optimization Tips

1. **Use GPU**: Ensure CUDA is available
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```

2. **Enable 4-bit Quantization**: Reduces memory by 75%
   ```bash
   export USE_4BIT_QUANTIZATION=true
   ```

3. **Parallel Execution**: Use multiple instances with SKIP_SAMPLES
   ```bash
   # Instance 1: Samples 0-49
   # Instance 2: Samples 50-99
   # Instance 3: Samples 100-142
   ```

4. **Pre-cache Models**: Download before running
   ```bash
   python -c "from transformers import LlavaNextProcessor; LlavaNextProcessor.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf')"
   ```

5. **Monitor Resource Usage**:
   ```bash
   # GPU
   nvidia-smi -l 1
   
   # CPU/RAM
   htop
   
   # Disk I/O
   iotop
   ```

---

## Expected Results (2 Samples)

| Metric | Expected Value | Status |
|--------|---------------|--------|
| AUROC | ≥ 0.70 | ✓ Target |
| Accuracy | ≥ 90% | ✓ Good |
| Precision | ≥ 0.85 | ✓ Good |
| Recall | ≥ 0.85 | ✓ Good |
| F1-Score | ≥ 0.85 | ✓ Good |
| FES Consistency | ≥ 80% | ✓ Robust |
| FCS Discrimination | ≥ 80% | ✓ Discriminative |
| Runtime | 5-10 min | ✓ Fast |
| Total Predictions | 38 | ✓ Comprehensive |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/festa_with_apis.py` | Main evaluation script |
| `src/complement_generator.py` | FES/FCS sample generation |
| `src/festa_metrics.py` | Metrics calculation & visualization |
| `src/festa_evaluation.py` | LLaVA model wrapper |
| `src/prompts_text.py` | Text generation prompts |
| `src/prompts_image.py` | Image analysis prompts |
| `requirements.txt` | Python dependencies |
| `.env` | API keys and configuration |

---

## Support

For issues, check:
1. Log files in `output/logs/`
2. Error messages in terminal output
3. Generated samples in `output/api_run/generated_samples/`
4. GPU memory usage with `nvidia-smi`

---

- ✓ Uses OpenAI GPT-4o-mini API for text generation
- ✓ Uses PIL locally for all image transformations (no image API costs)

FESTA is a comprehensive VLM evaluation framework that:
- ✓ Generates diverse test cases automatically
- ✓ Uses state-of-the-art APIs (OpenAI + Gemini)
- ✓ Calculates 10+ metrics including AUROC
**API Usage Summary**:
- OpenAI GPT-4o-mini: Text paraphrases and contradictions (ACTIVE)
- Google Gemini Pro: Configured but NOT used (reserved)
- DALL-E: NOT used (class exists but not instantiated)
- PIL: ALL image transformations (LOCAL, CPU-based)

- ✓ Produces rich visualizations
- ✓ Scales from 2 to 143 samples
- ✓ Targets ≥0.7 AUROC for research quality

**Primary Command**:
```bash
NUM_SAMPLES=2 python src/festa_with_apis.py
```

**Primary Output**:
```
reports/festa_report_<timestamp>.json
```

**Primary Metric**:
```
AUROC ≥ 0.7
```

