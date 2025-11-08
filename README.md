# FESTA: Reality Estimation with Factual and Semantic Uncertainty

This project implements the FESTA (Factual and Semantic Uncertainty) framework for evaluating Vision-Language Models (VLMs) on their ability to distinguish between real and AI-generated images.

## Overview

FESTA is a comprehensive evaluation framework that tests VLMs' robustness through:
- **Factual Uncertainty**: Testing with semantically equivalent samples (FES)
- **Semantic Uncertainty**: Evaluating with semantically different samples (SES)
- **Combined Analysis**: Comprehensive metrics across text and image modalities

## Features

- ✨ **Multi-Modal Evaluation**: Support for both text-only and vision-language models
- 🔄 **Functional Equivalent Samples (FES)**: Image perturbations including grayscale, blur, noise, brightness, contrast, rotation, crop, and occlusion
- 📊 **Comprehensive Metrics**: AUROC, AUPRC, Brier Score, ECE, Accuracy, Precision, Recall, F1-Score
- 🎯 **Calibration Analysis**: Temperature scaling and Expected Calibration Error
- 📈 **Visualization**: Risk-Coverage curves, Accuracy-Coverage curves, ROC/PR curves
- 🚀 **GPU Acceleration**: Optimized for CUDA with efficient batch processing
- 🔌 **API Integration**: Support for OpenAI and Anthropic APIs

## Project Structure

```
REALITY_ESTIMATION_WITH_FESTA/
├── config/
│   └── config.yaml              # Configuration for generation & evaluation
├── src/
│   ├── festa_evaluation.py      # Main evaluation script
│   ├── festa_metrics.py         # Metrics calculation
│   ├── festa_calibration.py     # Calibration methods
│   ├── festa_4x14_gpu.py        # GPU-optimized 4x14 evaluation
│   ├── festa_with_apis.py       # API-based evaluation
│   ├── complement_generator.py  # FES/SES generation
│   ├── prompts_text.py          # Text prompts
│   ├── prompts_image.py         # Image prompts
│   ├── analyze_results.py       # Results analysis
│   └── metrics_csv_export.py    # CSV export utilities
├── generate_csv_only.py         # CSV report generation
├── paper/
│   └── 2509.16648v1.pdf        # FESTA paper reference
└── sample_report/               # Sample results and visualizations

```

## Installation

### Requirements

```bash
# Python 3.8+
pip install torch transformers pillow numpy pandas scikit-learn
pip install matplotlib seaborn tqdm datasets pyyaml
pip install accelerate bitsandbytes  # For GPU optimization
```

### Optional API Support

```bash
pip install openai anthropic  # For API-based evaluation
```

## Usage

### 1. Configure Evaluation

Edit `config/config.yaml` to customize:
- Number of samples
- FES perturbation parameters
- Model selection
- Output directories

### 2. Run Evaluation

#### GPU-Optimized Evaluation (4x14 Grid)
```bash
python src/festa_4x14_gpu.py
```

#### API-Based Evaluation
```bash
python src/festa_with_apis.py
```

#### Standard Evaluation
```bash
python src/festa_evaluation.py
```

### 3. Generate Reports

```bash
python generate_csv_only.py
```

This generates comprehensive CSV reports including:
- Metrics summary (AUROC, AUPRC, Accuracy, etc.)
- Per-sample predictions
- Calibration statistics
- Category-wise breakdowns

### 4. Analyze Results

```bash
python src/analyze_results.py
```

## Configuration

The `config/config.yaml` file controls all aspects of evaluation:

```yaml
generation:
  num_samples: 143
  fes:
    text_samples: 5
    image_samples: 5
    perturbations:
      - type: "grayscale"
      - type: "blur"
      - type: "noise"
      - type: "brightness"
      - type: "contrast"
      - type: "rotation"
      - type: "crop"
      - type: "occlusion"
```

## Metrics

The framework computes comprehensive metrics:

- **AUROC**: Area Under ROC Curve
- **AUPRC**: Area Under Precision-Recall Curve
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Classification quality metrics
- **Brier Score**: Probabilistic prediction quality
- **ECE**: Expected Calibration Error
- **Risk-Coverage Curves**: Selective prediction analysis
- **Accuracy-Coverage Curves**: Performance vs. coverage trade-offs

## Output

Results are organized in the `output/` directory:
- `comprehensive_metrics.json`: All computed metrics
- `festa_results.json`: Detailed predictions
- `csv_final/`: CSV reports with timestamps
- Visualization plots (ROC, PR, calibration curves)

## Models Supported

- **Vision-Language Models**: LLaVA-NeXT, CLIP-based models
- **Text-Only Models**: GPT-family, LLaMA, Mistral
- **API Models**: OpenAI GPT-4V, Claude with vision

## Paper Reference

This implementation is based on the FESTA framework described in:
- Paper: `paper/2509.16648v1.pdf`

## Contributors

1. **Samuel Joseph Polavarapu**
2. **Saravana Kumar**

## License

This project is provided for research and educational purposes.

## Citation

If you use this code in your research, please cite the FESTA paper and acknowledge this implementation.

---

**Last Updated**: November 2025

