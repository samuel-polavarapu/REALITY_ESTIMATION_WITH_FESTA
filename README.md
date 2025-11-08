# FESTA: Functionally Equivalent and Semantically Targeted Augmentation

**FESTA 4×14 Nested Loop Strategy** - Advanced multimodal evaluation framework for spatial reasoning in Vision-Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU: RTX 5090](https://img.shields.io/badge/GPU-RTX%205090-green.svg)](https://www.nvidia.com/)

## 🎯 Overview

FESTA is a comprehensive evaluation framework that tests Vision-Language Models (VLMs) on spatial reasoning tasks using:
- **FES (Functionally Equivalent Samples)**: 4 text paraphrases × 14 image variants = 56 combinations
- **FCS (Functionally Complementary Samples)**: 4 text contradictions × 14 image variants = 56 combinations
- **Total**: 112 inference runs per sample for robust uncertainty quantification

## 🚀 Key Features

- **Nested Loop Inference**: 4×14 strategy for comprehensive testing
- **GPU Optimized**: Full support for RTX 5090 with 4-bit quantization
- **Probability-based Predictions**: Top-k sampling with confidence calibration
- **Comprehensive Metrics**: AUROC, AUPRC, Accuracy, Precision, Recall, F1, Brier Score, ECE
- **API Integration**: OpenAI GPT-4o-mini for text, Gemini Pro for image analysis
- **Production Ready**: Modular design with full logging and error handling

## 📋 Requirements

```
python >= 3.8
torch >= 2.0
transformers >= 4.35
datasets
pillow
numpy
scikit-learn
openai
google-generativeai
python-dotenv
```

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/FESTA-4x14-Strategy.git
cd FESTA-4x14-Strategy

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=your_openai_key
# GEMINI_API_KEY=your_gemini_key
```

## 💡 Usage

### Quick Start - Run 3 Samples

```bash
# Run with GPU optimization
python run_nested_loop.py
```

### Run for All 143 Samples

```bash
# Set environment variables
export NUM_SAMPLES=143
export SKIP_SAMPLES=0

# Run the full evaluation
python run_nested_loop.py
```

### Custom Configuration

```python
import os

# Configure GPU settings
os.environ['BATCH_SIZE'] = '2'
os.environ['NUM_INFERENCE_SAMPLES'] = '20'
os.environ['USE_4BIT_QUANTIZATION'] = 'true'

# Configure samples
os.environ['NUM_SAMPLES'] = '10'
os.environ['SKIP_SAMPLES'] = '0'

# Run
from src import festa_nested_loop
festa_nested_loop.main()
```

## 📊 Strategy

### FES (Functionally Equivalent Samples)
Generates semantically equivalent variations to test model consistency:
- **Text**: 4 paraphrases (e.g., "Is the car beneath the cat?" → "Is the car located beneath the cat?")
- **Images**: 14 transformations (grayscale, blur, noise, contrast, brightness, etc.)
- **Expected**: Same answer as original

### FCS (Functionally Complementary Samples)  
Generates contradictory variations to test model discrimination:
- **Text**: 4 contradictions (e.g., "Is the car beneath the cat?" → "Is the cat beneath the car?")
- **Images**: Same 14 transformations as FES (reused for efficiency)
- **Expected**: Opposite answer from original

## 📈 Metrics

The framework calculates comprehensive metrics for both FES and FCS:

- **AUROC**: Area Under ROC Curve (target: ≥0.7)
- **AUPRC**: Area Under Precision-Recall Curve
- **Accuracy**: Overall correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **Brier Score**: Calibration metric
- **ECE**: Expected Calibration Error

## 🗂️ Project Structure

```
FESTA-4x14-Strategy/
├── src/
│   ├── festa_nested_loop.py       # Main nested loop implementation
│   ├── festa_evaluation.py        # LLaVA model wrapper
│   ├── complement_generator.py    # FES/FCS sample generation
│   ├── prompts_text.py           # Text prompt templates
│   ├── prompts_image.py          # Image prompt templates
│   └── festa_metrics.py          # Metrics calculation
├── run_nested_loop.py            # GPU-optimized launcher
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── README.md                     # This file
└── FESTA_4X14_DEPENDENCIES.md   # Technical documentation
```

## 🔬 Technical Details

### Nested Loop Implementation

```python
# Outer loop: 4 MCQ variations
for mcq_idx, mcq_text in enumerate(fes_mcqs, 1):
    # Inner loop: 14 image variations
    for img_idx, img_path in enumerate(fes_images, 1):
        # Run inference with probability-based top-k
        topk_results = llava.run_topk_inference(
            image, mcq_text,
            options={'A': 'yes', 'B': 'no'},
            k=4, n_samples=5
        )
        # Store prediction
        predictions.append({
            'mcq_index': mcq_idx,
            'image_index': img_idx,
            'prediction': pred,
            'confidence': conf,
            'is_correct': is_correct
        })
```

### GPU Optimization

- **4-bit Quantization**: Reduces memory footprint from ~14GB to ~4GB
- **TF32**: Faster matrix operations on RTX 5090
- **Batch Processing**: Parallel inference with batch_size=2
- **Mixed Precision**: FP16 for speed, FP32 for accuracy-critical operations

## 📝 Output Format

Results are saved in JSON format:

```json
{
  "timestamp": "20251108_121545",
  "strategy": "4×14 Nested Loop",
  "num_samples": 3,
  "total_fes": 168,
  "total_fcs": 168,
  "results": [...],
  "fes_predictions": [...],
  "fcs_predictions": [...]
}
```

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@misc{festa2025,
  title={FESTA: Functionally Equivalent and Semantically Targeted Augmentation for VLM Evaluation},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/YOUR_USERNAME/FESTA-4x14-Strategy}}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

- GitHub Issues: [https://github.com/YOUR_USERNAME/FESTA-4x14-Strategy/issues](https://github.com/YOUR_USERNAME/FESTA-4x14-Strategy/issues)
- Email: your.email@example.com

## 🙏 Acknowledgments

- **BLINK Benchmark**: Spatial reasoning dataset
- **LLaVA**: Vision-language model
- **OpenAI**: GPT-4o-mini for text generation
- **Google**: Gemini Pro for image analysis

---

**Last Updated**: November 8, 2025

# FESTA-4x14-Strategy
