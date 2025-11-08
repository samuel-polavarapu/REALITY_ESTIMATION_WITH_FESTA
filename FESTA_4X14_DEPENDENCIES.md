# FESTA 4x14 Strategy - File Dependencies
## Dependency Diagram
```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXECUTION LAYER                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  run_4x14_strategy.sh  ────────────────────────────────────────────────┐ │
│         │                                                               │ │
│         └──> festa_4x14_gpu.py                                         │ │
│                  │                                                      │ │
│                  │ [Sets Environment Variables]                        │ │
│                  │  • BATCH_SIZE=2                                     │ │
│                  │  • NUM_SAMPLES=3                                    │ │
│                  │  • USE_4BIT_QUANTIZATION=true                       │ │
│                  │  • GENERATE_DOCS=false                              │ │
│                  │                                                      │ │
│                  │ [Patches ComplementGenerator]                       │ │
│                  │  • _generate_fes_image_variants → 14 variants       │ │
│                  │                                                      │ │
│                  └──> festa_with_apis.py (main())                      │ │
│                           │                                             │ │
└───────────────────────────┼─────────────────────────────────────────────┘ │
                            │                                               │
┌───────────────────────────┼─────────────────────────────────────────────┐ │
│                      CORE EVALUATION LAYER                              │ │
├───────────────────────────┼─────────────────────────────────────────────┤ │
│                           │                                             │ │
│  festa_with_apis.py  <────┘                                            │ │
│         │                                                               │ │
│         ├──> complement_generator.py                                   │ │
│         │         │                                                     │ │
│         │         ├──> GPT4TextTransformer                             │ │
│         │         │      • generate_fes_paraphrases(n=4)               │ │
│         │         │      • generate_fcs_contradiction(n=4)             │ │
│         │         │                                                     │ │
│         │         └──> GeminiImageTransformer                          │ │
│         │                • generate_fes_image_variants(n=14) ◄─PATCHED │ │
│         │                • generate_fcs_image_contradiction(n=4)       │ │
│         │                                                               │ │
│         ├──> festa_evaluation.py                                       │ │
│         │         │                                                     │ │
│         │         ├──> LLaVAModel                                      │ │
│         │         │      • run_topk_inference(k=4, n_samples=20)      │ │
│         │         │      • get_combined_pred()                         │ │
│         │         │                                                     │ │
│         │         └──> Config                                          │ │
│         │                • BATCH_SIZE (from env)                       │ │
│         │                • USE_4BIT_QUANTIZATION (from env)            │ │
│         │                                                               │ │
│         └──> festa_metrics.py                                          │ │
│                   • generate_comprehensive_report() [DISABLED]         │ │
│                   • FESTAMetrics                                       │ │
│                   • FESTAVisualizer                                    │ │
│                                                                          │ │
└──────────────────────────────────────────────────────────────────────────┘ │
                                                                             │
┌────────────────────────────────────────────────────────────────────────┐  │
│                      SUPPORT LAYER                                     │  │
├────────────────────────────────────────────────────────────────────────┤  │
│                                                                         │  │
│  prompts_text.py                                                       │  │
│    • get_fes_text_prompt()                                            │  │
│    • get_fcs_text_prompt()                                            │  │
│                                                                         │  │
│  prompts_image.py                                                      │  │
│    • get_fes_image_prompt()                                           │  │
│    • get_fcs_image_prompt()                                           │  │
│                                                                         │  │
│  metrics_csv_export.py                                                 │  │
│    • export_metrics_to_csv()                                          │  │
│                                                                         │  │
└────────────────────────────────────────────────────────────────────────┘  │
                                                                             │
┌────────────────────────────────────────────────────────────────────────┐  │
│                      EXTERNAL APIS                                     │  │
├────────────────────────────────────────────────────────────────────────┤  │
│                                                                         │  │
│  OpenAI API (gpt-4o-mini)                                             │  │
│    • Text paraphrasing (FES)                                          │  │
│    • Text contradiction (FCS)                                         │  │
│                                                                         │  │
│  Gemini Pro API                                                        │  │
│    • Image analysis                                                    │  │
│    • PIL transformations (local)                                      │  │
│                                                                         │  │
│  Hugging Face Models                                                   │  │
│    • llava-hf/llava-v1.6-mistral-7b-hf                               │  │
│    • 4-bit quantization                                               │  │
│                                                                         │  │
└────────────────────────────────────────────────────────────────────────┘  │
                                                                             │
┌────────────────────────────────────────────────────────────────────────┐  │
│                      OUTPUT FILES                                      │  │
├────────────────────────────────────────────────────────────────────────┤  │
│                                                                         │  │
│  output/api_run/                                                       │  │
│    ├── api_evaluation_results.json                                    │  │
│    ├── api_evaluation_report.md                                       │  │
│    └── generated_samples/                                             │  │
│         ├── sample_{id}_original.png                                  │  │
│         ├── sample_{id}_fes_text_{1..4}.json                         │  │
│         ├── sample_{id}_fcs_text_{1..4}.json                         │  │
│         ├── sample_{id}_fes_variant_{1..14}.png                      │  │
│         └── sample_{id}_fcs_contradiction_{1..4}.png                 │  │
│                                                                         │  │
│  reports/                                                              │  │
│    └── festa_report_YYYYMMDD_HHMMSS.json                             │  │
│                                                                         │  │
│  logs/                                                                 │  │
│    └── festa_4x14_gpu_quantized.log                                  │  │
│                                                                         │  │
└────────────────────────────────────────────────────────────────────────┘  │
                                                                             │
                                                                             │
════════════════════════════════════════════════════════════════════════════┘
```
## Detailed File Dependencies
### 1. **festa_4x14_gpu.py** (NEW - Entry Point)
**Depends On:**
- `torch` - GPU configuration
- `complement_generator.py` - Patches image generation
- `festa_with_apis.py` - Executes main evaluation
**Dependencies:**
```python
import os
import torch
import sys
from pathlib import Path
from src import complement_generator
from src import festa_with_apis
```
**What It Does:**
1. Configures GPU settings (TF32, quantization, batch size)
2. Sets environment variables for configuration
3. Patches `ComplementGenerator._generate_fes_image_variants` to generate 14 instead of 4
4. Disables documentation generation
5. Calls `festa_with_apis.main()`
**Modified Behavior:**
- Original: 4 image variants per sample
- New: 14 image variants per sample (via monkey patching)
---
### 2. **complement_generator.py** (MODIFIED)
**Changes Made:**
- Added `num_variants` parameter support in `_generate_fes_image_variants()`
- Added `num_variants` parameter support in `_generate_fcs_image_contradiction()`
- Added `n` parameter support in `_generate_fes_mcq_paraphrase()`
- Added `n` parameter support in `_generate_fcs_mcq_contradiction()`
**Modified Methods:**
```python
def _generate_fes_image_variants(self, item: Dict[str, Any]):
    # Now reads: num_variants = item.get('num_variants', 4)
    # Allows festa_4x14_gpu.py to inject num_variants=14
def _generate_fes_mcq_paraphrase(self, item: Dict[str, Any]):
    # Now reads: n = item.get('n', 4)
def _generate_fcs_mcq_contradiction(self, item: Dict[str, Any]):
    # Now reads: n = item.get('n', 4)
```
**Depends On:**
- `prompts_text.py` - Text generation prompts
- `prompts_image.py` - Image generation prompts
- `openai` package - GPT-4o-mini API
- `google.generativeai` - Gemini Pro API
---
### 3. **festa_with_apis.py** (MODIFIED)
**Changes Made:**
- Added conditional documentation generation based on environment variables
- Respects `GENERATE_DOCS`, `GENERATE_REPORTS` flags
**Modified Section:**
```python
# Around line 737
generate_docs = os.getenv('GENERATE_DOCS', 'true').lower() != 'false'
generate_reports = os.getenv('GENERATE_REPORTS', 'true').lower() != 'false'
if generate_docs or generate_reports:
    comprehensive_metrics = generate_comprehensive_report(...)
else:
    logger.info("⊗ Documentation generation disabled")
    # Saves only basic JSON
```
**Depends On:**
- `complement_generator.py` - Generate FES/FCS samples
- `festa_evaluation.py` - LLaVA model inference
- `festa_metrics.py` - Metrics calculation (conditional)
- `metrics_csv_export.py` - CSV export (conditional)
---
### 4. **run_4x14_strategy.sh** (NEW - Execution Script)
**Contents:**
```bash
#!/bin/bash
echo "FESTA 4x14 Strategy - GPU Optimized"
python src/festa_4x14_gpu.py 2>&1 | tee logs/festa_4x14_gpu_$(date +%Y%m%d_%H%M%S).log
```
**Depends On:**
- `festa_4x14_gpu.py`
- bash shell
- Python 3 interpreter
---
## Data Flow
```
┌──────────────────┐
│ User Runs Script │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────┐
│ run_4x14_strategy.sh    │
│ (Bash Launcher)         │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ festa_4x14_gpu.py                       │
│ • Set env: BATCH_SIZE=2                 │
│ • Set env: USE_4BIT_QUANTIZATION=true   │
│ • Set env: GENERATE_DOCS=false          │
│ • Patch: num_variants=14                │
└────────┬────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ festa_with_apis.py                       │
│ Loop for each sample (3):                │
│   ├─ Initialize ComplementGenerator      │
│   ├─ Initialize LLaVAModel (w/ 4-bit)   │
│   └─ Load BLINK dataset                  │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ FOR EACH SAMPLE:                             │
│                                              │
│ 1. Generate FES Text (4 paraphrases)        │
│    └─> OpenAI API → 4 variations            │
│                                              │
│ 2. Generate FES Image (14 variants)         │
│    └─> PIL Transforms → 14 images           │
│         (PATCHED from 4 to 14)               │
│                                              │
│ 3. Generate FCS Text (4 contradictions)     │
│    └─> OpenAI API → 4 variations            │
│                                              │
│ 4. Generate FCS Image (4 contradictions)    │
│    └─> PIL Transforms → 4 images            │
│                                              │
│ 5. Run Inference on All Variants            │
│    ├─> LLaVA Model (4-bit quantized)        │
│    ├─> Top-k sampling (k=4, n=20)           │
│    └─> Get combined predictions             │
│                                              │
└────────┬─────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ Calculate Metrics                      │
│ • AUROC, AUPRC                         │
│ • Accuracy, Precision, Recall          │
│ • Brier Score, ECE                     │
│ • Per type: FES/FCS, Text/Image        │
└────────┬───────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Save Results                            │
│ ├─ JSON: api_evaluation_results.json   │
│ ├─ JSON: festa_report_*.json           │
│ ├─ MD: api_evaluation_report.md        │
│ └─ Files: generated_samples/*          │
└─────────────────────────────────────────┘
```
## Environment Variables Flow
```
festa_4x14_gpu.py Sets:
├─ CUDA_VISIBLE_DEVICES=0
├─ PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
├─ BATCH_SIZE=2
├─ NUM_INFERENCE_SAMPLES=20
├─ MAX_NEW_TOKENS=64
├─ USE_4BIT_QUANTIZATION=true
├─ USE_MIXED_PRECISION=true
├─ NUM_SAMPLES=3
├─ SKIP_SAMPLES=0
├─ GENERATE_DOCS=false
├─ GENERATE_REPORTS=false
├─ SAVE_VISUALIZATIONS=false
├─ GENERATE_HTML=false
└─ GENERATE_MARKDOWN=false
        │
        ▼
festa_with_apis.py Reads:
├─ NUM_SAMPLES → Number of samples to process
├─ SKIP_SAMPLES → Starting index
├─ GENERATE_DOCS → Skip comprehensive reports if false
└─ GENERATE_REPORTS → Skip HTML/MD reports if false
        │
        ▼
festa_evaluation.py (Config) Reads:
├─ BATCH_SIZE → Parallel inference count
├─ NUM_INFERENCE_SAMPLES → Sampling iterations
├─ MAX_NEW_TOKENS → Generation length
├─ USE_4BIT_QUANTIZATION → Memory optimization
└─ USE_MIXED_PRECISION → Speed optimization
        │
        ▼
torch/CUDA Uses:
├─ CUDA_VISIBLE_DEVICES → GPU selection
└─ PYTORCH_CUDA_ALLOC_CONF → Memory management
```
## Function Call Chain
```
run_4x14_strategy.sh
  └─> festa_4x14_gpu.py
       ├─> torch.backends.cuda.matmul.allow_tf32 = True
       ├─> torch.backends.cudnn.allow_tf32 = True
       ├─> os.environ['...'] = '...' (×13 vars)
       ├─> [PATCH] ComplementGenerator._generate_fes_image_variants
       └─> festa_with_apis.main()
            ├─> ComplementGenerator.__init__()
            │    ├─> GPT4TextTransformer.__init__()
            │    └─> GeminiImageTransformer.__init__()
            ├─> LLaVAModel.__init__()
            │    ├─> Config.setup_environment()
            │    ├─> LlavaNextProcessor.from_pretrained()
            │    └─> LlavaNextForConditionalGeneration.from_pretrained()
            │         └─> BitsAndBytesConfig(load_in_4bit=True)
            └─> for sample in dataset:
                 ├─> complement_gen.generate_complement(type='fes', item_type='mcq')
                 │    └─> _generate_fes_mcq_paraphrase()
                 │         └─> text_transformer.generate_fes_paraphrases(n=4)
                 │              └─> OpenAI API Call
                 ├─> complement_gen.generate_complement(type='fes', item_type='image')
                 │    └─> _generate_fes_image_variants(item['num_variants']=14) ◄─PATCHED
                 │         └─> image_transformer.generate_fes_image_variants(n=14)
                 │              └─> PIL Image Transformations (×14)
                 ├─> complement_gen.generate_complement(type='fcs', item_type='mcq')
                 │    └─> _generate_fcs_mcq_contradiction()
                 │         └─> text_transformer.generate_fcs_contradiction(n=4)
                 │              └─> OpenAI API Call
                 ├─> complement_gen.generate_complement(type='fcs', item_type='image')
                 │    └─> _generate_fcs_image_contradiction()
                 │         └─> image_transformer.generate_fcs_image_contradiction(n=4)
                 │              └─> PIL Image Transformations (×4)
                 └─> for variant in all_variants:
                      └─> llava.run_topk_inference(k=4, n_samples=20)
                           ├─> strict_prompt(k=4)
                           ├─> processor.apply_chat_template()
                           ├─> model.generate(do_sample=True, temperature=0.7)
                           └─> get_combined_pred(topk_results)
```
## Key Modification Points
### 1. Monkey Patching (festa_4x14_gpu.py)
```python
# Original method reference
_original_generate_fes_image = complement_generator.ComplementGenerator._generate_fes_image_variants
# Wrapper that injects num_variants
def patched_generate_fes_image(self, item):
    item['num_variants'] = 14  # ← Force 14 variants
    return _original_generate_fes_image(self, item)
# Replace method
complement_generator.ComplementGenerator._generate_fes_image_variants = patched_generate_fes_image
```
### 2. Parameter Injection (complement_generator.py)
```python
# Modified to accept item['num_variants']
def _generate_fes_image_variants(self, item: Dict[str, Any]):
    num_variants = item.get('num_variants', 4)  # ← Default 4, override via item dict
    images, logs = self.image_transformer.generate_fes_image_variants(
        reference_image=image,
        question=question,
        num_variants=num_variants,  # ← Use overridden value
        base_name=base_name
    )
```
### 3. Conditional Execution (festa_with_apis.py)
```python
# Check environment flags
generate_docs = os.getenv('GENERATE_DOCS', 'true').lower() != 'false'
generate_reports = os.getenv('GENERATE_REPORTS', 'true').lower() != 'false'
if generate_docs or generate_reports:
    # Generate full reports
    comprehensive_metrics = generate_comprehensive_report(...)
else:
    # Skip expensive report generation
    logger.info("⊗ Documentation generation disabled")
    comprehensive_metrics = None
```
## Summary of New Files
1. **festa_4x14_gpu.py** - Main entry point with GPU optimization
2. **run_4x14_strategy.sh** - Bash launcher script
## Summary of Modified Files
1. **complement_generator.py** - Added configurable num_variants/n parameters
2. **festa_with_apis.py** - Added conditional documentation generation
3. **prompts_image.py** - Fixed syntax error (removed corrupted lines)
## Integration Points
The 4x14 strategy integrates with existing code at these points:
1. **Environment Variables** → `festa_evaluation.Config`
2. **Monkey Patch** → `complement_generator.ComplementGenerator`
3. **Conditional Logic** → `festa_with_apis.main()`
4. **GPU Configuration** → PyTorch/CUDA backends
---
*Generated: November 8, 2025*
*For: FESTA 4x14 Combination Strategy Implementation*
