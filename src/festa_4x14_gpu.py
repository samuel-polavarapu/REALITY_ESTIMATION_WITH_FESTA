#!/usr/bin/env python3
"""
FESTA with 4x14 Strategy - GPU Optimized
Generates 14 image variants and runs 4x14=56 combinations
Maximizes RTX 5090 24GB GPU utilization
"""
import os
import torch
# ============================================================================
# GPU OPTIMIZATION CONFIGURATION
# ============================================================================
print("Configuring GPU for maximum performance...")
# Use all available GPU memory
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
# Enable performance optimizations for RTX 5090
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# Increase GPU utilization parameters
os.environ['BATCH_SIZE'] = '2'  # Reduce batch size to save memory
os.environ['NUM_INFERENCE_SAMPLES'] = '20'  # More samples = better GPU usage
os.environ['MAX_NEW_TOKENS'] = '64'  # Longer generation = more GPU work
os.environ['USE_4BIT_QUANTIZATION'] = 'true'  # Enable quantization to fit in available GPU memory
os.environ['USE_MIXED_PRECISION'] = 'true'  # Use FP16 for speed
# Sample configuration  
os.environ['NUM_SAMPLES'] = '3'
os.environ['SKIP_SAMPLES'] = '0'
# DISABLE ALL DOCUMENTATION GENERATION
os.environ['GENERATE_DOCS'] = 'false'
os.environ['GENERATE_REPORTS'] = 'false'
os.environ['SAVE_VISUALIZATIONS'] = 'false'
os.environ['GENERATE_HTML'] = 'false'
os.environ['GENERATE_MARKDOWN'] = 'false'
print(f"✓ GPU Configuration Complete")
print(f"  CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
# ============================================================================
# IMPORT AND PATCH
# ============================================================================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src import complement_generator
# Patch to generate 14 image variants instead of 4
_original_generate_fes_image = complement_generator.ComplementGenerator._generate_fes_image_variants
def patched_generate_fes_image(self, item):
    """Patched version that generates 14 variants instead of 4"""
    item['num_variants'] = 14
    return _original_generate_fes_image(self, item)
complement_generator.ComplementGenerator._generate_fes_image_variants = patched_generate_fes_image
# ============================================================================
# EXECUTION
# ============================================================================
print("="*80)
print("FESTA 4x14 COMBINATION STRATEGY - GPU OPTIMIZED")
print("="*80)
print("Configuration:")
print("  • 4 FES text paraphrases")
print("  • 14 FES image variants (increased from 4)")
print("  • 4 × 14 = 56 FES combinations")
print("  • 4 FCS text contradictions")
print("  • 14 image variants reused")
print("  • 4 × 14 = 56 FCS combinations")
print("  • Total: 112 combinations/sample × 3 samples = 336 predictions")
print("")
print("GPU Optimizations:")
print("  • Batch size: 2")
print("  • Inference samples: 20")
print("  • 4-bit quantization (enabled to fit in memory)")
print("  • TF32 enabled")
print("  • Documentation generation: DISABLED")
print("="*80)
print()
# Run the main FESTA program
from src import festa_with_apis
festa_with_apis.main()
