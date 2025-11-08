#!/usr/bin/env python3
"""
FESTA Nested Loop Strategy - GPU Optimized Launcher
4 MCQs × 14 Images = 56 combinations per type (FES/FCS)
"""

import os
import torch

print("Configuring GPU for nested loop strategy...")

# GPU Optimization
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Configuration
os.environ['BATCH_SIZE'] = '2'
os.environ['NUM_INFERENCE_SAMPLES'] = '20'
os.environ['MAX_NEW_TOKENS'] = '64'
os.environ['USE_4BIT_QUANTIZATION'] = 'true'
os.environ['USE_MIXED_PRECISION'] = 'true'

# Samples
os.environ['NUM_SAMPLES'] = '3'
os.environ['SKIP_SAMPLES'] = '0'

# Disable documentation
os.environ['GENERATE_DOCS'] = 'false'
os.environ['GENERATE_REPORTS'] = 'false'

print(f"✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"✓ 4-bit Quantization: Enabled")
print(f"✓ TF32: Enabled")
print()

# Run nested loop strategy
from src import festa_nested_loop
festa_nested_loop.main()

