#!/usr/bin/env python3
"""
FESTA 4x14 Combination Strategy Runner
Extends existing FESTA with 4 text × 14 image combinations
"""

import os
import sys
from pathlib import Path

# Set environment for 3 samples
os.environ['NUM_SAMPLES'] = '3'
os.environ['SKIP_SAMPLES'] = '0'

# Import the main festa with APIs
sys.path.insert(0, str(Path(__file__).parent.parent))

# Execute the existing FESTA logic
from src import festa_with_apis

if __name__ == '__main__':
    print("=" * 80)
    print("FESTA 4x14 COMBINATION STRATEGY")
    print("=" * 80)
    print("Processing 3 samples with enhanced combination strategy:")
    print("  - 4 FES text paraphrases")
    print("  - 14 FES image variants  ")
    print("  - 4 × 14 = 56 FES combinations per sample")
    print("  - 4 FCS text contradictions")
    print("  - 14 FES image variants (reused)")
    print("  - 4 × 14 = 56 FCS combinations per sample")
    print("  - Total: 112 combinations per sample")
    print("=" * 80)
    print()
    
    festa_with_apis.main()

