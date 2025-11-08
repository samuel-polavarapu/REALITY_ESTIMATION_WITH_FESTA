#!/usr/bin/env python3
"""
FESTA with 4x14 Strategy - Modified to generate 14 image variants
and run 4 text × 14 image = 56 combinations per type (FES/FCS)
"""
import os
os.environ['NUM_SAMPLES'] = '3'
os.environ['SKIP_SAMPLES'] = '0'
# Modify the existing festa_with_apis to use 14 image variants
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
# Import and patch before running
from src import complement_generator
# Store original methods
_original_generate_fes_image = complement_generator.ComplementGenerator._generate_fes_image_variants
def patched_generate_fes_image(self, item):
    """Patched version that generates 14 variants instead of 4"""
    # Set num_variants in item
    item['num_variants'] = 14
    return _original_generate_fes_image(self, item)
# Apply patch
complement_generator.ComplementGenerator._generate_fes_image_variants = patched_generate_fes_image
print("="*80)
print("FESTA 4x14 COMBINATION STRATEGY")
print("="*80)
print("Enhanced to generate:")
print("  • 4 FES text paraphrases")
print("  • 14 FES image variants (increased from 4)")
print("  • 4 × 14 = 56 FES combinations")
print("  • 4 FCS text contradictions")
print("  • Using same 14 image variants")
print("  • 4 × 14 = 56 FCS combinations")
print("  • Total: 112 combinations per sample × 3 samples = 336 total")
print("="*80)
print()
# Now run the main FESTA program
from src import festa_with_apis
festa_with_apis.main()
