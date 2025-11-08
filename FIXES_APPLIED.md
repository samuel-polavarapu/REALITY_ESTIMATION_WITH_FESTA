# Fixes Applied to FESTA FES Image Generation Issue

## Problem Statement
The program was failing to generate FES Images with the error:
```
ERROR - Failed to generate complement: '33, 55'
```

## Root Cause Analysis
The error `KeyError: '33, 55'` occurred when Python's `.format()` method encountered unescaped curly braces `{3×3, 5×5}` in the FES_IMAGE_SYSTEM_PROMPT template in `src/prompts_image.py`. Python interpreted these as format placeholders but couldn't find matching arguments.

## Fixes Applied

### Fix 1: `src/prompts_image.py` (Line 41)
**Issue**: Curly braces in kernel size notation
```python
# BEFORE (causing KeyError: '33, 55')
- Mild blur (Gaussian): kernel ∈ {3×3, 5×5}, σ ∈ [0.3, 1.0] pixels.

# AFTER (escaped braces)
- Mild blur (Gaussian): kernel ∈ {{3×3, 5×5}}, σ ∈ [0.3, 1.0] pixels.
```

### Fix 2: `src/prompts_image.py` (Line 103)
**Issue**: Unescaped placeholder in JSON example
```python
# BEFORE (causing KeyError: 'base_name')
"filename": "{base_name}_fes_01.png"

# AFTER (escaped to remain as literal in output)
"filename": "{{base_name}}_fes_01.png"
```

### Fix 3: `src/festa_with_apis.py` (Line 28)
**Issue**: Import path not finding src module
```python
# BEFORE
sys.path.insert(0, str(Path(__file__).parent))

# AFTER (add project root to path)
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Fix 4: Missing Dependencies
**Issue**: Required packages not installed
```bash
# Installed packages
pip install openai google-generativeai
pip install pyarrow==21.0.0
pip install --upgrade datasets
```

## Verification

### Test Command
```bash
cd /data/sam/Kaggle/code/LLAVA-V5-2
NUM_SAMPLES=2 SKIP_SAMPLES=0 python3 src/festa_with_apis.py
```

### Test Results
✅ **All 32 samples generated successfully** (8 FES Text, 8 FES Image, 8 FCS Text, 8 FCS Image)

#### Sample Output:
```
2025-10-28 18:21:40,362 - INFO - Using OpenAI DALL-E 3 to generate 4 FES image variants...
2025-10-28 18:21:40,412 - INFO - Generated 4 FES images using PIL transformations
2025-10-28 18:21:40,593 - INFO -   ✓ FES Image 1 generated and saved to output/api_run/generated_samples/sample_1_fes_image_fes_variant_1.png
2025-10-28 18:21:40,593 - INFO -   ✓ FES Image 2 generated and saved to output/api_run/generated_samples/sample_1_fes_image_fes_variant_2.png
2025-10-28 18:21:40,593 - INFO -   ✓ FES Image 3 generated and saved to output/api_run/generated_samples/sample_1_fes_image_fes_variant_3.png
2025-10-28 18:21:40,593 - INFO -   ✓ FES Image 4 generated and saved to output/api_run/generated_samples/sample_1_fes_image_fes_variant_4.png
```

## Files Modified

1. `/data/sam/Kaggle/code/LLAVA-V5-2/src/prompts_image.py`
   - Line 41: Escaped curly braces in kernel notation
   - Line 103: Escaped base_name placeholder

2. `/data/sam/Kaggle/code/LLAVA-V5-2/src/festa_with_apis.py`
   - Line 28: Fixed sys.path to include project root

## Impact

### Before Fix
- FES Image generation: ❌ Failed
- FES Text generation: ✅ Working
- FCS Image generation: ✅ Working
- FCS Text generation: ✅ Working

### After Fix
- FES Image generation: ✅ **FIXED**
- FES Text generation: ✅ Working
- FCS Image generation: ✅ Working
- FCS Text generation: ✅ Working

## FES Image Generation Details

The FES image generation now properly applies PIL-based transformations:

1. **Gaussian Noise**: σ ∈ [0.003, 0.02]
2. **Gaussian Blur**: σ ∈ [0.3, 1.0]
3. **Contrast Adjustment**: factor ∈ [0.90, 1.10]
4. **Brightness Adjustment**: shift ∈ [-0.03, +0.03]

Each variant randomly selects 1-3 operations from the above list, ensuring spatial relations are preserved.

## Conclusion

All issues have been resolved. The FESTA evaluation system can now successfully generate:
- ✅ FES Text (paraphrases)
- ✅ FES Images (relation-preserving variants)
- ✅ FCS Text (contradictions)
- ✅ FCS Images (relation-flipping variants)

The system is ready for full-scale evaluation runs.

