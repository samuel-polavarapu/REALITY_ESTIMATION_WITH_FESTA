# FESTA System - API Usage Clarification

**Date**: October 31, 2025  
**Document Purpose**: Clarify exactly which APIs and models are used in the FESTA evaluation system

---

## Executive Summary

After thorough code review, here is the **ACTUAL** implementation:

### ✅ ACTIVELY USED

1. **OpenAI GPT-4o-mini** (Text API)
   - **Purpose**: Generate FES text (paraphrases) and FCS text (contradictions)
   - **Usage**: Active - makes API calls for every sample
   - **Rate Limit**: 3 requests/minute
   - **Cost**: ~$1.50 for 143 samples

2. **PIL (Python Imaging Library / Pillow)**
   - **Purpose**: ALL image transformations (FES and FCS)
   - **Usage**: Active - processes all images locally
   - **Execution**: CPU-based, local processing
   - **Cost**: $0 (no API calls)

3. **LLaVA v1.6 Mistral 7B**
   - **Purpose**: Primary model being evaluated
   - **Usage**: Active - runs inference on all samples
   - **Execution**: GPU-based (CUDA), local

### ❌ NOT USED (Despite Being Configured)

1. **Google Gemini Pro**
   - **Status**: API key configured in .env
   - **Code**: `GeminiImageTransformer` class exists
   - **Reality**: NOT called in actual workflow
   - **Why**: PIL handles all image transformations locally

2. **DALL-E (OpenAI Images API)**
   - **Status**: Class definition exists (`OpenAIImageTransformer`)
   - **Code**: References DALL-E 2 model
   - **Reality**: Class is NEVER instantiated
   - **Why**: Not part of the active workflow

---

## Detailed Breakdown

### Text Generation: OpenAI GPT-4o-mini

**File**: `src/complement_generator.py`  
**Class**: `GPT4TextTransformer`

**Model ID**: `gpt-4o-mini` (configurable via `OPENAI_TEXT_MODEL` env var)

**What it does**:
```python
# FES Text Generation
def generate_fes_paraphrases(self, question: str, n: int = 4) -> List[str]:
    # Calls OpenAI Chat Completions API
    # Generates semantic paraphrases
    # Example: "Is A left of B?" → "Is A positioned to the left of B?"
    
# FCS Text Generation  
def generate_fcs_contradiction(self, question: str, n: int = 1) -> List[str]:
    # Calls OpenAI Chat Completions API
    # Generates contradictory questions
    # Example: "Is A left of B?" → "Is A right of B?"
```

**API Calls Per Sample**:
- 1 call for FES text generation (generates 5 paraphrases)
- 1 call for FCS text generation (generates 5 contradictions)
- Total: 2 API calls per sample

**For 143 samples**: 286 API calls total

---

### Image Generation: PIL (LOCAL)

**File**: `src/complement_generator.py`  
**Class**: `GeminiImageTransformer` (misleading name - actually uses PIL)

**What it ACTUALLY does**:

```python
def _generate_images_from_prompt(self, image: Image.Image, prompt: str, n: int):
    """
    Despite the class name, this function uses PIL LOCALLY.
    NO external API calls are made.
    """
    for i in range(n):
        transformed = image.copy()
        
        if is_fes:
            # FES: Relation-preserving transformations
            # - Gaussian noise (σ ∈ [0.003, 0.02])
            # - Gaussian blur (σ ∈ [0.3, 1.0])
            # - Contrast (factor ∈ [0.90, 1.10])
            # - Brightness (shift ∈ [-0.03, +0.03])
            # - Grayscale conversion
            # - Dotted masking (1-3% pixels)
            
        else:  # FCS
            # FCS: Relation-flipping transformations
            # - Horizontal flip (left ↔ right)
            # - Vertical flip (above ↔ below)
            # - Rotation 180° (both axes)
            # Plus subtle photometric adjustments
```

**Key Operations**:
- `ImageFilter.GaussianBlur()` - blur
- `ImageEnhance.Contrast()` - contrast
- `ImageEnhance.Brightness()` - brightness
- `ImageOps.mirror()` - horizontal flip
- `ImageOps.flip()` - vertical flip
- `image.rotate()` - rotation
- Numpy operations for noise and masking

**API Calls**: **ZERO** - Everything runs locally on CPU

---

### Why the Confusion?

#### Class Name Mismatch
The class is named `GeminiImageTransformer` but doesn't actually use Gemini:

```python
class GeminiImageTransformer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if genai is not None:
            genai.configure(api_key=self.api_key)  # Configured but not used
        self.model = genai.GenerativeModel(GOOGLE_GEMINI_MODEL)  # Created but not called
```

The `analyze_image_relations()` method in this class CAN call Gemini, but it's **not used in the main workflow**.

#### DALL-E Class Exists But Unused

```python
class OpenAIImageTransformer:
    """
    This class exists in the code but is NEVER instantiated
    in ComplementGenerator or the main evaluation script.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=self.api_key)
        self.model_id = "dall-e-2"
```

This class is **dead code** - not part of the active execution path.

---

## Verification: What Gets Instantiated?

**File**: `src/complement_generator.py`  
**Class**: `ComplementGenerator.__init__()`

```python
class ComplementGenerator:
    def __init__(self, output_dir: str = 'output/complements', seed: int = 42):
        # TEXT: Uses OpenAI
        self.text_transformer = GPT4TextTransformer()  # ✅ USED
        
        # IMAGES: Uses PIL (despite the class name)
        self.image_transformer = GeminiImageTransformer()  # ✅ USED (but calls PIL, not Gemini)
        
        # NOTE: OpenAIImageTransformer is NEVER instantiated
```

---

## Cost Analysis

### Actual Costs (143 samples)

| Component | API Used | Calls | Estimated Cost |
|-----------|----------|-------|----------------|
| Text FES | OpenAI GPT-4o-mini | 143 | ~$0.75 |
| Text FCS | OpenAI GPT-4o-mini | 143 | ~$0.75 |
| Image FES | PIL (local) | 0 | $0.00 |
| Image FCS | PIL (local) | 0 | $0.00 |
| **TOTAL** | | **286** | **~$1.50** |

### What You're NOT Paying For

- ❌ Google Gemini API calls: $0 (not used)
- ❌ DALL-E image generation: $0 (not used)
- ❌ Image analysis APIs: $0 (not used)

---

## Environment Variables

**Required** (actually used):
```bash
OPENAI_API_KEY=sk-...     # ✅ Used for text generation
HF_TOKEN=hf_...           # ✅ Used for LLaVA model
```

**Configured but NOT used**:
```bash
GOOGLE_API_KEY=AIza...    # ❌ Configured but API not called
```

**Not needed**:
```bash
# No DALL-E specific key needed - not used
```

---

## Performance Implications

### Why PIL Instead of APIs?

**Advantages**:
1. **Speed**: No network latency
2. **Cost**: $0 for image transformations
3. **Control**: Deterministic results with seed control
4. **Scalability**: Can process multiple images in parallel locally
5. **Reliability**: No API rate limits or failures

**Disadvantages**:
1. **Quality**: PIL transformations are simpler than AI-generated variants
2. **Creativity**: Less diverse transformations compared to generative models
3. **Analysis**: No semantic understanding of image content

### Runtime Breakdown (143 samples)

| Phase | Processing | Time |
|-------|-----------|------|
| Text generation (OpenAI) | 286 API calls @ 21s each | ~1.7 hours |
| Image transformation (PIL) | 143 × 8 images locally | ~10 minutes |
| LLaVA inference | 143 × 19 predictions | ~6-8 hours |
| Metrics calculation | Local processing | ~5 minutes |
| **TOTAL** | | **~8-10 hours** |

---

## Code Evidence

### OpenAI API Calls (Text)

**File**: `src/complement_generator.py`, Line ~125

```python
response = self.client.chat.completions.create(
    model=self.model_id,  # 'gpt-4o-mini'
    messages=[{"role": "user", "content": f"SYSTEM: {system_prompt}\n\n{user_prompt}"}],
    temperature=0.7,
    max_tokens=500,
    n=1
)
```

### PIL Image Processing (Local)

**File**: `src/complement_generator.py`, Line ~335

```python
# FES transformations
for op in selected_ops:
    if op == 'noise':
        sigma = random.uniform(0.003, 0.02)
        img_array = np.array(transformed).astype(np.float32) / 255.0
        noise = np.random.normal(0, sigma, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 1)
        transformed = Image.fromarray((img_array * 255).astype(np.uint8))
        
    elif op == 'blur':
        sigma = random.uniform(0.3, 1.0)
        transformed = transformed.filter(ImageFilter.GaussianBlur(radius=sigma))
```

### Gemini NOT Called

**File**: `src/complement_generator.py`

The `analyze_image_relations()` method exists but is **never called** in the workflow:

```python
def analyze_image_relations(self, image: Image.Image, question: str) -> Dict[str, Any]:
    # This method CAN call Gemini, but it's NOT invoked in the main execution path
    # The _generate_images_from_prompt() method doesn't call this
```

---

## Recommendations

### For Documentation
- ✅ Clarify that only OpenAI is used for API calls
- ✅ Emphasize PIL handles all image processing locally
- ✅ Note Gemini is configured but not actively used
- ✅ Remove any implications that DALL-E is involved

### For Future Enhancement
If you want to actually use the configured APIs:

1. **Gemini Pro**: Uncomment/implement calls in `analyze_image_relations()`
2. **DALL-E**: Instantiate `OpenAIImageTransformer` instead of using PIL
3. **Trade-offs**: Higher quality but slower + more expensive

---

## Summary Table

| Component | Status | Type | Usage |
|-----------|--------|------|-------|
| **OpenAI GPT-4o-mini** | ✅ Active | Cloud API | Text FES/FCS generation |
| **PIL/Pillow** | ✅ Active | Local Library | ALL image transformations |
| **LLaVA v1.6** | ✅ Active | Local Model (GPU) | Primary evaluation target |
| **Google Gemini Pro** | ⚠️ Configured | Cloud API | Reserved, not called |
| **DALL-E** | ❌ Unused | Cloud API | Class exists but not used |

---

## Conclusion

**The FESTA system uses**:
1. **OpenAI GPT-4o-mini** for text generation (active, ~$1.50 for 143 samples)
2. **PIL** for ALL image transformations (local, $0 cost)
3. **LLaVA** for inference (local GPU, $0 API cost)

**The FESTA system does NOT use**:
1. Google Gemini Pro (configured but not called)
2. DALL-E (class exists but not instantiated)
3. Any other image generation APIs

This is a **cost-effective, fast, and reliable** implementation that prioritizes local processing over external API dependencies for image transformations.

