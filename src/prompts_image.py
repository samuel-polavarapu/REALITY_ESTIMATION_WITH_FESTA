
#!/usr/bin/env python3
"""
FESTA Image Prompts for FES and FCS Generation
Consolidated prompts for image transformations
"""

# =============================================================================
# GLOBAL IMAGE CONSTRAINTS (Apply to both FES and FCS)
# =============================================================================

GLOBAL_IMAGE_CONSTRAINTS = """
Global Constraints (apply to all modes)
- Keep every original object present; do not add or remove objects.
- Maintain object identities (same categories/instances), appearance, and approximate scale.
- No cropping, no aspect-ratio change, no perspective/affine warps, no borders/frames/text.
- Output resolution must equal input resolution.
"""


# =============================================================================
# FES IMAGE PROMPTS (Functionally Equivalent Samples)
# =============================================================================

FES_IMAGE_SYSTEM_PROMPT = """You are an image-transformation assistant for generating dataset variants used in visual spatial reasoning.
You MUST preserve object identities and counts, keep the canvas size & aspect ratio unchanged, and avoid any textual overlays or artificial artifacts.

{global_constraints}

MODE: FES (Functionally Equivalent Samples)

SOURCE REQUIREMENT
- The ORIGINAL IMAGE is provided as input and is the ONLY source to transform.
- Do not synthesize from scratch, inpaint new objects, or remove existing ones.

GOAL
Produce variants that keep the model’s answer unchanged. All pairwise spatial relationships (left/right, above/below), containment/overlap, and coarse distance ordering must remain identical to the original.
Start from the ORIGINAL IMAGE and apply only subtle, relation-preserving perturbations.

ALLOWED OPS (choose 1–3 per variant; parameters MUST be within bounds)
- Gaussian noise (additive, per-pixel): σ ∈ [0.03, 0.02] in normalized [0,1] intensity; clip outputs to valid range.
- Mild blur (Gaussian): kernel ∈ {{3×3, 5×5}}, σ ∈ [0.5, 1.0] pixels.
- Global contrast & brightness (image-wide, no object-selective edits):
  • contrast multiplier ∈ [0.90, 1.10]
  • brightness shift ∈ [-0.05, +0.05] (additive in normalized [0,1])
- Grayscale conversion: Convert to grayscale while preserving spatial relations
  • Recommended: Use luminance-based conversion (e.g., 0.299*R + 0.587*G + 0.114*B)
  • Apply to entire image uniformly
- Dotted/Slight masking (sparse dots for texture variation):
  • ≤ 3% of pixels masked with tiny dots
  • dot size: 1×1 or 2×2 pixels
  • randomly distributed across image
  • mask value: mean image intensity or slight variation
  • do not cover > 0.3% area of any object
- Sparse masking (tiny neutral patches; never on salient regions):
  • ≤ 5% of pixels masked total
  • patch size ≤ 3×3
  • do not cover > 0.5% area of any annotated/visible object
- Small rotation (around image center):
  • angle θ ∈ [-6°, +6°]
  • pad with reflect or edge (no cropping)
  • verify left/right and above/below relations are unchanged
- Small translation (whole image):
  • shift |dx| ≤ 4% width, |dy| ≤ 4% height
  • pad with reflect or edge to keep all content visible

FORBIDDEN IN FES
- Flips (horizontal/vertical), perspective/affine warps, rescaling that changes aspect ratio
- Strong rotations (|θ| > 5°), large translations (> 3%)
- Color edits targeted to specific objects; local recoloring; style transfer
- Any occlusion change beyond “Sparse masking” limits
- Cropping, content loss, EXIF orientation changes
- Text overlays, watermarks, borders/frames, or symbols (including words like “NOT”)

VALIDATION (must pass for every variant)
- No object leaves the canvas; no new objects appear.
- All pairwise left/right and above/below relations are unchanged.
- Containment (inside/outside) and overlap relations are unchanged.
- Output resolution == input resolution; aspect ratio identical.
- Variant is recognizably derived from the ORIGINAL IMAGE (no style/domain shift).
- Parameter bounds respected; pad mode recorded.
(Recommended similarity checks: PSNR ≥ 28 dB and/or SSIM ≥ 0.95; if below, resample parameters.)

REPRODUCIBILITY
- Use and record a PRNG seed per variant.
- When composing multiple ops, for each variant, explicitly specify the source image for every operation:
    • If an operation is to be applied to the ORIGINAL IMAGE, state this clearly.
    • If an operation is to be applied to the result of a previous operation, specify which one.
- By default, each operation should be applied to the ORIGINAL IMAGE unless the variant definition requires cumulative transformations.
- When cumulative transformations are required (e.g., noise followed by grayscale), clearly document the sequence and intermediate sources.
- In the JSON log, for each operation, include a "source" field indicating the image (e.g., "original", "result_of_gaussian_noise", etc.).

OUTPUTS
- Produce exactly {num_variants} variants of the ORIGINAL IMAGE.
- Return images plus a compact JSON log describing each variant, including the exact sequence and source of each operation.

JSON LOG SCHEMA (array of objects)
[
  {
    "id": "fes_01",
    "mode": "FES",
    "seed": 42,
    "ops": [
      {"type": "brightness_contrast", "contrast": 1.04, "brightness": -0.02, "source": "original"},
      {"type": "gaussian_noise", "sigma": 0.008, "source": "result_of_brightness_contrast"},
      {"type": "rotate", "theta_deg": 2.1, "pad": "reflect", "source": "result_of_gaussian_noise"}
    ],
    "validation": {
      "relations_unchanged": true,
      "containment_unchanged": true,
      "overlap_unchanged": true,
      "no_object_left_canvas": true,
      "same_resolution": true,
      "psnr_db": 29.7,
      "ssim": 0.962,
      "bounds_ok": true
    },
    "filename": "{{base_name}}_fes_01.png"
  }
]

INSTRUCTIONS
- Apply only permitted ops within bounds.
- For each variant, document the transformation sequence and the source image for each operation.
- If any validation fails, resample parameters (same ops) with a new seed until all checks pass.
- Use PNG (or the input format) to avoid extra compression artifacts.
"""

FES_IMAGE_USER_PROMPT_TEMPLATE = """ORIGINAL IMAGE: Provided below and MUST be used as the base for all transformations.
Task: Generate FES variants that preserve all spatial relations and object visibility.

Input image: <ORIGINAL_IMAGE_PROVIDED>
Context question (for relation preservation): {question}
Mode: FES
num_variants: {num_variants}
base_name: {base_name}

IMPORTANT
- Transform the ORIGINAL IMAGE only (no synthesis).
- Keep canvas size and aspect ratio unchanged.
- For each variant, explicitly specify the source image for every operation in the transformation sequence.
- By default, each operation should be applied to the ORIGINAL IMAGE unless the variant definition requires cumulative transformations.
- If cumulative transformations are required, clearly document the sequence and intermediate sources.
- Follow the System Prompt exactly. Return the transformed images and the JSON log conforming to the schema, including the source for each operation.
"""



# =============================================================================
# FCS IMAGE PROMPTS (Functionally Contradictory Samples)
# =============================================================================

# FCS_IMAGE_SYSTEM_PROMPT removed - now using FCS_IMAGE_ENHANCED_PROMPT only
# FCS_IMAGE_USER_PROMPT_TEMPLATE removed - now using FCS_IMAGE_ENHANCED_PROMPT only


# =============================================================================
# ENHANCED FCS IMAGE PROMPT (Production-Ready)
# =============================================================================

FCS_IMAGE_ENHANCED_PROMPT = """You are an image-transformation assistant. Create Functionally Complementary Samples (FCS) that REVERSE the target spatial relation so the correct answer should change, while keeping the task, objects, and realism intact.

ORIGINAL IMAGE REQUIREMENT:
The original image MUST be provided as input and used as the base for ALL transformations.
Do NOT generate new images from scratch. Apply transformations TO THE PROVIDED ORIGINAL IMAGE.

INPUTS:
- Original image: {image_ref} [REQUIRED - Must be provided]
- Original question (context): {original_question}
- Target relation to invert (entities + predicate): {target_relation}
- Objects / bboxes (optional): {objects_or_bboxes}
- num_variants: {num_variants}
- base_name: {base_name}

GOAL (FCS):
- Start with the ORIGINAL IMAGE provided as input
- Change ONLY the specified spatial relation (e.g., left↔right, above↔below, under↔on top of, in front of↔behind, inside↔outside, nearer↔farther) so that the answer to the question flips.
- Keep all object identities and counts the SAME. Keep appearance and approximate scale similar.
- Preserve canvas size and aspect ratio. No cropping.

PRIMARY STRATEGY (choose EXACTLY ONE per variant):
1) Horizontal flip (preferred when the predicate is left/right). If flipping would undesirably change many unrelated left/right relations, prefer strategy (2).
2) Minimal direct repositioning of ONLY the named entities to invert the target relation. Keep both entities fully visible; avoid unrealistic deformations. Recommended bounds: translate ≤ 20% of width/height; scale change ≤ 10%; rotation |θ| ≤ 5°.

OPTIONAL SUPPORT (subtle photometric tweaks ONLY if needed):
- Contrast 0.95–1.05; brightness −0.02…+0.02; Gaussian noise σ 0.003–0.01; blur σ 0.3–0.8.
- Do NOT obscure or confound the manipulated relation.

FORBIDDEN:
- Adding/removing/renaming objects; aspect-ratio change; perspective/affine warps; heavy rotation; large scale changes; new occlusions that hide the target entity; borders/frames/text/arrows.
- No textual overlays or negation cues anywhere (do NOT use words like 'not', 'no', etc.).
- Creating new images from scratch instead of transforming the original

VALIDATION (must pass for each output):
- The target relation is inverted (e.g., 'cat left-of car' → 'cat right-of car').
- All original objects remain present, recognizable, and fully visible; identities/counts unchanged.
- Image resolution equals input; unrelated relations remain as stable as feasible.
- Output images are recognizably derived from the ORIGINAL IMAGE provided

OUTPUTS:
- Produce exactly {num_variants} images named {base_name}_fcs_{{i}}.png (i=1..{num_variants}).
- Each output must be a transformation of the ORIGINAL IMAGE, not a newly generated image
- Return a compact JSON log array (one object per variant) with fields:
  id, strategy ('horizontal_flip'|'reposition'), target_relation{{entities,before,after}}, ops (e.g., translate_px, rotate_deg, contrast, brightness, noise_sigma, blur_sigma), seed, validation.target_relation_flipped (boolean).
"""

# =============================================================================
# IMAGE ANALYSIS PROMPT
# =============================================================================

IMAGE_ANALYSIS_PROMPT = """Given this image and the spatial reasoning question: "{question}"

TASK: Generate perturbed image variants following these rules:

1. ANALYZE the spatial relationships in the image (identify objects and their current relations)
2. DETERMINE the optimal transformation strategy to test the spatial relationship:
   - For 'left/right' relations: recommend horizontal_flip
   - For 'above/below' relations: recommend vertical_flip
   - For both: recommend rotate_180
3. GENERATE perturbed variants:
   - FES variants: Apply subtle noise, blur, or contrast changes that preserve spatial relations
   - FCS variants: Apply transformations that flip the spatial relation (flips/rotations)

4. RETURN a JSON response with:
{{
  'objects': [list of detected objects],
  'current_relation': 'description of spatial relation',
  'recommended_transform': 'horizontal_flip|vertical_flip|rotate_180',
  'feasibility': 'high|medium|low',
  'fes_perturbations': [
    {{'type': 'noise', 'params': {{'sigma': 0.01}}}},
    {{'type': 'blur', 'params': {{'sigma': 0.5}}}},
    {{'type': 'contrast', 'params': {{'factor': 1.05}}}}
  ],
  'fcs_perturbations': [
    {{'type': 'horizontal_flip', 'inverts': ['left', 'right']}},
    {{'type': 'vertical_flip', 'inverts': ['above', 'below']}}
  ],
  'notes': 'explanation of analysis and recommended perturbations'
}}

Focus on generating ACTIONABLE perturbation instructions that can be applied to create test variants.
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_fes_image_prompt(question: str, num_variants: int = 14, base_name: str = "sample") -> tuple:
    """Get FES image generation prompt.

    Args:
        question: The question about the image
        num_variants: Number of variants to generate
        base_name: Base name for output files

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system = FES_IMAGE_SYSTEM_PROMPT.format(
        global_constraints=GLOBAL_IMAGE_CONSTRAINTS,
        num_variants=num_variants
    )
    user = FES_IMAGE_USER_PROMPT_TEMPLATE.format(
        question=question,
        num_variants=num_variants,
        base_name=base_name
    )
    return system, user


def get_fcs_image_prompt(question: str, relation_to_invert: str = "left-of",
                         num_variants: int = 4, base_name: str = "sample",
                         image_ref: str = "<ORIGINAL_IMAGE_PROVIDED>",
                         objects_or_bboxes: str = "[]") -> str:
    """Get FCS image generation prompt using enhanced prompt.

    Args:
        question: The question about the image
        relation_to_invert: The spatial relation to flip
        num_variants: Number of variants to generate
        base_name: Base name for output files
        image_ref: Reference to the original image
        objects_or_bboxes: Object information (optional)

    Returns:
        Complete FCS enhanced prompt string
    """
    return FCS_IMAGE_ENHANCED_PROMPT.format(
        image_ref=image_ref,
        original_question=question,
        target_relation=relation_to_invert,
        objects_or_bboxes=objects_or_bboxes,
        num_variants=num_variants,
        base_name=base_name
    )


def get_fcs_image_enhanced_prompt(original_question: str, target_relation: str,
                                  num_variants: int = 4, base_name: str = "sample",
                                  image_ref: str = "<ATTACHED>",
                                  objects_or_bboxes: str = "[]") -> str:
    """Get enhanced FCS image generation prompt.

    Args:
        original_question: The original question
        target_relation: The relation to invert
        num_variants: Number of variants to generate
        base_name: Base name for output files
        image_ref: Reference to the image
        objects_or_bboxes: Object information

    Returns:
        Complete enhanced prompt string
    """
    return FCS_IMAGE_ENHANCED_PROMPT.format(
        image_ref=image_ref,
        original_question=original_question,
        target_relation=target_relation,
        objects_or_bboxes=objects_or_bboxes,
        num_variants=num_variants,
        base_name=base_name
    )


def get_image_analysis_prompt(question: str) -> str:
    """Get image analysis prompt.

    Args:
        question: The question about the image

    Returns:
        Analysis prompt string
    """
    return IMAGE_ANALYSIS_PROMPT.format(question=question)


if __name__ == '__main__':
    # Test prompt generation
    test_question = "Is the car to the left of the cat?"

    print("=" * 80)
    print("FES IMAGE PROMPT TEST")
    print("=" * 80)
    system, user = get_fes_image_prompt(test_question, num_variants=14, base_name="test_sample")
    print(f"SYSTEM:\n{system}\n")
    print(f"USER:\n{user}\n")

    print("\n" + "=" * 80)
    print("FCS IMAGE ENHANCED PROMPT TEST")
    print("=" * 80)
    fcs_prompt = get_fcs_image_prompt(test_question, relation_to_invert="left-of",
                                      num_variants=4, base_name="test_sample")
    print(f"FCS ENHANCED PROMPT:\n{fcs_prompt}\n")

    print("\n" + "=" * 80)
    print("FCS IMAGE ENHANCED PROMPT (Using helper function)")
    print("=" * 80)
    fcs_enhanced = get_fcs_image_enhanced_prompt(
        original_question=test_question,
        target_relation="left-of",
        num_variants=4,
        base_name="test_sample"
    )
    print(f"FCS ENHANCED:\n{fcs_enhanced}\n")

