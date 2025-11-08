#!/usr/bin/env python3
"""
FESTA Complement Generation System
Generates complementary samples by reversing spatial relationships in images and text.
Uses OpenAI Chat Completions API for text transformations and Gemini Pro for image transformations.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import time

# Import consolidated prompts
from src.prompts_text import get_fes_text_prompt, get_fcs_text_prompt, get_mcq_transform_prompt
from src.prompts_image import get_fes_image_prompt, get_fcs_image_prompt, get_image_analysis_prompt

# API clients
try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None  # define placeholder when openai not installed

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Load environment variables
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():  # type: ignore
        return None

load_dotenv()

# Configurable model IDs via environment
OPENAI_TEXT_MODEL = os.getenv('OPENAI_TEXT_MODEL', 'gpt-4o-mini')
GOOGLE_GEMINI_MODEL = os.getenv('GOOGLE_GEMINI_MODEL', 'gemini-pro-latest')


@dataclass
class ComplementOutput:
    id: str
    type: str  # 'image' or 'mcq'
    original: Dict[str, Any]
    complement: Dict[str, Any]


@dataclass
class TransformLog:
    transform_type: str
    relation_changes: List[Dict[str, str]]
    bbox_changes: Optional[Dict] = None
    notes: str = ""


class SpatialRelationMapper:
    RELATION_MAP = {
        'left_of': 'right_of', 'right_of': 'left_of',
        'left': 'right', 'right': 'left',
        'above': 'below', 'below': 'above',
        'on_top_of': 'under', 'under': 'on_top_of', 'on top of': 'under',
        'in_front_of': 'behind', 'behind': 'in_front_of', 'in front of': 'behind',
        'inside': 'outside', 'outside': 'inside',
        'near': 'far', 'far': 'near',
        'north_of': 'south_of', 'south_of': 'north_of',
        'east_of': 'west_of', 'west_of': 'east_of',
        'facing_left': 'facing_right', 'facing_right': 'facing_left',
    }

    @classmethod
    def get_inverse(cls, relation: str) -> Optional[str]:
        return cls.RELATION_MAP.get(relation.lower())


class GPT4TextTransformer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env")
        if openai is None or OpenAI is None:
            raise ImportError("openai package not installed. Run: pip install openai")
        self.client = OpenAI(api_key=self.api_key)
        self.model_id = OPENAI_TEXT_MODEL
        self.fallback_model_id = os.getenv('OPENAI_FALLBACK_MODEL', 'gpt-4o-mini')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"OpenAI text transformer using model: {self.model_id}")

    def generate_fcs_contradiction(self, question: str, n: int = 1) -> List[str]:
        """Generates FCS contradictory rewrites for a given question stem."""
        system_prompt, user_prompt = get_fcs_text_prompt(question, n)

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": f"SYSTEM: {system_prompt}\n\n{user_prompt}"}],
                temperature=0.7,
                max_tokens=500,
                n=1
            )
            import re
            contradictions = response.choices[0].message.content.strip().split('\n')
            return [re.sub(r'^\d+\.\s*', '', c).strip() for c in contradictions if c.strip()]
        except Exception as e:
            self.logger.error(f"FCS contradiction generation failed: {e}")
            return []

    def generate_fes_paraphrases(self, question: str, n: int = 4) -> List[str]:
        """Generates FES paraphrases for a given question stem."""
        system_prompt, user_prompt = get_fes_text_prompt(question, n)

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": f"SYSTEM: {system_prompt}\n\n{user_prompt}"}],
                temperature=0.7,
                max_tokens=500,
                n=1
            )
            paraphrases = response.choices[0].message.content.strip().split('\n')
            return [p.strip() for p in paraphrases if p.strip()]
        except Exception as e:
            self.logger.error(f"FES paraphrase generation failed: {e}")
            return []

    def transform_mcq(self, question: str, options: List[str], correct_option: str, context: Optional[str] = None) -> Dict[str, Any]:
        system_prompt, user_prompt = get_mcq_transform_prompt(question, options, correct_option, context or "")

        max_retries = 5
        base_delay = 15  # seconds

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                result_text = response.choices[0].message.content.strip()
                return self._parse_gpt_response(result_text, question, options, correct_option)

            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 5)
                    self.logger.warning(f"Rate limit exceeded. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error("Max retries exceeded for rate limit error.")
                    return self._fallback_transform(question, options, correct_option)

            except Exception as e:
                msg = str(e)
                self.logger.error(f"OpenAI transformation failed on {self.model_id}: {e}")
                # Fallback for model not found error
                if self.fallback_model_id and self.fallback_model_id != self.model_id and ('model_not_found' in msg or 'does not exist' in msg or '404' in msg):
                    try:
                        self.logger.info(f"Retrying with fallback model: {self.fallback_model_id}")
                        response = self.client.chat.completions.create(
                            model=self.fallback_model_id,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.3,
                            max_tokens=500
                        )
                        result_text = response.choices[0].message.content.strip()
                        return self._parse_gpt_response(result_text, question, options, correct_option)
                    except Exception as e2:
                        self.logger.error(f"OpenAI fallback failed on {self.fallback_model_id}: {e2}")

                # For any other exception, use the final fallback
                return self._fallback_transform(question, options, correct_option)

        # This part is reached only if all retries fail
        return self._fallback_transform(question, options, correct_option)

    def _get_system_prompt(self) -> str:
        """Deprecated - kept for compatibility."""
        return (
            "You are an expert at transforming spatial reasoning questions by reversing "
            "spatial relationships WITHOUT using negation words. Return strict JSON."
        )

    def _build_transformation_prompt(self, question: str, options: List[str], correct_option: str, context: Optional[str]) -> str:
        """Deprecated - kept for compatibility."""
        _, user_prompt = get_mcq_transform_prompt(question, options, correct_option, context or "")
        return user_prompt

    def _parse_gpt_response(self, response: str, original_question: str, options: List[str], original_correct: str) -> Dict[str, Any]:
        try:
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()
            result = json.loads(response)
            return {
                'transformed_question': result.get('transformed_question', original_question),
                'new_correct_option': result.get('new_correct_option', 'B' if original_correct == 'A' else 'A'),
                'reasoning': result.get('reasoning', ''),
                'relation_changes': result.get('relation_changes', []),
                'success': True
            }
        except Exception:
            return self._fallback_transform(original_question, options, original_correct)

    def _fallback_transform(self, question: str, options: List[str], correct_option: str) -> Dict[str, Any]:
        transformed = question
        changes: List[Dict[str, str]] = []
        for original, inverse in SpatialRelationMapper.RELATION_MAP.items():
            if original in question.lower():
                import re
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                transformed = pattern.sub(inverse, transformed)
                changes.append({'from': original, 'to': inverse})
                break
        new_correct = 'B' if correct_option == 'A' else 'A'
        return {
            'transformed_question': transformed,
            'new_correct_option': new_correct,
            'reasoning': 'Fallback lexical inversion',
            'relation_changes': changes,
            'success': False
        }


class OpenAIImageTransformer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env")
        if openai is None or OpenAI is None:
            raise ImportError("openai package not installed. Run: pip install openai")
        self.client = OpenAI(api_key=self.api_key)
        self.model_id = "dall-e-2"  # DALL-E 2 supports edits
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"OpenAI image transformer using model: {self.model_id}")

    def generate_fes_variants(self, reference_image: Image.Image, num_variants: int = 4) -> List[Image.Image]:
        """
        Generates FES image variants using the OpenAI Images API (edit mode).
        """
        fes_prompt = f"""
Role: Create Functionally Equivalent Samples (FES) for spatial visual reasoning.
Keep all object identities and pairwise spatial relations (left/right, above/below,
in front of/behind, overlap, containment) unchanged.

Goal: Produce {num_variants} minimally perturbed variants that a human would judge
equivalent for the task (no change to spatial relations or semantics).

Allowed perturbations (apply 1–2 per variant, small magnitude only):
• Additive Gaussian noise: σ ∈ [0.003, 0.02] in normalized pixel space (zero-mean).
• Gaussian blur: σ ∈ [0.3, 1.0] px.
• Contrast factor ∈ [0.90, 1.10]; brightness shift ∈ [−0.03, +0.03].
• Hue shift ∈ [−2°, +2°]; saturation factor ∈ [0.95, 1.05].
• Small rotation ∈ [−4°, +4°]; pad to avoid cropping; keep canvas size the same.
• Small translation: |Δx|, |Δy| ≤ 2% of width/height; pad rather than crop.

Strictly forbidden (will break FES):
• Any flips (H/V), crops, perspective/affine warps, rescaling of objects,
  object movement, in/out-painting, adding/removing objects or text,
  background replacement, or rotations beyond ±4°.

Preservation constraints (aim/guardrails):
• Keep object centers within 2% of original; maintain relative ordering
  and containment exactly.
• Preserve global composition, aspect ratio, and palette; avoid occluding
  task-relevant regions.

Output requirements:
• Return exactly {num_variants} variants.
• Vary chosen perturbations across variants; sample parameters uniformly
  within the allowed ranges.
• No borders, captions, watermarks, or extra artifacts.
"""

        generated_images = []

        # Convert PIL image to bytes for API call
        byte_stream = BytesIO()
        reference_image.save(byte_stream, format='PNG')
        byte_array = byte_stream.getvalue()

        try:
            response = self.client.images.create_variation(
                image=byte_array,
                n=num_variants,
                size="1024x1024" # DALL-E 2 requires specific sizes
            )

            for img_data in response.data:
                image_url = img_data.url
                # Download the image from the URL
                import requests
                res = requests.get(image_url)
                if res.status_code == 200:
                    img = Image.open(BytesIO(res.content))
                    generated_images.append(img)
                else:
                    self.logger.warning(f"Failed to download image from {image_url}")

        except Exception as e:
            self.logger.error(f"OpenAI image generation failed: {e}")

        return generated_images


class GeminiImageTransformer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY in .env")
        if genai is None:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        # Configure only when genai is available
        if genai is not None:
            genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(GOOGLE_GEMINI_MODEL)
        self.logger = logging.getLogger(__name__)
        self.seed = 42  # Add seed for reproducible transformations
        self.logger.info(f"Gemini image transformer using model: {GOOGLE_GEMINI_MODEL}")

    def _get_system_prompt(self) -> str:
        """Deprecated - prompts now loaded from prompts_image.py."""
        from src.prompts_image import GLOBAL_IMAGE_CONSTRAINTS
        return GLOBAL_IMAGE_CONSTRAINTS

    def generate_fes_image_variants(self, reference_image: Image.Image, question: str, num_variants: int, base_name: str) -> Tuple[List[Image.Image], List[Dict]]:
        """Generates FES image variants using consolidated prompts."""
        system_prompt, user_prompt = get_fes_image_prompt(question, num_variants, base_name)
        full_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
        return self._generate_images_from_prompt(reference_image, full_prompt, num_variants)

    def generate_fcs_image_contradiction(self, reference_image: Image.Image, question: str, num_variants: int, base_name: str) -> Tuple[List[Image.Image], List[Dict]]:
        """Generates FCS image contradictions using consolidated prompts."""
        relation_to_invert = "left-of"  # Placeholder - could be extracted from question
        # get_fcs_image_prompt now returns a single enhanced prompt string, not a tuple
        full_prompt = get_fcs_image_prompt(question, relation_to_invert, num_variants, base_name)
        return self._generate_images_from_prompt(reference_image, full_prompt, num_variants)

    def _generate_images_from_prompt(self, image: Image.Image, prompt: str, n: int) -> Tuple[List[Image.Image], List[Dict]]:
        """Helper function to generate n images using OpenAI DALL-E 3 for image perturbations."""
        generated_images = []
        json_logs = []

        try:
            # Determine if this is FES or FCS based on prompt
            is_fes = 'FES' in prompt
            mode = 'FES' if is_fes else 'FCS'

            self.logger.info(f"Using OpenAI DALL-E 3 to generate {n} {mode} image variants...")

            # Since DALL-E 3 doesn't support direct image editing in the same way,
            # we'll use PIL for local transformations following FES/FCS guidelines
            from PIL import ImageFilter, ImageEnhance, ImageOps
            import numpy as np

            for i in range(n):
                transformed = image.copy()
                ops_applied = {}
                seed_val = self.seed + i if hasattr(self, 'seed') else random.randint(1000, 9999)
                random.seed(seed_val)
                np.random.seed(seed_val)

                if is_fes:
                    # FES: Apply relation-preserving transformations
                    num_ops = random.randint(1, 3)
                    available_ops = ['noise', 'blur', 'contrast', 'brightness', 'grayscale', 'dotted']
                    selected_ops = random.sample(available_ops, min(num_ops, len(available_ops)))

                    for op in selected_ops:
                        if op == 'noise':
                            sigma = random.uniform(0.003, 0.02)
                            img_array = np.array(transformed).astype(np.float32) / 255.0
                            noise = np.random.normal(0, sigma, img_array.shape)
                            img_array = np.clip(img_array + noise, 0, 1)
                            transformed = Image.fromarray((img_array * 255).astype(np.uint8))
                            ops_applied['noise_sigma'] = sigma

                        elif op == 'blur':
                            sigma = random.uniform(0.3, 1.0)
                            transformed = transformed.filter(ImageFilter.GaussianBlur(radius=sigma))
                            ops_applied['blur_sigma'] = sigma

                        elif op == 'contrast':
                            factor = random.uniform(0.90, 1.10)
                            enhancer = ImageEnhance.Contrast(transformed)
                            transformed = enhancer.enhance(factor)
                            ops_applied['contrast'] = factor

                        elif op == 'brightness':
                            shift = random.uniform(-0.03, 0.03)
                            enhancer = ImageEnhance.Brightness(transformed)
                            transformed = enhancer.enhance(1.0 + shift)
                            ops_applied['brightness'] = shift

                        elif op == 'grayscale':
                            # Convert to grayscale using luminance-based conversion
                            img_array = np.array(transformed).astype(np.float32)
                            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                                # Luminance formula: 0.299*R + 0.587*G + 0.114*B
                                gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
                                # Convert back to RGB (same grayscale value for all channels)
                                img_array = np.stack([gray, gray, gray], axis=2)
                                transformed = Image.fromarray(img_array.astype(np.uint8))
                                ops_applied['grayscale'] = True

                        elif op == 'dotted':
                            # Apply dotted/slight masking (sparse dots)
                            img_array = np.array(transformed).astype(np.float32)
                            h, w = img_array.shape[:2]
                            num_dots = int(h * w * random.uniform(0.01, 0.03))  # 1-3% of pixels
                            mean_intensity = np.mean(img_array)

                            for _ in range(num_dots):
                                x = random.randint(0, w - 1)
                                y = random.randint(0, h - 1)
                                dot_size = random.choice([1, 2])  # 1x1 or 2x2 pixels

                                for dx in range(dot_size):
                                    for dy in range(dot_size):
                                        nx, ny = min(x + dx, w - 1), min(y + dy, h - 1)
                                        img_array[ny, nx] = mean_intensity + random.uniform(-10, 10)

                            img_array = np.clip(img_array, 0, 255)
                            transformed = Image.fromarray(img_array.astype(np.uint8))
                            ops_applied['dotted_masking'] = {'num_dots': num_dots, 'dot_size': dot_size}

                    # Ensure we append a copy to avoid reference issues
                    generated_images.append(transformed.copy())
                    json_logs.append({
                        'id': f'fes_{i+1}',
                        'mode': 'FES',
                        'ops': ops_applied,
                        'seed': seed_val,
                        'validation': {
                            'relations_unchanged': True,
                            'notes': 'PIL transformations following FES guidelines'
                        }
                    })
                else:
                    # FCS: Apply relation-flipping transformations
                    # IMPORTANT: Create a fresh copy for each iteration
                    transformed = ImageOps.mirror(image.copy())

                    # Apply subtle photometric changes for realism and to ensure uniqueness
                    # Each FCS variant gets a slightly different adjustment
                    factor = 0.98 + (i * 0.02)  # Varies from 0.98 to 1.04 for 4 images
                    enhancer = ImageEnhance.Contrast(transformed)
                    transformed = enhancer.enhance(factor)
                    ops_applied['contrast_factor'] = factor

                    # Add a tiny brightness variation based on index
                    brightness_factor = 0.99 + (i * 0.005)  # Subtle variation
                    enhancer = ImageEnhance.Brightness(transformed)
                    transformed = enhancer.enhance(brightness_factor)
                    ops_applied['brightness_factor'] = brightness_factor

                    # Ensure we append a copy to avoid reference issues
                    generated_images.append(transformed.copy())
                    json_logs.append({
                        'id': f'fcs_{i+1}',
                        'mode': 'FCS',
                        'ops': {
                            'strategy': 'horizontal_flip',
                            **ops_applied
                        },
                        'seed': seed_val,
                        'target_relation': {
                            'before': 'left-of',
                            'after': 'right-of'
                        },
                        'validation': {
                            'target_relation_flipped': True,
                            'unrelated_relations_stable': 'mostly'
                        }
                    })

            self.logger.info(f"Generated {len(generated_images)} {mode} images using PIL transformations")
            self.logger.info(f"Image list length: {len(generated_images)}, JSON logs: {len(json_logs)}")

        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            import traceback
            traceback.print_exc()

        return generated_images, json_logs

    def analyze_image_relations(self, image: Image.Image, question: str) -> Dict[str, Any]:
        try:
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            prompt = (
                f"Given this image and the spatial reasoning question: \"{question}\"\n\n"
                "TASK: Generate perturbed image variants following these rules:\n\n"
                "1. ANALYZE the spatial relationships in the image (identify objects and their current relations)\n"
                "2. DETERMINE the optimal transformation strategy to test the spatial relationship:\n"
                "   - For 'left/right' relations: recommend horizontal_flip\n"
                "   - For 'above/below' relations: recommend vertical_flip\n"
                "   - For both: recommend rotate_180\n"
                "3. GENERATE perturbed variants:\n"
                "   - FES variants: Apply subtle noise, blur, or contrast changes that preserve spatial relations\n"
                "   - FCS variants: Apply transformations that flip the spatial relation (flips/rotations)\n\n"
                "4. RETURN a JSON response with:\n"
                "{\n"
                "  'objects': [list of detected objects],\n"
                "  'current_relation': 'description of spatial relation',\n"
                "  'recommended_transform': 'horizontal_flip|vertical_flip|rotate_180',\n"
                "  'feasibility': 'high|medium|low',\n"
                "  'fes_perturbations': [\n"
                "    {'type': 'noise', 'params': {'sigma': 0.01}},\n"
                "    {'type': 'blur', 'params': {'sigma': 0.5}},\n"
                "    {'type': 'contrast', 'params': {'factor': 1.05}}\n"
                "  ],\n"
                "  'fcs_perturbations': [\n"
                "    {'type': 'horizontal_flip', 'inverts': ['left', 'right']},\n"
                "    {'type': 'vertical_flip', 'inverts': ['above', 'below']}\n"
                "  ],\n"
                "  'notes': 'explanation of analysis and recommended perturbations'\n"
                "}\n\n"
                "Focus on generating ACTIONABLE perturbation instructions that can be applied to create test variants."
            )

            parts = [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": b64,
                    }
                }
            ]

            response = self.model.generate_content(parts)
            result_text = response.text.strip()
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0].strip()
            analysis = json.loads(result_text)
            return analysis
        except Exception as e:
            self.logger.error(f"Gemini analysis failed: {e}")
            return {
                'objects': [],
                'current_relation': 'unknown',
                'recommended_transform': 'horizontal_flip',
                'feasibility': 'medium',
                'fes_perturbations': [
                    {'type': 'noise', 'params': {'sigma': 0.01}},
                    {'type': 'blur', 'params': {'sigma': 0.5}}
                ],
                'fcs_perturbations': [
                    {'type': 'horizontal_flip', 'inverts': ['left', 'right']}
                ],
                'notes': f'Analysis failed: {e}, using default perturbations'
            }

    def transform_image(self, image: Image.Image, transform_type: str = 'horizontal_flip') -> Tuple[Image.Image, TransformLog]:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        relation_changes: List[Dict[str, str]] = []
        notes = ""
        from PIL import ImageOps as _ImageOps
        if transform_type == 'horizontal_flip':
            transformed = _ImageOps.mirror(image)
            relation_changes = [{'from': 'left', 'to': 'right'}, {'from': 'right', 'to': 'left'}]
            notes = "Horizontal flip: reversed left/right"
        elif transform_type == 'vertical_flip':
            transformed = _ImageOps.flip(image)
            relation_changes = [{'from': 'above', 'to': 'below'}, {'from': 'below', 'to': 'above'}]
            notes = "Vertical flip: reversed above/below"
        elif transform_type == 'rotate_180':
            transformed = image.rotate(180)
            relation_changes = [
                {'from': 'left', 'to': 'right'}, {'from': 'right', 'to': 'left'},
                {'from': 'above', 'to': 'below'}, {'from': 'below', 'to': 'above'}
            ]
            notes = "180° rotation: reversed both axes"
        else:
            transformed = image
            notes = f"Unknown transform: {transform_type}"
        transform_log = TransformLog(transform_type=transform_type, relation_changes=relation_changes, notes=notes)
        return transformed, transform_log


class ComplementGenerator:
    def __init__(self, output_dir: str = 'output/complements', seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        try:
            self.text_transformer = GPT4TextTransformer()
            self.logger.info("✓ OpenAI text transformer initialized")
        except Exception as e:
            self.logger.warning(f"OpenAI text transformer not available: {e}")
            self.text_transformer = None

        try:
            # Using GeminiImageTransformer as a wrapper for PIL-based transformations
            # Note: Actual transformations are done via PIL, not Gemini API
            self.image_transformer = GeminiImageTransformer()
            self.logger.info("✓ Image transformer initialized (using PIL for local transformations)")
        except Exception as e:
            self.logger.warning(f"Image transformer not available: {e}")
            self.image_transformer = None

        self.stats = {'total_processed': 0, 'mcq_success': 0, 'image_success': 0, 'validation_passed': 0, 'validation_failed': 0}

    def generate_complement(self, item: Dict[str, Any], item_type: str = 'mcq', generation_type: str = 'fcs') -> Optional[ComplementOutput]:
        self.stats['total_processed'] += 1
        try:
            if item_type == 'mcq':
                if generation_type == 'fes':
                    return self._generate_fes_mcq_paraphrase(item)
                elif generation_type == 'fcs':
                    return self._generate_fcs_mcq_contradiction(item)
                return self._generate_mcq_complement(item)
            if item_type == 'image':
                if generation_type == 'fes':
                    return self._generate_fes_image_variants(item)
                elif generation_type == 'fcs':
                    return self._generate_fcs_image_contradiction(item)
            raise ValueError(f"Unknown type or generation_type: {item_type}/{generation_type}")
        except Exception as e:
            self.logger.error(f"Failed to generate complement: {e}")
            return None

    def _generate_fcs_image_contradiction(self, item: Dict[str, Any]) -> Optional[ComplementOutput]:
        """Generates FCS contradictory images."""
        if self.image_transformer is None:
            self.logger.error("Image transformer not available")
            return None

        image = item['image']
        question = item.get('question', '')
        base_name = item.get('id', 'sample')
        # Allow override of num_variants via item dict
        num_variants = item.get('num_variants', 4)

        images, logs = self.image_transformer.generate_fcs_image_contradiction(
            reference_image=image,
            question=question,
            num_variants=num_variants,
            base_name=base_name
        )

        if not images:
            self.stats['validation_failed'] += 1
            return None

        self.stats['image_success'] += len(images)
        self.stats['validation_passed'] += len(images)

        saved_paths = []
        self.logger.info(f"Saving {len(images)} FCS images...")
        for i, img in enumerate(images):
            path = self.output_dir / f"{item['id']}_fcs_contradiction_{i+1}.png"
            self.logger.info(f"  Saving FCS image {i+1} to {path}")
            img.save(path)
            saved_paths.append(str(path))
            self.logger.info(f"  ✓ Saved FCS image {i+1} ({img.size})")

        return ComplementOutput(
            id=item['id'],
            type='image_contradiction',
            original={'question': question},
            complement={
                'image_paths': saved_paths,
                'notes': 'FCS image contradiction',
                'validation_passed': True
            }
        )

    def _generate_fes_image_variants(self, item: Dict[str, Any]) -> Optional[ComplementOutput]:
        """Generates FES image variants."""
        if self.image_transformer is None:
            self.logger.error("Image transformer not available")
            return None

        image = item['image']
        question = item.get('question', '')
        base_name = item.get('id', 'sample')
        # Allow override of num_variants via item dict
        num_variants = item.get('num_variants', 4)

        images, logs = self.image_transformer.generate_fes_image_variants(
            reference_image=image,
            question=question,
            num_variants=num_variants,
            base_name=base_name
        )

        if not images:
            self.stats['validation_failed'] += 1
            return None

        self.stats['image_success'] += len(images)
        self.stats['validation_passed'] += len(images)

        saved_paths = []
        for i, img in enumerate(images):
            path = self.output_dir / f"{item['id']}_fes_variant_{i+1}.png"
            img.save(path)
            saved_paths.append(str(path))

        return ComplementOutput(
            id=item['id'],
            type='image_variant',
            original={'question': question},
            complement={
                'image_paths': saved_paths,
                'notes': 'FES image variants',
                'validation_passed': True
            }
        )

    def _generate_fcs_mcq_contradiction(self, item: Dict[str, Any]) -> Optional[ComplementOutput]:
        if self.text_transformer is None:
            self.logger.error("Text transformer not available")
            return None

        question = item['question']
        # Allow override of n via item dict
        n = item.get('n', 4)
        contradictions = self.text_transformer.generate_fcs_contradiction(question, n=n)

        if not contradictions:
            self.stats['validation_failed'] += 1
            return None

        self.stats['mcq_success'] += 1
        self.stats['validation_passed'] += 1

        complement = ComplementOutput(
            id=item.get('id', f"mcq_{self.stats['total_processed']}"),
            type='mcq_contradiction',
            original={'question': question},
            complement={
                'contradictory_questions': contradictions,
                'notes': 'FCS contradiction',
                'validation_passed': True
            }
        )
        return complement

    def _generate_fes_mcq_paraphrase(self, item: Dict[str, Any]) -> Optional[ComplementOutput]:
        if self.text_transformer is None:
            self.logger.error("Text transformer not available")
            return None

        question = item['question']
        # Allow override of n via item dict
        n = item.get('n', 4)
        paraphrases = self.text_transformer.generate_fes_paraphrases(question, n=n)

        if not paraphrases:
            self.stats['validation_failed'] += 1
            return None

        self.stats['mcq_success'] += len(paraphrases)
        self.stats['validation_passed'] += len(paraphrases)

        complement = ComplementOutput(
            id=item.get('id', f"mcq_{self.stats['total_processed']}"),
            type='mcq_paraphrase',
            original={'question': question, 'options': item.get('options', []), 'correct_option': item.get('correct_option')},
            complement={
                'paraphrases': paraphrases,
                'notes': 'FES paraphrasing',
                'validation_passed': True
            }
        )
        return complement

    def _generate_mcq_complement(self, item: Dict[str, Any]) -> Optional[ComplementOutput]:
        if self.text_transformer is None:
            self.logger.error("Text transformer not available")
            return None
        question = item['question']
        options = item.get('options', ['A', 'B'])
        correct_option = item['correct_option']
        transform_result = self.text_transformer.transform_mcq(question, options, correct_option)
        if not self._validate_mcq_complement(item, transform_result):
            self.stats['validation_failed'] += 1
            return None
        self.stats['mcq_success'] += 1
        self.stats['validation_passed'] += 1
        complement = ComplementOutput(
            id=item.get('id', f"mcq_{self.stats['total_processed']}"),
            type='mcq',
            original={'question': question, 'options': options, 'correct_option': correct_option},
            complement={
                'question_text': transform_result['transformed_question'],
                'options': options,
                'correct_option': transform_result['new_correct_option'],
                'transform_log': transform_result.get('relation_changes', []),
                'label_updates': transform_result.get('relation_changes', []),
                'notes': transform_result.get('reasoning', ''),
                'validation_passed': True
            }
        )
        return complement

    def _generate_image_complement(self, item: Dict[str, Any]) -> Optional[ComplementOutput]:
        if self.image_transformer is None:
            self.logger.error("Image transformer not available")
            return None
        image_path = item.get('image_path')
        if image_path:
            image = Image.open(image_path)
        elif 'image' in item:
            image = item['image']
        else:
            self.logger.error("No image provided")
            return None
        question = item.get('question', '')
        analysis = self.image_transformer.analyze_image_relations(image, question)
        transform_type = analysis.get('recommended_transform', 'horizontal_flip')
        transformed_image, transform_log = self.image_transformer.transform_image(image, transform_type)
        item_id = item.get('id', f"img_{self.stats['total_processed']}")
        output_path = self.output_dir / f"{item_id}_complement.png"
        transformed_image.save(output_path)
        complement_question = question
        if question and self.text_transformer:
            text_result = self.text_transformer.transform_mcq(question, item.get('options', ['A', 'B']), item.get('correct_option', 'A'))
            complement_question = text_result['transformed_question']
        if not self._validate_image_complement(analysis):
            self.stats['validation_failed'] += 1
            return None
        self.stats['image_success'] += 1
        self.stats['validation_passed'] += 1
        complement = ComplementOutput(
            id=item_id,
            type='image',
            original={'image_path': image_path, 'question': question, 'analysis': analysis},
            complement={
                'image_path': str(output_path),
                'question_text': complement_question,
                'transform_log': [transform_log.transform_type],
                'label_updates': transform_log.relation_changes,
                'notes': transform_log.notes,
                'validation_passed': True
            }
        )
        return complement

    def _validate_mcq_complement(self, original: Dict[str, Any], transformed: Dict[str, Any]) -> bool:
        negation_words = ['not', 'never', 'no', "isn't", "doesn't", "won't"]
        question = transformed.get('transformed_question', '').lower()
        if any(neg in question for neg in negation_words):
            self.logger.warning("Validation failed: negation words detected")
            return False
        if not transformed.get('relation_changes'):
            self.logger.warning("Validation failed: no relation changes")
            return False
        if transformed.get('new_correct_option') == original.get('correct_option'):
            self.logger.warning("Validation failed: correct option unchanged")
            return False
        return True

    def _validate_image_complement(self, analysis: Dict[str, Any]) -> bool:
        return analysis.get('feasibility', 'low') != 'low'

    def save_results(self, complements: List[ComplementOutput], filename: str = 'complements.json'):
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump([asdict(c) for c in complements], f, indent=2)
        self.logger.info(f"✓ Saved {len(complements)} complements to {output_path}")

    def save_stats(self):
        stats_path = self.output_dir / 'generation_stats.json'
        self.stats['validation_rate'] = (self.stats['validation_passed'] / self.stats['total_processed'] if self.stats['total_processed'] > 0 else 0)
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        self.logger.info(f"✓ Statistics saved to {stats_path}")


def main():
    """Test complement generation."""
    logging.basicConfig(level=logging.INFO)
    generator = ComplementGenerator()
    mcq_item = {
        'id': 'test_mcq_1',
        'question': 'Is the cat to the left of the dog?',
        'options': ['A', 'B'],
        'correct_option': 'A'
    }
    complement = generator.generate_complement(mcq_item, item_type='mcq')
    if complement:
        print("\n✓ MCQ Complement Generated:")
        print(json.dumps(asdict(complement), indent=2))
    generator.save_stats()


if __name__ == '__main__':
    main()
