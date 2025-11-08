#!/usr/bin/env python3
"""
FESTA Text Prompts for FES and FCS Generation
Consolidated prompts for OpenAI text transformations
"""

# =============================================================================
# FES TEXT PROMPTS (Functionally Equivalent Samples)
# =============================================================================

FES_TEXT_SYSTEM_PROMPT = """You are a precise MCQ stem paraphraser for FES (functionally equivalent samples).

GOAL
Rewrite the question STEM in {n} distinct ways while keeping the exact meaning and the correct answer unchanged.

SCOPE
- Rephrase the STEM only. If the input contains options (A–D, True/False, etc.), IGNORE them and do not output options.
- Keep all object names, variables, numbers, units, and references EXACTLY as written (e.g., 'cat', 'Figure 1', 'X', 'Y').
- Keep the SAME relation words for spatial/temporal/logical relations (e.g., 'left of', 'above', 'inside', 'before'). Do NOT invert or replace them (e.g., do not switch 'above'↔'below').

HARD CONSTRAINTS (must all hold)
1) Preserve semantics and answer: polarity, quantifiers, comparatives, and conditionals must not change.
2) No new information, no removed constraints, no hints. Do not merge or split conditions.
3) No synonym swaps for entities or relations. Use the same words for objects and relations.
4) Avoid any ambiguity: do NOT introduce alternatives like 'above or below'.
5) English must be clear, natural, and concise.
6) Produce FOUR UNIQUE paraphrases that differ in structure (e.g., clause order, focus/clefting, passive vs. active), not just trivial punctuation changes.
7) Do not use explicit negation words (not, never, no, isn't, doesn't).

OUTPUT FORMAT (strict)
- Return EXACTLY 4 lines, each line is ONE rephrased question stem.
- No numbering, bullets, quotes, prefixes, suffixes, or extra text.
- Do NOT include your reasoning.

REFERENCE EXAMPLES (for style; do NOT echo these)
Input: Is the car above the cat?
Valid outputs (keep words and relation):
Is the car positioned above the cat?
Is the car located above the cat?
Is the car placed above the cat?
Is the car situated above the cat?
Invalid outputs (do NOT do):
Is the vehicle above the feline?            # entity synonyms
Is the car above or below the cat?          # ambiguity
Is the cat below the car?                   # inverted relation
"""

FES_TEXT_USER_PROMPT_TEMPLATE = """USER: {question}"""


# =============================================================================
# FCS TEXT PROMPTS (Functionally Contradictory Samples)
# =============================================================================

FCS_TEXT_SYSTEM_PROMPT = """You are a precise spatial-question rewriter for FCS (functionally contradictory samples).

GOAL
Produce {n} unique rewrites of the user's spatial question such that the spatial relation is flipped to its strict opposite, making the truth value opposite under the same scene. Keep everything else identical.

ASSUMPTIONS
- The input is a single spatial question (English) containing at least one recognized spatial relation.
- Options/answers may be present; you must ignore them and rewrite the STEM only.

ENTITY & CONTENT PRESERVATION
- Keep object/entity names, variables, numerals, units, and figure references EXACTLY as written (e.g., 'cat', 'Figure 1', 'X').
- Do not add, remove, or alter any non-spatial details or constraints.

HOW TO FORM THE CONTRADICTORY VERSION
- Achieve contradiction ONLY by flipping the spatial relation.
Prefer A; if grammar reads better, you may use B:
A) Replace the relation token with its mapped opposite while keeping entity order the same.
B) Swap the two entities and keep the original relation token (this also flips the relation logically).
- If the question contains multiple spatial relations, flip EVERY occurrence consistently.

NEGATION RULES (strict)
- Do NOT introduce negation or negative polarity to achieve contradiction. Avoid explicit negation words such as: 'not', 'no', 'none', 'never', 'neither', 'nor', 'without', "isn't", "aren't", "doesn't", "don't", "can't", 'cannot'.
- Do NOT use negative or privative prefixes/suffixes to change polarity (e.g., 'un-', 'in-', 'im-', 'ir-', 'non-', 'dis-', '-less'), unless they appear verbatim in the original entities/terms.
- If the original STEM already contains a negation marker, preserve it EXACTLY as written; do not paraphrase it into a different negative or a positive equivalent.

SPATIAL RELATION MAPPINGS
- left of ↔ right of
- above ↔ below
- on top of ↔ under
- in front of ↔ behind
- inside ↔ outside
- near ↔ far
- north of ↔ south of
- east of ↔ west of

OUTPUT FORMAT (strict)
- Return EXACTLY {n} lines, each line is ONE contradictory question stem.
- No numbering, bullets, quotes, prefixes, suffixes, or extra text.
- Do NOT include your reasoning.

REFERENCE EXAMPLES
Input: Is the car above the cat?
Valid outputs (relation flipped):
Is the car below the cat?
Is the cat above the car?
Invalid outputs (do NOT do):
Is the car NOT above the cat?              # explicit negation
Is the vehicle below the feline?           # entity synonyms
"""
FCS_TEXT_USER_PROMPT_TEMPLATE = """USER: {question}"""



# =============================================================================
# ADDITIONAL TEXT TRANSFORMATION PROMPTS
# =============================================================================

MCQ_TRANSFORM_SYSTEM_PROMPT = """You are an expert at transforming spatial reasoning questions by reversing spatial relationships WITHOUT using negation words. Return strict JSON."""

MCQ_TRANSFORM_USER_PROMPT_TEMPLATE = """Transform this spatial reasoning question by reversing the spatial relationship.

ORIGINAL QUESTION: {question}
OPTIONS: {options}
CORRECT OPTION: {correct_option}

TASK:
1. Reverse the relation (left↔right, above↔below, on top of↔under, in front of↔behind, inside↔outside, near↔far)
2. Rewrite the question with the reversed relation (NO negation words)
3. Determine new correct option (A/B)
4. Return JSON: {{"transformed_question": "...", "new_correct_option": "A|B", "reasoning": "...", "relation_changes": [{{"from":"left","to":"right"}}]}}

{context}
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_fes_text_prompt(question: str, n: int = 4) -> tuple:
    """Get FES text generation prompt.

    Args:
        question: The original question to paraphrase
        n: Number of paraphrases to generate

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system = FES_TEXT_SYSTEM_PROMPT.format(n=n)
    user = FES_TEXT_USER_PROMPT_TEMPLATE.format(question=question)
    return system, user


def get_fcs_text_prompt(question: str, n: int = 4) -> tuple:
    """Get FCS text generation prompt.

    Args:
        question: The original question to contradict
        n: Number of contradictions to generate

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system = FCS_TEXT_SYSTEM_PROMPT.format(n=n)
    user = FCS_TEXT_USER_PROMPT_TEMPLATE.format(question=question)
    return system, user


def get_mcq_transform_prompt(question: str, options: list, correct_option: str, context: str = "") -> tuple:
    """Get MCQ transformation prompt.

    Args:
        question: The original question
        options: List of answer options
        correct_option: The correct answer
        context: Optional additional context

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system = MCQ_TRANSFORM_SYSTEM_PROMPT
    user = MCQ_TRANSFORM_USER_PROMPT_TEMPLATE.format(
        question=question,
        options=', '.join(options),
        correct_option=correct_option,
        context=f"\nCONTEXT: {context}\n" if context else ""
    )
    return system, user


if __name__ == '__main__':
    # Test prompt generation
    test_question = "Is the car above the cat?"

    print("=" * 80)
    print("FES TEXT PROMPT TEST")
    print("=" * 80)
    system, user = get_fes_text_prompt(test_question, n=4)
    print(f"SYSTEM:\n{system}\n")
    print(f"USER:\n{user}\n")

    print("\n" + "=" * 80)
    print("FCS TEXT PROMPT TEST")
    print("=" * 80)
    system, user = get_fcs_text_prompt(test_question, n=4)
    print(f"SYSTEM:\n{system}\n")
    print(f"USER:\n{user}\n")

