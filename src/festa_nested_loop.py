#!/usr/bin/env python3
"""
FESTA with Complete 4×14 Inference Strategy
Implements nested loops for FES and FCS combinations:
- 4 text variations × 14 image variations = 56 combinations each
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from PIL import Image
import torch
from datasets import load_dataset
from dotenv import load_dotenv
import traceback
import time
import numpy as np
# Load environment variables
load_dotenv()
# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))
# Import FESTA components
from src.festa_evaluation import Config, LLaVAModel, get_combined_pred
from src.complement_generator import ComplementGenerator
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
def run_nested_inference(llava, fes_mcqs, fes_images, sample, ground_truth, inference_type='FES'):
    """
    Run nested loop inference: 4 MCQs × 14 images = 56 combinations
    """
    predictions = []
    total_combinations = len(fes_mcqs) * len(fes_images)
    logger.info(f"\n{'='*80}")
    logger.info(f"{inference_type} NESTED LOOP: {len(fes_mcqs)} MCQs × {len(fes_images)} Images = {total_combinations}")
    logger.info(f"{'='*80}")
    combination_count = 0
    # OUTER LOOP: MCQ variations (4)
    for mcq_idx, mcq_text in enumerate(fes_mcqs, 1):
        logger.info(f"\n[{inference_type}] MCQ {mcq_idx}/4: {mcq_text[:60]}...")
        # INNER LOOP: Image variations (14)
        for img_idx, img_path in enumerate(fes_images, 1):
            combination_count += 1
            image = Image.open(img_path)
            topk_results = llava.run_topk_inference(
                image, mcq_text,
                dict(zip(['A', 'B'], sample.get('choices', ['yes', 'no']))),
                k=4, n_samples=5
            )
            pred = get_combined_pred(topk_results)
            if topk_results:
                pred_results = [prob for guess, prob in topk_results if guess == pred]
                conf = float(np.mean(pred_results)) if pred_results else 0.5
            else:
                conf = 0.5
            if inference_type == 'FES':
                expected_answer = ground_truth
            else:
                expected_answer = 'B' if ground_truth == 'A' else 'A'
            is_correct = (pred == expected_answer)
            predictions.append({
                'combination_id': combination_count,
                'mcq_index': mcq_idx,
                'image_index': img_idx,
                'mcq_text': mcq_text,
                'image_path': img_path,
                'prediction': pred,
                'confidence': conf,
                'ground_truth': ground_truth,
                'expected_answer': expected_answer,
                'is_correct': is_correct,
                'type': inference_type
            })
            if combination_count % 14 == 0:
                accuracy = sum(p['is_correct'] for p in predictions[-14:]) / 14
                logger.info(f"  MCQ {mcq_idx} complete: Accuracy: {accuracy:.3f}")
    total_correct = sum(p['is_correct'] for p in predictions)
    overall_accuracy = total_correct / total_combinations
    logger.info(f"\n{inference_type} Complete: {total_correct}/{total_combinations} correct ({overall_accuracy:.4f})")
    return predictions
def main():
    logger.info("=" * 80)
    logger.info("FESTA 4×14 NESTED LOOP STRATEGY")
    logger.info("=" * 80)
    num_samples = int(os.getenv('NUM_SAMPLES', '3'))
    skip_samples = int(os.getenv('SKIP_SAMPLES', '0'))
    output_dir = Path('output/api_run')
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / 'generated_samples'
    samples_dir.mkdir(exist_ok=True)
    logger.info(f"Processing {num_samples} samples")
    complement_gen = ComplementGenerator(output_dir=str(samples_dir), seed=42)
    config = Config()
    llava = LLaVAModel(config, logger)
    dataset = load_dataset("BLINK-Benchmark/BLINK", "Spatial_Relation", split='val')
    all_results = []
    all_fes_predictions = []
    all_fcs_predictions = []
    for sample_idx in range(skip_samples, skip_samples + num_samples):
        actual_sample_num = sample_idx - skip_samples + 1
        logger.info(f"\n{'='*80}")
        logger.info(f"SAMPLE {actual_sample_num}/{num_samples}")
        logger.info(f"{'='*80}")
        try:
            sample = dataset[sample_idx]
            sample_id = sample.get('id', f'sample_{sample_idx+1}')
            ground_truth = sample['answer'].strip("()")
            logger.info(f"Question: {sample['question']}")
            logger.info(f"Ground Truth: {ground_truth}")
            original_img_path = samples_dir / f"{sample_id}_original.png"
            sample['image_1'].save(original_img_path)
            # Generate 4 FES MCQs
            logger.info("\n→ Generating 4 FES MCQs...")
            complement = complement_gen.generate_complement(
                {'id': f"{sample_id}_fes", 'question': sample['question'], 'n': 4},
                item_type='mcq', generation_type='fes'
            )
            fes_mcqs = []
            if complement and complement.complement.get('paraphrases'):
                fes_mcqs = complement.complement['paraphrases'][:4]
            while len(fes_mcqs) < 4:
                fes_mcqs.append(sample['question'])
            # Generate 14 FES Images
            logger.info("\n→ Generating 14 FES Images...")
            complement = complement_gen.generate_complement(
                {'id': f"{sample_id}_img", 'image': sample['image_1'], 
                 'question': sample['question'], 'num_variants': 14},
                item_type='image', generation_type='fes'
            )
            fes_image_paths = []
            if complement and complement.complement.get('image_paths'):
                fes_image_paths = complement.complement['image_paths'][:14]
            while len(fes_image_paths) < 14:
                fes_image_paths.append(str(original_img_path))
            logger.info(f"✓ Generated {len(fes_image_paths)} images")
            # FES Nested Loop (4×14=56)
            fes_predictions = run_nested_inference(
                llava, fes_mcqs, fes_image_paths, sample, ground_truth, 'FES'
            )
            all_fes_predictions.extend(fes_predictions)
            # Generate 4 FCS MCQs
            time.sleep(21)
            logger.info("\n→ Generating 4 FCS MCQs...")
            complement = complement_gen.generate_complement(
                {'id': f"{sample_id}_fcs", 'question': sample['question'], 'n': 4},
                item_type='mcq', generation_type='fcs'
            )
            fcs_mcqs = []
            if complement and complement.complement.get('contradictory_questions'):
                fcs_mcqs = complement.complement['contradictory_questions'][:4]
            while len(fcs_mcqs) < 4:
                fcs_mcqs.append(sample['question'].replace('Is', 'Is not'))
            # FCS Nested Loop (4×14=56)
            fcs_predictions = run_nested_inference(
                llava, fcs_mcqs, fes_image_paths, sample, ground_truth, 'FCS'
            )
            all_fcs_predictions.extend(fcs_predictions)
            all_results.append({
                'sample_id': sample_id,
                'question': sample['question'],
                'ground_truth': ground_truth,
                'fes_combinations': len(fes_predictions),
                'fcs_combinations': len(fcs_predictions),
                'fes_accuracy': sum(p['is_correct'] for p in fes_predictions) / len(fes_predictions),
                'fcs_accuracy': sum(p['is_correct'] for p in fcs_predictions) / len(fcs_predictions)
            })
        except Exception as e:
            logger.error(f"Error: {e}")
            traceback.print_exc()
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL METRICS")
    logger.info("=" * 80)
    # FES Metrics
    fes_y_true = [1 if p['expected_answer'] == 'A' else 0 for p in all_fes_predictions]
    fes_y_pred = [1 if p['prediction'] == 'A' else 0 for p in all_fes_predictions]
    fes_y_scores = [p['confidence'] if p['prediction'] == 'A' else 1-p['confidence'] 
                     for p in all_fes_predictions]
    logger.info(f"\nFES ({len(all_fes_predictions)} predictions):")
    logger.info(f"  Accuracy:  {accuracy_score(fes_y_true, fes_y_pred):.4f}")
    logger.info(f"  Precision: {precision_score(fes_y_true, fes_y_pred, zero_division=0):.4f}")
    logger.info(f"  Recall:    {recall_score(fes_y_true, fes_y_pred, zero_division=0):.4f}")
    logger.info(f"  F1:        {f1_score(fes_y_true, fes_y_pred, zero_division=0):.4f}")
    if len(np.unique(fes_y_true)) > 1:
        logger.info(f"  AUROC:     {roc_auc_score(fes_y_true, fes_y_scores):.4f}")
    # FCS Metrics
    fcs_y_true = [1 if p['expected_answer'] == 'A' else 0 for p in all_fcs_predictions]
    fcs_y_pred = [1 if p['prediction'] == 'A' else 0 for p in all_fcs_predictions]
    fcs_y_scores = [p['confidence'] if p['prediction'] == 'A' else 1-p['confidence'] 
                     for p in all_fcs_predictions]
    logger.info(f"\nFCS ({len(all_fcs_predictions)} predictions):")
    logger.info(f"  Accuracy:  {accuracy_score(fcs_y_true, fcs_y_pred):.4f}")
    logger.info(f"  Precision: {precision_score(fcs_y_true, fcs_y_pred, zero_division=0):.4f}")
    logger.info(f"  Recall:    {recall_score(fcs_y_true, fcs_y_pred, zero_division=0):.4f}")
    logger.info(f"  F1:        {f1_score(fcs_y_true, fcs_y_pred, zero_division=0):.4f}")
    if len(np.unique(fcs_y_true)) > 1:
        logger.info(f"  AUROC:     {roc_auc_score(fcs_y_true, fcs_y_scores):.4f}")
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'nested_loop_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'strategy': '4×14 Nested Loop',
            'num_samples': num_samples,
            'total_fes': len(all_fes_predictions),
            'total_fcs': len(all_fcs_predictions),
            'results': all_results,
            'fes_predictions': all_fes_predictions,
            'fcs_predictions': all_fcs_predictions
        }, f, indent=2)
    logger.info(f"\n✓ Saved: {results_file}")
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)
if __name__ == '__main__':
    main()
