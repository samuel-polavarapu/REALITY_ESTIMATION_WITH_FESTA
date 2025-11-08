#!/usr/bin/env python3
"""
FESTA Evaluation with OpenAI GPT-5 and Gemini Pro APIs
Generates FES/FCS samples according to FESTA paper specifications
Runs for 2 samples and saves generated samples to output folder
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from PIL import Image
import torch
from datasets import load_dataset
from dotenv import load_dotenv
import traceback
import time
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import numpy as np

# Load environment variables
load_dotenv()

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import FESTA components
from src.festa_evaluation import Config, LLaVAModel, get_combined_pred

# Import API-based complement generator
try:
    from src.complement_generator import ComplementGenerator
except ImportError:
    print("ERROR: complement_generator.py not found")
    sys.exit(1)

# Import enhanced metrics and visualization
try:
    from src.festa_metrics import generate_comprehensive_report, FESTAMetrics, FESTAVisualizer
except ImportError:
    print("ERROR: festa_metrics.py not found")
    sys.exit(1)

# Import CSV export module
try:
    from src.metrics_csv_export import export_metrics_to_csv
except ImportError:
    print("WARNING: metrics_csv_export.py not found, CSV export will be skipped")
    export_metrics_to_csv = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    
    logger.info("=" * 80)
    logger.info("FESTA EVALUATION WITH OPENAI (TEXT & IMAGE)")
    logger.info("=" * 80)
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    num_samples = int(os.getenv('NUM_SAMPLES', '2'))
    skip_samples = int(os.getenv('SKIP_SAMPLES', '0'))
    logger.info(f"Number of samples: {num_samples}")
    logger.info(f"Skipping first: {skip_samples} samples")
    logger.info(f"Processing samples: {skip_samples + 1} to {skip_samples + num_samples}")

    # Create output directory
    output_dir = Path('output/api_run')
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / 'generated_samples'
    samples_dir.mkdir(exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Generated samples will be saved in: {samples_dir}")
    
    # Check API keys
    logger.info("\nChecking API keys...")
    openai_key = os.getenv('OPENAI_API_KEY')
    google_key = os.getenv('GOOGLE_API_KEY')
    
    if not openai_key or 'your-openai-api-key-here' in openai_key:
        logger.error("❌ OPENAI_API_KEY not properly configured in .env")
        return
    
    if not google_key or 'your-google-api-key-here' in google_key:
        logger.error("❌ GOOGLE_API_KEY not properly configured in .env")
        return
    
    logger.info("✓ API keys verified")
    
    # Initialize complement generator
    logger.info("\nInitializing API-based complement generator...")
    try:
        complement_gen = ComplementGenerator(
            output_dir=str(samples_dir),
            seed=42
        )
        logger.info("✓ Complement generator initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize complement generator: {e}")
        traceback.print_exc()
        return
    
    # Initialize LLaVA model
    logger.info("\nInitializing LLaVA model...")
    try:
        config = Config()
        llava = LLaVAModel(config, logger)
        logger.info("✓ LLaVA model loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load LLaVA: {e}")
        traceback.print_exc()
        return
    
    # Load dataset
    logger.info("\nLoading BLINK Spatial Relation dataset...")
    try:
        dataset = load_dataset(
            "BLINK-Benchmark/BLINK",
            "Spatial_Relation",
            split='val'
        )
        logger.info(f"✓ Loaded {len(dataset)} total samples")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        traceback.print_exc()
        return
    
    # Process samples
    results = []
    all_generated_samples = {
        'fes_text': [],
        'fes_image': [],
        'fcs_text': [],
        'fcs_image': []
    }
    
    for sample_idx in range(skip_samples, skip_samples + num_samples):
        actual_sample_num = sample_idx - skip_samples + 1
        logger.info("")
        logger.info("="*80)
        logger.info(f"PROCESSING SAMPLE {actual_sample_num}/{num_samples} (Dataset Index: {sample_idx})")
        logger.info("="*80)
        
        try:
            sample = dataset[sample_idx]
            sample_id = sample.get('id', f'sample_{sample_idx+1}')
            
            logger.info(f"Sample ID: {sample_id}")
            logger.info(f"Question: {sample['question']}")
            logger.info(f"Ground Truth: {sample['answer']}")
            
            # Save original image
            original_img_path = samples_dir / f"{sample_id}_original.png"
            sample['image_1'].save(original_img_path)
            logger.info(f"✓ Saved original image: {original_img_path}")
            
            # Original inference using probability-based top-k approach
            logger.info("\n→ Running original inference with probability-based prompts...")
            topk_results = llava.run_topk_inference(
                sample['image_1'],
                sample['question'],
                dict(zip(['A', 'B'], sample.get('choices', ['A', 'B']))),
                k=4,
                n_samples=5
            )
            logger.info(f"  Top-k results: {topk_results[:8]}")  # Show first 8 results

            # Get combined prediction
            orig_pred = get_combined_pred(topk_results)

            # Calculate confidence as weighted average
            if topk_results:
                pred_results = [prob for guess, prob in topk_results if guess == orig_pred]
                orig_conf = float(np.mean(pred_results)) if pred_results else 0.5
            else:
                orig_conf = 0.5

            logger.info(f"  Combined Prediction: {orig_pred} (confidence: {orig_conf:.3f})")

            # Generate FES Text samples using the new paraphrasing prompt
            logger.info("\n→ Generating FES TEXT paraphrases using OpenAI...")
            fes_text_samples = []
            try:
                text_item = {
                    'id': f"{sample_id}_fes_paraphrase",
                    'question': sample['question'],
                }

                complement = complement_gen.generate_complement(
                    text_item,
                    item_type='mcq',
                    generation_type='fes'
                )

                if complement and complement.complement.get('paraphrases'):
                    paraphrases = complement.complement['paraphrases']
                    for i, paraphrase in enumerate(paraphrases):
                        fes_text_samples.append({
                            'original_question': sample['question'],
                            'generated_question': paraphrase,
                            'method': 'OpenAI FES Paraphrasing',
                            'type': 'FES Text',
                        })
                        
                        # Save each paraphrase to a file
                        sample_file = samples_dir / f"{sample_id}_fes_text_{i+1}.json"
                        with open(sample_file, 'w') as f:
                            json.dump(fes_text_samples[-1], f, indent=2)
                        
                        logger.info(f"  ✓ FES Text {i+1} generated and saved")
                        logger.info(f"    Generated: {paraphrase[:60]}...")
                        all_generated_samples['fes_text'].append(fes_text_samples[-1])
            except Exception as e:
                logger.warning(f"  ✗ FES paraphrase generation failed: {e}")

            # Generate FCS Text sample using the new contradiction prompt
            logger.info("\n→ Generating FCS TEXT contradiction using OpenAI...")
            fcs_text_samples = []
            try:
                text_item = {
                    'id': f"{sample_id}_fcs_contradiction",
                    'question': sample['question'],
                }
                
                complement = complement_gen.generate_complement(
                    text_item,
                    item_type='mcq',
                    generation_type='fcs'
                )
                
                if complement and complement.complement.get('contradictory_questions'):
                    contradictions = complement.complement['contradictory_questions']
                    for i, contradiction in enumerate(contradictions):
                        fcs_text_samples.append({
                            'original_question': sample['question'],
                            'generated_question': contradiction,
                            'method': 'OpenAI FCS Contradiction',
                            'type': 'FCS Text',
                        })

                        # Save each contradiction to a file
                        sample_file = samples_dir / f"{sample_id}_fcs_text_{i+1}.json"
                        with open(sample_file, 'w') as f:
                            json.dump(fcs_text_samples[-1], f, indent=2)

                        logger.info(f"  ✓ FCS Text {i+1} generated and saved")
                        logger.info(f"    Generated: {contradiction[:60]}...")
                        all_generated_samples['fcs_text'].append(fcs_text_samples[-1])
            except Exception as e:
                logger.warning(f"  ✗ FCS contradiction generation failed: {e}")

            # Add a final delay after the FCS call for the current sample
            logger.info(f"Waiting for 21 seconds to avoid rate limiting...")
            time.sleep(21)

            # Generate FES Image samples using the new image prompt
            logger.info("\n→ Generating FES IMAGE variants using OpenAI...")
            fes_image_samples = []
            try:
                image_item = {
                    'id': f"{sample_id}_fes_image",
                    'image': sample['image_1'],
                    'question': sample['question'],
                }
                complement = complement_gen.generate_complement(
                    image_item,
                    item_type='image',
                    generation_type='fes'
                )
                if complement and complement.complement.get('image_paths'):
                    for i, path in enumerate(complement.complement['image_paths']):
                        fes_image_samples.append({
                            'original_image': str(original_img_path),
                            'generated_image': path,
                            'method': 'OpenAI PIL Transformations (FES)',
                            'type': 'FES Image',
                        })
                        logger.info(f"  ✓ FES Image {i+1} generated and saved to {path}")
                        all_generated_samples['fes_image'].append(fes_image_samples[-1])
            except Exception as e:
                logger.warning(f"  ✗ FES image generation failed: {e}")

            # Generate FCS Image sample using the new image prompt
            logger.info("\n→ Generating FCS IMAGE contradiction using OpenAI...")
            fcs_image_samples = []
            try:
                image_item = {
                    'id': f"{sample_id}_fcs_image",
                    'image': sample['image_1'],
                    'question': sample['question'],
                }
                complement = complement_gen.generate_complement(
                    image_item,
                    item_type='image',
                    generation_type='fcs'
                )
                if complement and complement.complement.get('image_paths'):
                    for i, path in enumerate(complement.complement['image_paths']):
                        fcs_image_samples.append({
                            'original_image': str(original_img_path),
                            'generated_image': path,
                            'method': 'OpenAI PIL Transformations (FCS)',
                            'type': 'FCS Image',
                        })
                        logger.info(f"  ✓ FCS Image {i+1} generated and saved to {path}")
                        all_generated_samples['fcs_image'].append(fcs_image_samples[-1])
            except Exception as e:
                logger.warning(f"  ✗ FCS image generation failed: {e}")

            # Record results (remove parentheses from ground_truth for consistency)
            ground_truth_clean = sample['answer'].strip("()")
            result = {
                'sample_id': sample_id,
                'question': sample['question'],
                'ground_truth': ground_truth_clean,
                'original_prediction': orig_pred,
                'original_confidence': float(orig_conf),
                'is_correct': orig_pred == ground_truth_clean,
                'generated_samples': {
                    'fes_text_count': len(fes_text_samples),
                    'fes_image_count': len(fes_image_samples),
                    'fcs_text_count': len(fcs_text_samples),
                    'fcs_image_count': len(fcs_image_samples)
                },
                'api_usage': {
                    'text_generation': 'OpenAI GPT-5',
                    'image_processing': 'Gemini Pro'
                }
            }

            results.append(result)

            logger.info("\n→ Sample Summary:")
            logger.info(f"  FES Text: {len(fes_text_samples)} generated")
            logger.info(f"  FES Image: {len(fes_image_samples)} generated")
            logger.info(f"  FCS Text: {len(fcs_text_samples)} generated")
            logger.info(f"  FCS Image: {len(fcs_image_samples)} generated")

            # Run inference on generated samples (both text and image)
            logger.info("\n→ Running inference on generated samples...")

            # Run inference on FES Text samples with probability-based prompts
            for gen_sample in fes_text_samples:
                logger.info(f"  Running probability-based inference on FES Text: {gen_sample['generated_question'][:60]}...")
                topk_results = llava.run_topk_inference(
                    sample['image_1'],
                    gen_sample['generated_question'],
                    dict(zip(['A', 'B'], sample.get('choices', ['A', 'B']))),
                    k=4,
                    n_samples=5
                )
                pred = get_combined_pred(topk_results)

                # Calculate confidence
                if topk_results:
                    pred_results = [prob for guess, prob in topk_results if guess == pred]
                    conf = float(np.mean(pred_results)) if pred_results else 0.5
                else:
                    conf = 0.5

                gen_sample['prediction'] = pred
                gen_sample['confidence'] = float(conf)
                gen_sample['is_correct'] = pred == ground_truth_clean
                gen_sample['topk_results'] = topk_results[:8]  # Store first 8 for debugging
                logger.info(f"    Prediction: {pred} (confidence: {conf:.3f})")

            # Run inference on FCS Text samples with probability-based prompts
            for gen_sample in fcs_text_samples:
                logger.info(f"  Running probability-based inference on FCS Text: {gen_sample['generated_question'][:60]}...")
                topk_results = llava.run_topk_inference(
                    sample['image_1'],
                    gen_sample['generated_question'],
                    dict(zip(['A', 'B'], sample.get('choices', ['A', 'B']))),
                    k=4,
                    n_samples=5
                )
                pred = get_combined_pred(topk_results)

                # Calculate confidence
                if topk_results:
                    pred_results = [prob for guess, prob in topk_results if guess == pred]
                    conf = float(np.mean(pred_results)) if pred_results else 0.5
                else:
                    conf = 0.5

                gen_sample['prediction'] = pred
                gen_sample['confidence'] = float(conf)
                # FCS flips the relation, so expected answer is opposite
                flipped_gt = 'B' if ground_truth_clean == 'A' else 'A'
                gen_sample['is_correct'] = pred == flipped_gt
                gen_sample['topk_results'] = topk_results[:8]  # Store first 8 for debugging
                logger.info(f"    Prediction: {pred} (confidence: {conf:.3f}, expected: {flipped_gt})")

            # Run inference on FES Image samples with probability-based prompts
            for gen_sample in fes_image_samples:
                logger.info(f"  Running probability-based inference on FES Image: {gen_sample['generated_image']}")
                generated_img = Image.open(gen_sample['generated_image'])
                topk_results = llava.run_topk_inference(
                    generated_img,
                    sample['question'],
                    dict(zip(['A', 'B'], sample.get('choices', ['A', 'B']))),
                    k=4,
                    n_samples=5
                )
                pred = get_combined_pred(topk_results)

                # Calculate confidence
                if topk_results:
                    pred_results = [prob for guess, prob in topk_results if guess == pred]
                    conf = float(np.mean(pred_results)) if pred_results else 0.5
                else:
                    conf = 0.5

                gen_sample['prediction'] = pred
                gen_sample['confidence'] = float(conf)
                gen_sample['is_correct'] = pred == ground_truth_clean
                gen_sample['topk_results'] = topk_results[:8]  # Store first 8 for debugging
                logger.info(f"    Prediction: {pred} (confidence: {conf:.3f})")

            # Run inference on FCS Image samples with probability-based prompts
            for gen_sample in fcs_image_samples:
                logger.info(f"  Running probability-based inference on FCS Image: {gen_sample['generated_image']}")
                generated_img = Image.open(gen_sample['generated_image'])
                topk_results = llava.run_topk_inference(
                    generated_img,
                    sample['question'],
                    dict(zip(['A', 'B'], sample.get('choices', ['A', 'B']))),
                    k=4,
                    n_samples=5
                )
                pred = get_combined_pred(topk_results)

                # Calculate confidence
                if topk_results:
                    pred_results = [prob for guess, prob in topk_results if guess == pred]
                    conf = float(np.mean(pred_results)) if pred_results else 0.5
                else:
                    conf = 0.5

                gen_sample['prediction'] = pred
                gen_sample['confidence'] = float(conf)
                # FCS flips the relation, so expected answer is opposite
                flipped_gt = 'B' if ground_truth_clean == 'A' else 'A'
                gen_sample['is_correct'] = pred == flipped_gt
                gen_sample['topk_results'] = topk_results[:8]  # Store first 8 for debugging
                logger.info(f"    Prediction: {pred} (confidence: {conf:.3f}, expected: {flipped_gt})")

        except Exception as e:
            logger.error(f"❌ Failed to process sample {sample_idx + 1}: {e}")
            traceback.print_exc()

    # Calculate and log metrics - ENHANCED FOR AUROC
    # Initialize all metric variables with default values
    auroc = None
    accuracy = 0.0
    precision = None
    recall = None
    f1 = None

    if results:
        # Original predictions (baseline)
        y_true_orig = [1 if r['ground_truth'] == 'A' else 0 for r in results]
        y_pred_orig = [1 if r['original_prediction'] == 'A' else 0 for r in results]
        y_scores_orig = [r['original_confidence'] if r['original_prediction'] == 'A' else 1 - r['original_confidence'] for r in results]

        # ENHANCED: Collect predictions from ALL generated samples (FES + FCS)
        # This creates diversity in predictions and ground truth for AUROC calculation
        y_true_all = []
        y_pred_all = []
        y_scores_all = []

        # Add original predictions
        y_true_all.extend(y_true_orig)
        y_pred_all.extend(y_pred_orig)
        y_scores_all.extend(y_scores_orig)

        # Add FES text predictions (should maintain same ground truth)
        for gen_sample in all_generated_samples.get('fes_text', []):
            if 'prediction' in gen_sample and 'is_correct' in gen_sample:
                # FES should maintain original ground truth
                original_gt = next((r['ground_truth'] for r in results
                                   if r['question'] == gen_sample['original_question']), 'A')
                y_true_all.append(1 if original_gt == 'A' else 0)
                y_pred_all.append(1 if gen_sample['prediction'] == 'A' else 0)
                y_scores_all.append(gen_sample['confidence'] if gen_sample['prediction'] == 'A'
                                  else 1 - gen_sample['confidence'])

        # Add FCS text predictions (should flip ground truth for contradictions)
        for gen_sample in all_generated_samples.get('fcs_text', []):
            if 'prediction' in gen_sample and 'is_correct' in gen_sample:
                # FCS flips the relation, so expected ground truth should be opposite
                original_gt = next((r['ground_truth'] for r in results
                                   if r['question'] == gen_sample['original_question']), 'A')
                # For FCS, the expected answer is the opposite (flipped relation)
                flipped_gt = 'B' if original_gt == 'A' else 'A'
                y_true_all.append(1 if flipped_gt == 'A' else 0)
                y_pred_all.append(1 if gen_sample['prediction'] == 'A' else 0)
                y_scores_all.append(gen_sample['confidence'] if gen_sample['prediction'] == 'A'
                                  else 1 - gen_sample['confidence'])

        # Calculate metrics with enhanced dataset
        if len(y_true_all) > 0:
            accuracy_orig = accuracy_score(y_true_orig, y_pred_orig) if len(y_true_orig) > 0 else 0.0
            accuracy_all = accuracy_score(y_true_all, y_pred_all)

            # Set accuracy variable for use in reports
            accuracy = accuracy_all

            # AUROC calculation with enhanced dataset
            if len(np.unique(y_true_all)) > 1:
                auroc = roc_auc_score(y_true_all, y_scores_all)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true_all, y_pred_all, average='binary', zero_division=0
                )

                logger.info("\n" + "="*80)
                logger.info("PERFORMANCE METRICS (Enhanced with FES/FCS samples)")
                logger.info("="*80)
                logger.info(f"  AUROC: {auroc:.4f} (calculated from {len(y_true_all)} predictions)")
                logger.info(f"  Accuracy (Original): {accuracy_orig:.4f}")
                logger.info(f"  Accuracy (All): {accuracy_all:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1-Score: {f1:.4f}")
                logger.info(f"  Classes distribution: {np.bincount(y_true_all)}")
                logger.info("="*80)
            else:
                logger.warning(f"AUROC calculation skipped: only {len(np.unique(y_true_all))} class present in {len(y_true_all)} predictions")
                logger.warning("Need samples with different ground truth labels for AUROC")
                auroc = None
                precision = None
                recall = None
                f1 = None
                logger.info(f"Accuracy (All): {accuracy:.4f}")
        else:
            logger.warning("No predictions available for metrics calculation")
            auroc = None
            accuracy = 0.0
            precision = None
            recall = None
            f1 = None

    # Save overall results
    logger.info("")
    logger.info("="*80)
    logger.info("SAVING RESULTS")
    logger.info("="*80)

    # Generate timestamp for unique report filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save to output/api_run (existing location)
    results_file = output_dir / 'api_evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_samples': num_samples,
            'api_configuration': {
                'text_mcq_generation': 'OpenAI GPT-5',
                'image_processing': 'Gemini Pro'
            },
            'samples_processed': len(results),
            'metrics': {
                'auroc': float(auroc) if auroc is not None else None,
                'accuracy': float(accuracy) if 'accuracy' in locals() else 0.0,
                'precision': float(precision) if precision is not None else None,
                'recall': float(recall) if recall is not None else None,
                'f1_score': float(f1) if f1 is not None else None,
                'total_predictions': len(y_true_all) if 'y_true_all' in locals() else 0,
                'classes_in_ground_truth': len(np.unique(y_true_all)) if 'y_true_all' in locals() else 0
            },
            'results': results,
            'generated_samples_summary': {
                'fes_text_total': len(all_generated_samples['fes_text']),
                'fes_image_total': len(all_generated_samples['fes_image']),
                'fcs_text_total': len(all_generated_samples['fcs_text']),
                'fcs_image_total': len(all_generated_samples['fcs_image'])
            },
            'all_generated_samples': all_generated_samples
        }, f, indent=2)

    logger.info(f"✓ Results saved: {results_file}")

    # Save comprehensive report to reports folder with timestamp
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    report_json_file = reports_dir / f'festa_report_{timestamp}.json'

    # Prepare comprehensive report data
    report_data = {
        'run_info': {
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_samples_requested': num_samples,
            'skip_samples': skip_samples,
            'samples_processed': len(results),
            'sample_range': f'{skip_samples + 1} to {skip_samples + num_samples}'
        },
        'api_configuration': {
            'text_generation': 'OpenAI GPT-4o-mini',
            'image_analysis': 'Google Gemini Pro',
            'image_transformation': 'PIL (local)',
            'llava_model': 'llava-hf/llava-v1.6-mistral-7b-hf'
        },
        'metrics': {
            'auroc': float(auroc) if auroc is not None else None,
            'accuracy': float(accuracy) if 'accuracy' in locals() else 0.0,
            'precision': float(precision) if precision is not None else None,
            'recall': float(recall) if recall is not None else None,
            'f1_score': float(f1) if f1 is not None else None,
            'total_predictions': len(y_true_all) if 'y_true_all' in locals() else 0,
            'classes_in_ground_truth': len(np.unique(y_true_all)) if 'y_true_all' in locals() else 0,
            'auroc_available': auroc is not None,
            'auroc_note': 'AUROC requires at least 2 classes in ground truth' if auroc is None else 'AUROC calculated from FES/FCS predictions'
        },
        'sample_details': results,
        'generated_samples_summary': {
            'fes_text_total': len(all_generated_samples['fes_text']),
            'fes_image_total': len(all_generated_samples['fes_image']),
            'fcs_text_total': len(all_generated_samples['fcs_text']),
            'fcs_image_total': len(all_generated_samples['fcs_image'])
        },
        'generation_details': all_generated_samples
    }

    with open(report_json_file, 'w') as f:
        json.dump(report_data, f, indent=2)

    logger.info(f"✓ Comprehensive report saved: {report_json_file}")

    # Create summary report
    report_file = output_dir / 'api_evaluation_report.md'
    with open(report_file, 'w') as f:
        f.write(f"# FESTA Evaluation with APIs - {num_samples} Samples\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## API Configuration\n\n")
        f.write(f"- **Text MCQ Generation (FES/FCS)**: OpenAI GPT-5 API\n")
        f.write(f"- **Image Processing (FES/FCS)**: Gemini Pro API\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Samples Processed**: {len(results)}\n")
        f.write(f"- **FES Text Generated**: {len(all_generated_samples['fes_text'])}\n")
        f.write(f"- **FES Image Generated**: {len(all_generated_samples['fes_image'])}\n")
        f.write(f"- **FCS Text Generated**: {len(all_generated_samples['fcs_text'])}\n")
        f.write(f"- **FCS Image Generated**: {len(all_generated_samples['fcs_image'])}\n")
        if results:
            if auroc is not None:
                f.write(f"- **AUROC**: {auroc:.4f} (from {len(y_true_all) if 'y_true_all' in locals() else 0} predictions)\n")
                f.write(f"- **Accuracy**: {accuracy:.2%}\n")
                f.write(f"- **Precision**: {precision:.4f}\n")
                f.write(f"- **Recall**: {recall:.4f}\n")
                f.write(f"- **F1-Score**: {f1:.4f}\n\n")
            else:
                acc_val = accuracy if 'accuracy' in locals() else 0.0
                f.write(f"- **Accuracy**: {acc_val:.2%}\n")
                f.write(f"- **Note**: AUROC not available (requires diverse ground truth labels)\n\n")
        else:
            f.write("- **Accuracy**: 0.00%\n\n")

        f.write(f"## Generated Samples Location\n\n")
        f.write(f"All generated samples saved in: `{samples_dir}`\n\n")

        f.write(f"## Sample Details\n\n")
        for r in results:
            f.write(f"### {r['sample_id']}\n\n")
            f.write(f"- **Question**: {r['question']}\n")
            f.write(f"- **Prediction**: {r['original_prediction']} (GT: {r['ground_truth']})\n")
            f.write(f"- **Correct**: {'✓' if r['is_correct'] else '✗'}\n")
            f.write(f"- **Generated Samples**:\n")
            f.write(f"  - FES Text: {r['generated_samples']['fes_text_count']}\n")
            f.write(f"  - FES Image: {r['generated_samples']['fes_image_count']}\n")
            f.write(f"  - FCS Text: {r['generated_samples']['fcs_text_count']}\n")
            f.write(f"  - FCS Image: {r['generated_samples']['fcs_image_count']}\n\n")

    logger.info(f"✓ Report saved: {report_file}")

    # Generate comprehensive metrics and visualizations
    logger.info("")
    logger.info("="*80)
    logger.info("GENERATING COMPREHENSIVE METRICS AND VISUALIZATIONS")
    logger.info("="*80)

    try:
        # Add missing fields to results for compatibility
        logger.info(f"Mapping {len(results)} results to generated samples...")
        logger.info(f"  FES Text total: {len(all_generated_samples.get('fes_text', []))}")
        logger.info(f"  FES Image total: {len(all_generated_samples.get('fes_image', []))}")
        logger.info(f"  FCS Text total: {len(all_generated_samples.get('fcs_text', []))}")
        logger.info(f"  FCS Image total: {len(all_generated_samples.get('fcs_image', []))}")

        for result in results:
            sample_id = result['sample_id']

            # Ensure all sample lists exist
            if 'fes_text_samples' not in result:
                result['fes_text_samples'] = []
            if 'fes_image_samples' not in result:
                result['fes_image_samples'] = []
            if 'fcs_text_samples' not in result:
                result['fcs_text_samples'] = []
            if 'fcs_image_samples' not in result:
                result['fcs_image_samples'] = []

            # Map from all_generated_samples to result structure
            for sample in all_generated_samples.get('fes_text', []):
                if sample.get('original_question') == result['question']:
                    result['fes_text_samples'].append(sample)

            # Map FES images by checking if they were generated for this sample
            original_img_name = f"{sample_id}_original.png"
            for sample in all_generated_samples.get('fes_image', []):
                if 'original_image' in sample and sample_id in sample['original_image']:
                    result['fes_image_samples'].append(sample)

            for sample in all_generated_samples.get('fcs_text', []):
                if sample.get('original_question') == result['question']:
                    result['fcs_text_samples'].append(sample)

            # Map FCS images by checking if they were generated for this sample
            for sample in all_generated_samples.get('fcs_image', []):
                if 'original_image' in sample and sample_id in sample['original_image']:
                    result['fcs_image_samples'].append(sample)

            logger.info(f"  {sample_id}: FES_text={len(result['fes_text_samples'])}, FES_img={len(result['fes_image_samples'])}, FCS_text={len(result['fcs_text_samples'])}, FCS_img={len(result['fcs_image_samples'])}")

        # Generate comprehensive report with all metrics and visualizations (only if not disabled)
        generate_docs = os.getenv('GENERATE_DOCS', 'true').lower() != 'false'
        generate_reports = os.getenv('GENERATE_REPORTS', 'true').lower() != 'false'

        comprehensive_metrics = None

        if generate_docs or generate_reports:
            comprehensive_metrics = generate_comprehensive_report(results, output_dir=str(output_dir))

            # Save comprehensive metrics to report
            report_data['comprehensive_metrics'] = comprehensive_metrics
            with open(report_json_file, 'w') as f:
                json.dump(report_data, f, indent=2)

            logger.info(f"✓ Enhanced report with comprehensive metrics saved: {report_json_file}")

            # Export metrics to CSV
            if export_metrics_to_csv is not None:
                try:
                    logger.info("\nExporting metrics to CSV format...")
                    csv_file = export_metrics_to_csv(results, comprehensive_metrics, output_dir=str(output_dir))
                    logger.info(f"✓ Metrics exported to CSV: {csv_file}")
                except Exception as csv_error:
                    logger.error(f"Failed to export CSV: {csv_error}")
            else:
                logger.warning("CSV export module not available, skipping CSV generation")
        else:
            logger.info("⊗ Documentation generation disabled - skipping comprehensive reports")
            # Still save basic JSON without comprehensive metrics
            with open(report_json_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            logger.info(f"✓ Basic results saved: {report_json_file}")

    except Exception as e:
        logger.error(f"Failed to generate comprehensive metrics: {e}")
        traceback.print_exc()

    logger.info("")
    logger.info("="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Samples processed: {len(results)}")
    logger.info(f"Total FES Text: {len(all_generated_samples['fes_text'])}")
    logger.info(f"Total FES Image: {len(all_generated_samples['fes_image'])}")
    logger.info(f"Total FCS Text: {len(all_generated_samples['fcs_text'])}")
    logger.info(f"Total FCS Image: {len(all_generated_samples['fcs_image'])}")
    logger.info(f"")
    logger.info(f"Results saved in: {output_dir}")
    logger.info(f"Generated samples saved in: {samples_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
