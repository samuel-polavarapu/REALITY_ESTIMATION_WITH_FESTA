#!/usr/bin/env python3
"""
Analyze FESTA results and generate calibrated risk-coverage charts
Uses reference implementation with sklearn LogisticRegression
"""

import json
import numpy as np
from pathlib import Path
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from festa_calibration import (
    calibrate_confidence,
    risk_coverage_peel_off_low,
    plot_risk_coverage,
    plot_accuracy_coverage,
    plot_multiple_risk_coverage,
    plot_multiple_accuracy_coverage,
    abstention_table,
    generate_calibrated_metrics
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_predictions_data(results_data):
    """Extract predictions data from results JSON.

    Returns:
        Dictionary with data for each category
    """
    data = {
        'original': {'confidence': [], 'correct': [], 'predictions': [], 'ground_truth': []},
        'fes_text': {'confidence': [], 'correct': [], 'predictions': [], 'ground_truth': []},
        'fes_image': {'confidence': [], 'correct': [], 'predictions': [], 'ground_truth': []},
        'fcs_text': {'confidence': [], 'correct': [], 'predictions': [], 'ground_truth': []},
        'fcs_image': {'confidence': [], 'correct': [], 'predictions': [], 'ground_truth': []},
    }

    all_samples = results_data.get('all_generated_samples', {})
    results = results_data.get('results', [])

    # Process original predictions
    for result in results:
        gt_binary = 1 if result['ground_truth'] == 'A' else 0
        pred_binary = 1 if result['original_prediction'] == 'A' else 0
        data['original']['confidence'].append(result['original_confidence'])
        data['original']['correct'].append(int(result['is_correct']))
        data['original']['predictions'].append(pred_binary)
        data['original']['ground_truth'].append(gt_binary)

    # Process FES Text
    for sample in all_samples.get('fes_text', []):
        if 'prediction' in sample and 'confidence' in sample:
            data['fes_text']['confidence'].append(sample['confidence'])
            data['fes_text']['correct'].append(int(sample.get('is_correct', False)))
            pred_binary = 1 if sample['prediction'] == 'A' else 0
            data['fes_text']['predictions'].append(pred_binary)
            # FES maintains original ground truth
            gt_binary = 1 if sample.get('is_correct') else 0  # Simplified
            data['fes_text']['ground_truth'].append(gt_binary)

    # Process FES Image
    for sample in all_samples.get('fes_image', []):
        if 'prediction' in sample and 'confidence' in sample:
            data['fes_image']['confidence'].append(sample['confidence'])
            data['fes_image']['correct'].append(int(sample.get('is_correct', False)))
            pred_binary = 1 if sample['prediction'] == 'A' else 0
            data['fes_image']['predictions'].append(pred_binary)
            gt_binary = 1 if sample.get('is_correct') else 0
            data['fes_image']['ground_truth'].append(gt_binary)

    # Process FCS Text
    for sample in all_samples.get('fcs_text', []):
        if 'prediction' in sample and 'confidence' in sample:
            data['fcs_text']['confidence'].append(sample['confidence'])
            data['fcs_text']['correct'].append(int(sample.get('is_correct', False)))
            pred_binary = 1 if sample['prediction'] == 'A' else 0
            data['fcs_text']['predictions'].append(pred_binary)
            gt_binary = 1 if sample.get('is_correct') else 0
            data['fcs_text']['ground_truth'].append(gt_binary)

    # Process FCS Image
    for sample in all_samples.get('fcs_image', []):
        if 'prediction' in sample and 'confidence' in sample:
            data['fcs_image']['confidence'].append(sample['confidence'])
            data['fcs_image']['correct'].append(int(sample.get('is_correct', False)))
            pred_binary = 1 if sample['prediction'] == 'A' else 0
            data['fcs_image']['predictions'].append(pred_binary)
            gt_binary = 1 if sample.get('is_correct') else 0
            data['fcs_image']['ground_truth'].append(gt_binary)

    return data


def generate_all_charts(results_path, output_dir='output/api_run/calibrated_charts'):
    """Generate all calibrated risk-coverage and accuracy-coverage charts.

    Args:
        results_path: Path to results JSON file
        output_dir: Directory to save charts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading results from: {results_path}")
    results_data = load_results(results_path)

    logger.info(f"Samples processed: {results_data['samples_processed']}")
    logger.info(f"Total predictions: {results_data['metrics']['total_predictions']}")
    logger.info(f"AUROC: {results_data['metrics']['auroc']:.4f}")

    # Extract data
    data = extract_predictions_data(results_data)

    # Generate individual category plots
    logger.info("\n" + "="*80)
    logger.info("GENERATING CALIBRATED RISK-COVERAGE AND ACCURACY-COVERAGE CHARTS")
    logger.info("="*80)

    for category, cat_data in data.items():
        if len(cat_data['confidence']) == 0:
            logger.warning(f"No data for {category}, skipping...")
            continue

        logger.info(f"\nProcessing {category.upper()}:")
        logger.info(f"  Samples: {len(cat_data['confidence'])}")

        confidence = np.array(cat_data['confidence'])
        correct = np.array(cat_data['correct'])
        predictions = np.array(cat_data['predictions'])
        ground_truth = np.array(cat_data['ground_truth'])

        # Calculate curves
        coverage, risk, accuracy, aurc = risk_coverage_peel_off_low(confidence, correct)

        logger.info(f"  AURC: {aurc:.4f}")
        logger.info(f"  Mean confidence: {confidence.mean():.4f}")
        logger.info(f"  Accuracy: {correct.mean():.4f}")

        # Plot individual curves
        plot_risk_coverage(
            coverage, risk,
            title=f"Risk-Coverage Curve - {category.upper().replace('_', ' ')}",
            output_path=output_path / f"{category}_risk_coverage.png",
            label=category.replace('_', ' ').title()
        )

        plot_accuracy_coverage(
            coverage, accuracy,
            title=f"Accuracy-Coverage Curve - {category.upper().replace('_', ' ')}",
            output_path=output_path / f"{category}_accuracy_coverage.png",
            label=category.replace('_', ' ').title()
        )

        # Print abstention table
        logger.info(f"\n  Abstention Analysis for {category.upper()}:")
        abstention_table(confidence, correct, step=0.1)

    # Generate combined plots
    logger.info("\n" + "="*80)
    logger.info("GENERATING COMBINED COMPARISON CHARTS")
    logger.info("="*80)

    # FES combined (text + image)
    fes_risk_dict = {}
    fes_acc_dict = {}
    for cat in ['fes_text', 'fes_image']:
        if len(data[cat]['confidence']) > 0:
            conf = np.array(data[cat]['confidence'])
            corr = np.array(data[cat]['correct'])
            cov, risk, acc, _ = risk_coverage_peel_off_low(conf, corr)
            fes_risk_dict[cat.replace('_', ' ').title()] = (cov, risk)
            fes_acc_dict[cat.replace('_', ' ').title()] = (cov, acc)

    if fes_risk_dict:
        plot_multiple_risk_coverage(fes_risk_dict,
                                    title="FES Risk-Coverage Comparison",
                                    output_path=output_path / "fes_risk_coverage_combined.png")
        plot_multiple_accuracy_coverage(fes_acc_dict,
                                       title="FES Accuracy-Coverage Comparison",
                                       output_path=output_path / "fes_accuracy_coverage_combined.png")

    # FCS combined (text + image)
    fcs_risk_dict = {}
    fcs_acc_dict = {}
    for cat in ['fcs_text', 'fcs_image']:
        if len(data[cat]['confidence']) > 0:
            conf = np.array(data[cat]['confidence'])
            corr = np.array(data[cat]['correct'])
            cov, risk, acc, _ = risk_coverage_peel_off_low(conf, corr)
            fcs_risk_dict[cat.replace('_', ' ').title()] = (cov, risk)
            fcs_acc_dict[cat.replace('_', ' ').title()] = (cov, acc)

    if fcs_risk_dict:
        plot_multiple_risk_coverage(fcs_risk_dict,
                                    title="FCS Risk-Coverage Comparison",
                                    output_path=output_path / "fcs_risk_coverage_combined.png")
        plot_multiple_accuracy_coverage(fcs_acc_dict,
                                       title="FCS Accuracy-Coverage Comparison",
                                       output_path=output_path / "fcs_accuracy_coverage_combined.png")

    # FESTA combined (all)
    festa_risk_dict = {}
    festa_acc_dict = {}
    for cat in ['original', 'fes_text', 'fes_image', 'fcs_text', 'fcs_image']:
        if len(data[cat]['confidence']) > 0:
            conf = np.array(data[cat]['confidence'])
            corr = np.array(data[cat]['correct'])
            cov, risk, acc, _ = risk_coverage_peel_off_low(conf, corr)
            festa_risk_dict[cat.replace('_', ' ').title()] = (cov, risk)
            festa_acc_dict[cat.replace('_', ' ').title()] = (cov, acc)

    if festa_risk_dict:
        plot_multiple_risk_coverage(festa_risk_dict,
                                    title="FESTA Risk-Coverage Comparison (All Categories)",
                                    output_path=output_path / "festa_risk_coverage_all.png")
        plot_multiple_accuracy_coverage(festa_acc_dict,
                                       title="FESTA Accuracy-Coverage Comparison (All Categories)",
                                       output_path=output_path / "festa_accuracy_coverage_all.png")

    logger.info("\n" + "="*80)
    logger.info("CHART GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"All charts saved to: {output_path}")
    logger.info(f"Total charts generated: {len(list(output_path.glob('*.png')))}")

    return output_path


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        results_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else 'output/api_run/calibrated_charts'
    else:
        results_path = 'output/api_run/api_evaluation_results.json'
        output_dir = 'output/api_run/calibrated_charts'

    if not Path(results_path).exists():
        logger.error(f"Results file not found: {results_path}")
        sys.exit(1)

    generate_all_charts(results_path, output_dir)

