#!/usr/bin/env python3
"""
CSV Export Module for FESTA Metrics
Exports comprehensive metrics to CSV format for easy analysis
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def export_metrics_to_csv(results: List[Dict[str, Any]],
                          comprehensive_metrics: Dict[str, Any],
                          output_dir: str = "output/api_run") -> str:
    """
    Export FESTA metrics to CSV files

    Args:
        results: List of sample results
        comprehensive_metrics: Dictionary of calculated metrics
        output_dir: Directory to save CSV files

    Returns:
        Path to the main metrics CSV file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. Export comprehensive metrics summary
    metrics_summary_file = output_path / f'metrics_summary_{timestamp}.csv'

    with open(metrics_summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header row
        writer.writerow(['Metric Type', 'Sample Count', 'AUROC', 'AUPRC', 'Accuracy',
                        'Precision', 'Recall', 'F1-Score', 'Brier Score', 'ECE'])

        # Write data rows
        for metric_key, metric_values in comprehensive_metrics.items():
            writer.writerow([
                metric_key.upper().replace('_', ' '),
                metric_values.get('sample_count', 0),
                f"{metric_values.get('auroc', 0):.4f}",
                f"{metric_values.get('auprc', 0):.4f}",
                f"{metric_values.get('accuracy', 0):.4f}",
                f"{metric_values.get('precision', 0):.4f}",
                f"{metric_values.get('recall', 0):.4f}",
                f"{metric_values.get('f1_score', 0):.4f}",
                f"{metric_values.get('brier_score', 0):.4f}",
                f"{metric_values.get('ece', 0):.4f}",
            ])

    logger.info(f"✓ Metrics summary exported to: {metrics_summary_file}")

    # 2. Export detailed sample-level results
    samples_detail_file = output_path / f'samples_detail_{timestamp}.csv'

    with open(samples_detail_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample ID', 'Question', 'Ground Truth', 'Original Prediction',
                        'Confidence', 'Is Correct', 'FES Text Count', 'FES Image Count',
                        'FCS Text Count', 'FCS Image Count'])

        for result in results:
            writer.writerow([
                result.get('sample_id', ''),
                result.get('question', ''),
                result.get('ground_truth', ''),
                result.get('original_prediction', ''),
                f"{result.get('original_confidence', 0):.4f}",
                result.get('is_correct', False),
                result.get('generated_samples', {}).get('fes_text_count', 0),
                result.get('generated_samples', {}).get('fes_image_count', 0),
                result.get('generated_samples', {}).get('fcs_text_count', 0),
                result.get('generated_samples', {}).get('fcs_image_count', 0),
            ])

    logger.info(f"✓ Sample details exported to: {samples_detail_file}")

    # 3. Export FES/FCS predictions detail
    predictions_file = output_path / f'predictions_detail_{timestamp}.csv'

    with open(predictions_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample ID', 'Type', 'Category', 'Original Question',
                        'Generated Question/Image', 'Prediction', 'Confidence',
                        'Ground Truth', 'Is Correct'])

        for result in results:
            sample_id = result.get('sample_id', '')
            gt = result.get('ground_truth', '')

            # FES Text samples
            for sample in result.get('fes_text_samples', []):
                writer.writerow([
                    sample_id,
                    'FES',
                    'Text',
                    sample.get('original_question', ''),
                    sample.get('generated_question', ''),
                    sample.get('prediction', ''),
                    f"{sample.get('confidence', 0):.4f}",
                    gt,
                    sample.get('is_correct', False),
                ])

            # FCS Text samples (flipped GT)
            flipped_gt = 'B' if gt == 'A' else 'A'
            for sample in result.get('fcs_text_samples', []):
                writer.writerow([
                    sample_id,
                    'FCS',
                    'Text',
                    sample.get('original_question', ''),
                    sample.get('generated_question', ''),
                    sample.get('prediction', ''),
                    f"{sample.get('confidence', 0):.4f}",
                    flipped_gt,
                    sample.get('is_correct', False),
                ])

            # FES Image samples
            for sample in result.get('fes_image_samples', []):
                writer.writerow([
                    sample_id,
                    'FES',
                    'Image',
                    sample.get('original_image', ''),
                    sample.get('generated_image', ''),
                    sample.get('prediction', ''),
                    f"{sample.get('confidence', 0):.4f}",
                    gt,
                    sample.get('is_correct', False),
                ])

            # FCS Image samples (flipped GT)
            for sample in result.get('fcs_image_samples', []):
                writer.writerow([
                    sample_id,
                    'FCS',
                    'Image',
                    sample.get('original_image', ''),
                    sample.get('generated_image', ''),
                    sample.get('prediction', ''),
                    f"{sample.get('confidence', 0):.4f}",
                    flipped_gt,
                    sample.get('is_correct', False),
                ])

    logger.info(f"✓ Predictions detail exported to: {predictions_file}")

    # 4. Create a master summary CSV with key metrics
    master_summary_file = output_path / f'master_metrics_{timestamp}.csv'

    with open(master_summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])

        # Overall statistics
        writer.writerow(['Total Samples Processed', len(results)])
        writer.writerow(['Timestamp', datetime.now().isoformat()])
        writer.writerow(['', ''])

        # Aggregate metrics (averaging across all types)
        all_auroc = []
        all_auprc = []
        all_accuracy = []
        all_f1 = []

        for metric_key, metric_values in comprehensive_metrics.items():
            if metric_values.get('auroc') and metric_values.get('auroc') > 0:
                all_auroc.append(metric_values['auroc'])
            if metric_values.get('auprc') and metric_values.get('auprc') > 0:
                all_auprc.append(metric_values['auprc'])
            if metric_values.get('accuracy'):
                all_accuracy.append(metric_values['accuracy'])
            if metric_values.get('f1_score'):
                all_f1.append(metric_values['f1_score'])

        if all_auroc:
            writer.writerow(['Average AUROC', f"{sum(all_auroc)/len(all_auroc):.4f}"])
        if all_auprc:
            writer.writerow(['Average AUPRC', f"{sum(all_auprc)/len(all_auprc):.4f}"])
        if all_accuracy:
            writer.writerow(['Average Accuracy', f"{sum(all_accuracy)/len(all_accuracy):.4f}"])
        if all_f1:
            writer.writerow(['Average F1-Score', f"{sum(all_f1)/len(all_f1):.4f}"])

        writer.writerow(['', ''])
        writer.writerow(['Category-wise Metrics', ''])
        writer.writerow(['', ''])

        for metric_key, metric_values in comprehensive_metrics.items():
            writer.writerow([f'{metric_key.upper()} - AUROC', f"{metric_values.get('auroc', 0):.4f}"])
            writer.writerow([f'{metric_key.upper()} - Accuracy', f"{metric_values.get('accuracy', 0):.4f}"])
            writer.writerow([f'{metric_key.upper()} - F1-Score', f"{metric_values.get('f1_score', 0):.4f}"])
            writer.writerow(['', ''])

    logger.info(f"✓ Master summary exported to: {master_summary_file}")

    return str(metrics_summary_file)


def load_and_export_from_json(json_file_path: str, output_dir: str = None) -> str:
    """
    Load results from JSON and export to CSV

    Args:
        json_file_path: Path to the JSON results file
        output_dir: Directory to save CSV files (default: same as JSON)

    Returns:
        Path to the main metrics CSV file
    """
    json_path = Path(json_file_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    comprehensive_metrics = data.get('comprehensive_metrics', {})

    if output_dir is None:
        output_dir = str(json_path.parent)

    return export_metrics_to_csv(results, comprehensive_metrics, output_dir)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else None

        print(f"Converting {json_file} to CSV...")
        csv_file = load_and_export_from_json(json_file, output)
        print(f"✓ CSV files created successfully!")
        print(f"Main file: {csv_file}")
    else:
        print("Usage: python metrics_csv_export.py <json_file_path> [output_dir]")

