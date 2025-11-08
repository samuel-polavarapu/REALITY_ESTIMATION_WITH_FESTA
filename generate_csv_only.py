#!/usr/bin/env python3
"""
Generate comprehensive CSV reports from existing FESTA results
No JSON output - CSV only
"""

import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

def generate_csv_reports(json_file_path, output_dir='output/api_run/csv_final'):
    """Generate all CSV reports from JSON results."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"Loading results from: {json_file_path}")
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Load comprehensive metrics
    metrics_file = Path(json_file_path).parent / 'comprehensive_metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            comprehensive_metrics = json.load(f)
    else:
        comprehensive_metrics = {}

    print(f"\n{'='*80}")
    print(f"GENERATING CSV REPORTS")
    print(f"{'='*80}\n")

    # 1. Comprehensive Metrics Summary CSV
    metrics_summary_file = output_path / f'comprehensive_metrics_{timestamp}.csv'
    with open(metrics_summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'Sample Count', 'AUROC', 'AUPRC', 'Accuracy',
                        'Precision', 'Recall', 'F1-Score', 'Brier Score', 'ECE'])

        for category, metrics in comprehensive_metrics.items():
            writer.writerow([
                category.upper().replace('_', ' '),
                metrics.get('sample_count', 0),
                f"{metrics.get('auroc', 0):.4f}",
                f"{metrics.get('auprc', 0):.4f}",
                f"{metrics.get('accuracy', 0):.4f}",
                f"{metrics.get('precision', 0):.4f}",
                f"{metrics.get('recall', 0):.4f}",
                f"{metrics.get('f1_score', 0):.4f}",
                f"{metrics.get('brier_score', 0):.4f}",
                f"{metrics.get('ece', 0):.4f}",
            ])

    print(f"✓ Generated: {metrics_summary_file}")

    # 2. Sample-Level Results CSV
    samples_file = output_path / f'sample_results_{timestamp}.csv'
    with open(samples_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample ID', 'Question', 'Ground Truth', 'Prediction',
                        'Confidence', 'Correct', 'FES Text', 'FES Image', 'FCS Text', 'FCS Image'])

        for result in data.get('results', []):
            writer.writerow([
                result.get('sample_id', ''),
                result.get('question', ''),
                result.get('ground_truth', ''),
                result.get('original_prediction', ''),
                f"{result.get('original_confidence', 0):.4f}",
                'Yes' if result.get('is_correct', False) else 'No',
                result.get('generated_samples', {}).get('fes_text_count', 0),
                result.get('generated_samples', {}).get('fes_image_count', 0),
                result.get('generated_samples', {}).get('fcs_text_count', 0),
                result.get('generated_samples', {}).get('fcs_image_count', 0),
            ])

    print(f"✓ Generated: {samples_file}")

    # 3. All Predictions Detail CSV
    predictions_file = output_path / f'all_predictions_{timestamp}.csv'
    with open(predictions_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample ID', 'Category', 'Type', 'Original Question',
                        'Generated Question/Image', 'Prediction', 'Confidence',
                        'Ground Truth', 'Expected Answer', 'Correct'])

        for result in data.get('results', []):
            sample_id = result.get('sample_id', '')
            gt = result.get('ground_truth', '')
            flipped_gt = 'B' if gt == 'A' else 'A'

            # FES Text
            for sample in data.get('all_generated_samples', {}).get('fes_text', []):
                if sample.get('original_question') == result['question']:
                    writer.writerow([
                        sample_id, 'FES', 'Text',
                        sample.get('original_question', ''),
                        sample.get('generated_question', ''),
                        sample.get('prediction', ''),
                        f"{sample.get('confidence', 0):.4f}",
                        gt, gt,
                        'Yes' if sample.get('is_correct', False) else 'No'
                    ])

            # FCS Text
            for sample in data.get('all_generated_samples', {}).get('fcs_text', []):
                if sample.get('original_question') == result['question']:
                    writer.writerow([
                        sample_id, 'FCS', 'Text',
                        sample.get('original_question', ''),
                        sample.get('generated_question', ''),
                        sample.get('prediction', ''),
                        f"{sample.get('confidence', 0):.4f}",
                        gt, flipped_gt,
                        'Yes' if sample.get('is_correct', False) else 'No'
                    ])

            # FES Image
            for sample in data.get('all_generated_samples', {}).get('fes_image', []):
                if sample_id in sample.get('original_image', ''):
                    writer.writerow([
                        sample_id, 'FES', 'Image',
                        sample.get('original_image', ''),
                        sample.get('generated_image', ''),
                        sample.get('prediction', ''),
                        f"{sample.get('confidence', 0):.4f}",
                        gt, gt,
                        'Yes' if sample.get('is_correct', False) else 'No'
                    ])

            # FCS Image
            for sample in data.get('all_generated_samples', {}).get('fcs_image', []):
                if sample_id in sample.get('original_image', ''):
                    writer.writerow([
                        sample_id, 'FCS', 'Image',
                        sample.get('original_image', ''),
                        sample.get('generated_image', ''),
                        sample.get('prediction', ''),
                        f"{sample.get('confidence', 0):.4f}",
                        gt, flipped_gt,
                        'Yes' if sample.get('is_correct', False) else 'No'
                    ])

    print(f"✓ Generated: {predictions_file}")

    # 4. Summary Statistics CSV
    summary_file = output_path / f'summary_statistics_{timestamp}.csv'
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])

        writer.writerow(['Total Samples', data.get('samples_processed', 0)])
        writer.writerow(['Timestamp', data.get('timestamp', '')])
        writer.writerow(['', ''])

        # Overall metrics
        metrics = data.get('metrics', {})
        writer.writerow(['Overall AUROC', f"{metrics.get('auroc', 0):.4f}"])
        writer.writerow(['Overall Accuracy', f"{metrics.get('accuracy', 0):.4f}"])
        writer.writerow(['Overall Precision', f"{metrics.get('precision', 0):.4f}"])
        writer.writerow(['Overall Recall', f"{metrics.get('recall', 0):.4f}"])
        writer.writerow(['Overall F1-Score', f"{metrics.get('f1_score', 0):.4f}"])
        writer.writerow(['Total Predictions', metrics.get('total_predictions', 0)])
        writer.writerow(['', ''])

        # Generation summary
        gen_summary = data.get('generated_samples_summary', {})
        writer.writerow(['FES Text Total', gen_summary.get('fes_text_total', 0)])
        writer.writerow(['FES Image Total', gen_summary.get('fes_image_total', 0)])
        writer.writerow(['FCS Text Total', gen_summary.get('fcs_text_total', 0)])
        writer.writerow(['FCS Image Total', gen_summary.get('fcs_image_total', 0)])
        writer.writerow(['', ''])

        # Category-wise AUROC
        writer.writerow(['Category-wise AUROC', ''])
        for category, cat_metrics in comprehensive_metrics.items():
            writer.writerow([f'{category.upper()} AUROC', f"{cat_metrics.get('auroc', 0):.4f}"])

    print(f"✓ Generated: {summary_file}")

    print(f"\n{'='*80}")
    print(f"CSV GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {output_path}")
    print(f"Total CSV files: 4")
    print(f"\nFiles generated:")
    print(f"  1. {metrics_summary_file.name} - Comprehensive metrics by category")
    print(f"  2. {samples_file.name} - Sample-level results")
    print(f"  3. {predictions_file.name} - All predictions detail")
    print(f"  4. {summary_file.name} - Summary statistics")

    return output_path


if __name__ == '__main__':
    json_file = sys.argv[1] if len(sys.argv) > 1 else 'output/api_run/api_evaluation_results.json'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'output/api_run/csv_final'

    if not Path(json_file).exists():
        print(f"ERROR: File not found: {json_file}")
        sys.exit(1)

    generate_csv_reports(json_file, output_dir)

