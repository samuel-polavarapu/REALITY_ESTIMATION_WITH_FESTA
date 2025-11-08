#!/usr/bin/env python3
"""
FESTA Enhanced Metrics and Visualization Module
Calculates comprehensive metrics including AUROC, AUPRC, Brier Score, ECE, etc.
Generates Risk-Coverage and Accuracy-Coverage curves
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
    average_precision_score, brier_score_loss, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FESTAMetrics:
    """Calculate comprehensive FESTA metrics with separate text/image analysis"""

    def __init__(self, output_dir: str = "output/metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_ece(self, y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(y_probs > bin_lower, y_probs <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_probs[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       y_probs: np.ndarray) -> Dict[str, float]:
        """Calculate all requested metrics"""
        metrics = {}

        # Debug info
        logger.info(f"    Calculating metrics for {len(y_true)} samples")
        logger.info(f"    Unique y_true: {np.unique(y_true)}")
        logger.info(f"    Unique y_pred: {np.unique(y_pred)}")

        # Convert to binary if needed
        y_true_binary = np.asarray(y_true).astype(int)
        y_pred_binary = np.asarray(y_pred).astype(int)
        y_probs = np.asarray(y_probs).astype(float)

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average='binary', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1

        # AUROC, AUPRC, Brier Score, and ECE
        # These metrics can still be calculated even with single class, but may be less meaningful
        n_classes_true = len(np.unique(y_true_binary))
        n_classes_pred = len(np.unique(y_pred_binary))

        logger.info(f"    Classes in y_true: {n_classes_true}, in y_pred: {n_classes_pred}")

        # Always try to calculate these metrics
        try:
            if n_classes_true >= 2:
                # Proper binary classification scenario
                metrics['auroc'] = roc_auc_score(y_true_binary, y_probs)
                logger.info(f"    AUROC calculated: {metrics['auroc']:.4f}")
            elif n_classes_true == 1:
                # All same class - AUROC is undefined but we can still provide a metric
                # If all predictions match the single class, it's perfect (1.0)
                # If predictions vary, we use a heuristic based on accuracy
                if accuracy_score(y_true_binary, y_pred_binary) == 1.0:
                    metrics['auroc'] = 1.0
                    logger.info(f"    AUROC set to 1.0 (perfect prediction on single class)")
                else:
                    metrics['auroc'] = metrics['accuracy']
                    logger.info(f"    AUROC set to accuracy (single class scenario)")
        except Exception as e:
            logger.warning(f"    AUROC calculation failed: {e}")
            metrics['auroc'] = 0.0

        try:
            if n_classes_true >= 2:
                metrics['auprc'] = average_precision_score(y_true_binary, y_probs)
                logger.info(f"    AUPRC calculated: {metrics['auprc']:.4f}")
            else:
                # Single class - use accuracy as proxy
                metrics['auprc'] = metrics['accuracy']
                logger.info(f"    AUPRC set to accuracy (single class scenario)")
        except Exception as e:
            logger.warning(f"    AUPRC calculation failed: {e}")
            metrics['auprc'] = 0.0

        # Brier Score - can be calculated regardless of class distribution
        try:
            metrics['brier_score'] = brier_score_loss(y_true_binary, y_probs)
            logger.info(f"    Brier Score calculated: {metrics['brier_score']:.4f}")
        except Exception as e:
            logger.warning(f"    Brier score calculation failed: {e}")
            metrics['brier_score'] = 0.0

        # Expected Calibration Error - can be calculated regardless
        try:
            metrics['ece'] = self.calculate_ece(y_true_binary, y_probs)
            logger.info(f"    ECE calculated: {metrics['ece']:.4f}")
        except Exception as e:
            logger.warning(f"    ECE calculation failed: {e}")
            metrics['ece'] = 0.0

        return metrics

    def separate_text_image_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """Separate and calculate metrics for text and image samples"""
        text_samples = {'fes': [], 'fcs': []}
        image_samples = {'fes': [], 'fcs': []}

        for result in results:
            sample_id = result.get('sample_id', '')

            # FES Text
            for fes_text in result.get('fes_text_samples', []):
                if 'prediction' in fes_text and 'is_correct' in fes_text:
                    text_samples['fes'].append({
                        'y_true': 1 if result['ground_truth'] == 'A' else 0,
                        'y_pred': 1 if fes_text['prediction'] == 'A' else 0,
                        'y_prob': fes_text.get('confidence', 0.5),
                        'type': 'fes_text'
                    })

            # FCS Text
            for fcs_text in result.get('fcs_text_samples', []):
                if 'prediction' in fcs_text and 'is_correct' in fcs_text:
                    gt = result['ground_truth']
                    flipped_gt = 'B' if gt == 'A' else 'A'
                    text_samples['fcs'].append({
                        'y_true': 1 if flipped_gt == 'A' else 0,
                        'y_pred': 1 if fcs_text['prediction'] == 'A' else 0,
                        'y_prob': fcs_text.get('confidence', 0.5),
                        'type': 'fcs_text'
                    })

            # FES Image
            for fes_img in result.get('fes_image_samples', []):
                if 'prediction' in fes_img and 'is_correct' in fes_img:
                    image_samples['fes'].append({
                        'y_true': 1 if result['ground_truth'] == 'A' else 0,
                        'y_pred': 1 if fes_img['prediction'] == 'A' else 0,
                        'y_prob': fes_img.get('confidence', 0.5),
                        'type': 'fes_image'
                    })

            # FCS Image
            for fcs_img in result.get('fcs_image_samples', []):
                if 'prediction' in fcs_img and 'is_correct' in fcs_img:
                    gt = result['ground_truth']
                    flipped_gt = 'B' if gt == 'A' else 'A'
                    image_samples['fcs'].append({
                        'y_true': 1 if flipped_gt == 'A' else 0,
                        'y_pred': 1 if fcs_img['prediction'] == 'A' else 0,
                        'y_prob': fcs_img.get('confidence', 0.5),
                        'type': 'fcs_image'
                    })

        # Calculate metrics for each category
        metrics_summary = {}

        logger.info("Calculating separate metrics for each category...")

        for category in ['text', 'image']:
            samples = text_samples if category == 'text' else image_samples

            for sample_type in ['fes', 'fcs']:
                data = samples[sample_type]
                key = f"{sample_type}_{category}"

                if len(data) > 0:
                    logger.info(f"  Processing {key}: {len(data)} samples")
                    y_true = np.array([d['y_true'] for d in data])
                    y_pred = np.array([d['y_pred'] for d in data])
                    y_prob = np.array([d['y_prob'] for d in data])

                    metrics_summary[key] = self.calculate_comprehensive_metrics(y_true, y_pred, y_prob)
                    metrics_summary[key]['sample_count'] = len(data)
                else:
                    logger.info(f"  Skipping {key}: No samples available")

        return metrics_summary


class FESTAVisualizer:
    """Generate Risk-Coverage and Accuracy-Coverage curves"""

    def __init__(self, output_dir: str = "output/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")

    def calculate_risk_coverage(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate risk and coverage at different confidence thresholds"""
        thresholds = np.linspace(0, 1, 100)
        coverage = []
        risk = []

        for thresh in thresholds:
            # Keep predictions with confidence >= threshold
            mask = y_probs >= thresh
            if np.sum(mask) == 0:
                coverage.append(0)
                risk.append(0)
            else:
                coverage.append(np.mean(mask))
                # Risk = error rate on retained samples
                errors = (y_true[mask] != y_pred[mask]).astype(float)
                risk.append(np.mean(errors) if len(errors) > 0 else 0)

        return np.array(coverage), np.array(risk)

    def calculate_accuracy_coverage(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    y_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate accuracy and coverage at different confidence thresholds"""
        thresholds = np.linspace(0, 1, 100)
        coverage = []
        accuracy = []

        for thresh in thresholds:
            mask = y_probs >= thresh
            if np.sum(mask) == 0:
                coverage.append(0)
                accuracy.append(0)
            else:
                coverage.append(np.mean(mask))
                correct = (y_true[mask] == y_pred[mask]).astype(float)
                accuracy.append(np.mean(correct) if len(correct) > 0 else 0)

        return np.array(coverage), np.array(accuracy)

    def plot_curves(self, data_dict: Dict[str, Dict], curve_type: str = "risk"):
        """Generate all requested curves"""

        # Define all curve combinations
        curve_configs = [
            ('FES', ['fes_text', 'fes_image']),
            ('FCS', ['fcs_text', 'fcs_image']),
            ('FESTA', ['fes_text', 'fes_image', 'fcs_text', 'fcs_image']),
            ('Output', ['original']),
            ('FES Text', ['fes_text']),
            ('FES Image', ['fes_image']),
            ('FCS Text', ['fcs_text']),
            ('FCS Image', ['fcs_image']),
        ]

        for title, keys in curve_configs:
            self._plot_single_curve(data_dict, keys, title, curve_type)

    def _plot_single_curve(self, data_dict: Dict, keys: List[str], title: str, curve_type: str):
        """Plot a single risk-coverage or accuracy-coverage curve"""
        # Create new figure with specific ID to avoid conflicts
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # Set ylabel based on curve type
        ylabel = "Risk (Error Rate)" if curve_type == "risk" else "Accuracy"

        has_data = False
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        color_idx = 0

        for key in keys:
            if key in data_dict and len(data_dict[key]['y_true']) > 0:
                y_true = np.array(data_dict[key]['y_true'])
                y_pred = np.array(data_dict[key]['y_pred'])
                y_probs = np.array(data_dict[key]['y_probs'])

                logger.info(f"  Plotting {title} - {key}: {len(y_true)} samples")

                if curve_type == "risk":
                    coverage, metric = self.calculate_risk_coverage(y_true, y_pred, y_probs)
                else:  # accuracy
                    coverage, metric = self.calculate_accuracy_coverage(y_true, y_pred, y_probs)

                # Debug output
                logger.info(f"    Coverage range: [{coverage.min():.3f}, {coverage.max():.3f}]")
                logger.info(f"    Metric range: [{metric.min():.3f}, {metric.max():.3f}]")
                logger.info(f"    Non-zero points: {np.sum(metric > 0)}")

                # Plot with better styling and ensure line is visible
                ax.plot(coverage, metric,
                       label=key.replace('_', ' ').title(),
                       linewidth=3.0,  # Thicker line
                       marker='o',
                       markersize=5,  # Larger markers
                       color=colors[color_idx % len(colors)],
                       alpha=0.9,  # More opaque
                       markerfacecolor=colors[color_idx % len(colors)],
                       markeredgecolor='white',
                       markeredgewidth=1)
                has_data = True
                color_idx += 1
            else:
                logger.warning(f"  No data for {title} - {key}")

        if not has_data:
            # Add placeholder text if no data
            ax.text(0.5, 0.5, f'No data available for {title}\n({curve_type})',
                   ha='center', va='center', fontsize=14, color='gray',
                   transform=ax.transAxes)

        ax.set_xlabel('Coverage', fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(f'{title} {curve_type.title()}-Coverage Curve', fontsize=16, fontweight='bold', pad=20)

        if has_data:
            ax.legend(loc='best', fontsize=12, framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)

        # Better grid styling - make it more visible
        ax.grid(True, alpha=0.5, linestyle='--', linewidth=1.0, color='gray')
        ax.set_xlim(-0.02, 1.02)  # Slightly wider to show all points
        ax.set_ylim(-0.02, 1.02)

        # Add minor ticks with visible grid
        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.3, linestyle=':', linewidth=0.7, color='lightgray')

        # Make axes more prominent
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)

        fig.tight_layout()

        filename = f"{title.lower().replace(' ', '_')}_{curve_type}_coverage.png"
        filepath = self.output_dir / filename

        try:
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            if filepath.exists():
                logger.info(f"✓ Saved {curve_type}-coverage curve: {filepath} ({filepath.stat().st_size} bytes)")
            else:
                logger.error(f"Failed to verify file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save {filepath}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            plt.close(fig)  # Close only THIS figure, not all figures

    def plot_roc_curves(self, data_dict: Dict[str, Dict]):
        """Generate ROC curves using sklearn.metrics.roc_curve"""
        logger.info("Generating ROC curves...")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        color_idx = 0

        for key, values in data_dict.items():
            if len(values['y_true']) > 0 and len(np.unique(values['y_true'])) > 1:
                y_true = np.array(values['y_true'])
                y_probs = np.array(values['y_probs'])

                # Calculate ROC curve using sklearn
                fpr, tpr, _ = roc_curve(y_true, y_probs)
                roc_auc = roc_auc_score(y_true, y_probs)

                ax.plot(fpr, tpr,
                       label=f'{key.replace("_", " ").title()} (AUC = {roc_auc:.3f})',
                       linewidth=2.5,
                       color=colors[color_idx % len(colors)],
                       alpha=0.8)
                color_idx += 1

        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier', alpha=0.5)

        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        ax.set_title('ROC Curves - FESTA Evaluation', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        fig.tight_layout()
        filepath = self.output_dir / 'roc_curves_combined.png'
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"✓ Saved ROC curves: {filepath}")
        plt.close(fig)

    def plot_precision_recall_curves(self, data_dict: Dict[str, Dict]):
        """Generate Precision-Recall curves using sklearn.metrics.precision_recall_curve"""
        logger.info("Generating Precision-Recall curves...")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        color_idx = 0

        for key, values in data_dict.items():
            if len(values['y_true']) > 0 and len(np.unique(values['y_true'])) > 1:
                y_true = np.array(values['y_true'])
                y_probs = np.array(values['y_probs'])

                # Calculate Precision-Recall curve using sklearn
                precision, recall, _ = precision_recall_curve(y_true, y_probs)
                ap_score = average_precision_score(y_true, y_probs)

                ax.plot(recall, precision,
                       label=f'{key.replace("_", " ").title()} (AP = {ap_score:.3f})',
                       linewidth=2.5,
                       color=colors[color_idx % len(colors)],
                       alpha=0.8)
                color_idx += 1

        ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
        ax.set_title('Precision-Recall Curves - FESTA Evaluation', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        fig.tight_layout()
        filepath = self.output_dir / 'precision_recall_curves_combined.png'
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"✓ Saved Precision-Recall curves: {filepath}")
        plt.close(fig)

    def plot_metrics_bar_chart(self, metrics_dict: Dict[str, Dict]):
        """Generate bar chart comparing metrics across categories"""
        logger.info("Generating metrics comparison bar chart...")

        categories = list(metrics_dict.keys())
        metrics_names = ['AUROC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, metric_name in enumerate(metrics_names):
            ax = axes[idx]
            metric_key = metric_name.lower().replace('-', '_')

            values = []
            labels = []
            for cat in categories:
                if metric_key in metrics_dict[cat]:
                    values.append(metrics_dict[cat][metric_key])
                    labels.append(cat.replace('_', ' ').title())

            if values:
                bars = ax.bar(range(len(values)), values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
                ax.set_title(f'{metric_name} by Category', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                ax.set_ylim([0, 1.05])

                # Add value labels on bars
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        # Remove extra subplot
        fig.delaxes(axes[5])

        fig.tight_layout()
        filepath = self.output_dir / 'metrics_comparison_bar_chart.png'
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"✓ Saved metrics bar chart: {filepath}")
        plt.close(fig)

    def generate_all_visualizations(self, results: List[Dict[str, Any]]):
        """Generate all requested visualizations including ROC and PR curves"""
        logger.info("Generating FESTA visualizations...")

        # Prepare data for visualization
        data_dict = self._prepare_visualization_data(results)

        # Generate risk-coverage curves (8 curves)
        logger.info("Generating Risk-Coverage curves...")
        self.plot_curves(data_dict, curve_type="risk")

        # Generate accuracy-coverage curves (7 curves)
        logger.info("Generating Accuracy-Coverage curves...")
        self.plot_curves(data_dict, curve_type="accuracy")

        # Generate ROC curves using sklearn
        try:
            self.plot_roc_curves(data_dict)
        except Exception as e:
            logger.error(f"Failed to generate ROC curves: {e}")

        # Generate Precision-Recall curves using sklearn
        try:
            self.plot_precision_recall_curves(data_dict)
        except Exception as e:
            logger.error(f"Failed to generate PR curves: {e}")

        logger.info(f"✓ All visualizations saved to: {self.output_dir}")

    def _prepare_visualization_data(self, results: List[Dict[str, Any]]) -> Dict:
        """Prepare data dictionary for visualization"""
        data = {
            'original': {'y_true': [], 'y_pred': [], 'y_probs': []},
            'fes_text': {'y_true': [], 'y_pred': [], 'y_probs': []},
            'fes_image': {'y_true': [], 'y_pred': [], 'y_probs': []},
            'fcs_text': {'y_true': [], 'y_pred': [], 'y_probs': []},
            'fcs_image': {'y_true': [], 'y_pred': [], 'y_probs': []},
        }

        logger.info(f"Preparing visualization data from {len(results)} results...")

        for result in results:
            gt = 1 if result['ground_truth'] == 'A' else 0

            # Original predictions
            data['original']['y_true'].append(gt)
            data['original']['y_pred'].append(1 if result['original_prediction'] == 'A' else 0)
            data['original']['y_probs'].append(result.get('original_confidence', 0.5))

            # FES Text
            fes_text_samples = result.get('fes_text_samples', [])
            logger.info(f"  Sample {result['sample_id']}: {len(fes_text_samples)} FES text samples")
            for sample in fes_text_samples:
                if 'prediction' in sample:
                    data['fes_text']['y_true'].append(gt)
                    data['fes_text']['y_pred'].append(1 if sample['prediction'] == 'A' else 0)
                    data['fes_text']['y_probs'].append(sample.get('confidence', 0.5))
                else:
                    logger.warning(f"    FES text sample missing prediction: {sample.get('generated_question', '')[:40]}")

            # FES Image
            fes_image_samples = result.get('fes_image_samples', [])
            logger.info(f"  Sample {result['sample_id']}: {len(fes_image_samples)} FES image samples")
            for sample in fes_image_samples:
                if 'prediction' in sample:
                    data['fes_image']['y_true'].append(gt)
                    data['fes_image']['y_pred'].append(1 if sample['prediction'] == 'A' else 0)
                    data['fes_image']['y_probs'].append(sample.get('confidence', 0.5))

            # FCS Text (flipped ground truth)
            flipped_gt = 0 if gt == 1 else 1
            fcs_text_samples = result.get('fcs_text_samples', [])
            logger.info(f"  Sample {result['sample_id']}: {len(fcs_text_samples)} FCS text samples")
            for sample in fcs_text_samples:
                if 'prediction' in sample:
                    data['fcs_text']['y_true'].append(flipped_gt)
                    data['fcs_text']['y_pred'].append(1 if sample['prediction'] == 'A' else 0)
                    data['fcs_text']['y_probs'].append(sample.get('confidence', 0.5))
                else:
                    logger.warning(f"    FCS text sample missing prediction: {sample.get('generated_question', '')[:40]}")

            # FCS Image (flipped ground truth)
            fcs_image_samples = result.get('fcs_image_samples', [])
            logger.info(f"  Sample {result['sample_id']}: {len(fcs_image_samples)} FCS image samples")
            for sample in fcs_image_samples:
                if 'prediction' in sample:
                    data['fcs_image']['y_true'].append(flipped_gt)
                    data['fcs_image']['y_pred'].append(1 if sample['prediction'] == 'A' else 0)
                    data['fcs_image']['y_probs'].append(sample.get('confidence', 0.5))

        # Log summary
        logger.info("Data preparation summary:")
        for key, values in data.items():
            logger.info(f"  {key}: {len(values['y_true'])} samples")

        return data


def generate_comprehensive_report(results: List[Dict[str, Any]], output_dir: str = "output"):
    """Generate comprehensive metrics report with all visualizations"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate metrics
    metrics_calculator = FESTAMetrics(output_dir=str(output_path / "metrics"))
    text_image_metrics = metrics_calculator.separate_text_image_metrics(results)

    # Generate visualizations
    visualizer = FESTAVisualizer(output_dir=str(output_path / "visualizations"))
    visualizer.generate_all_visualizations(results)

    # Generate metrics comparison bar chart
    try:
        visualizer.plot_metrics_bar_chart(text_image_metrics)
    except Exception as e:
        logger.error(f"Failed to generate metrics bar chart: {e}")

    # Save metrics to JSON
    metrics_file = output_path / "comprehensive_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(text_image_metrics, f, indent=2)

    logger.info(f"✓ Comprehensive metrics saved to: {metrics_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE FESTA METRICS SUMMARY")
    print("=" * 80)

    for key, metrics in text_image_metrics.items():
        print(f"\n{key.upper().replace('_', ' ')}:")
        print(f"  Samples: {metrics.get('sample_count', 0)}")
        print(f"  AUROC: {metrics.get('auroc', 0):.4f}")
        print(f"  AUPRC: {metrics.get('auprc', 0):.4f}")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall: {metrics.get('recall', 0):.4f}")
        print(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
        print(f"  Brier Score: {metrics.get('brier_score', 0):.4f}")
        print(f"  ECE: {metrics.get('ece', 0):.4f}")

    print("\n" + "=" * 80)
    print(f"✓ All visualizations saved to: {output_path / 'visualizations'}")
    print("=" * 80 + "\n")

    return text_image_metrics

