#!/usr/bin/env python3
"""
FESTA Calibration and Risk-Coverage Analysis Module
Uses reference implementation with LogisticRegression calibration
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from typing import Tuple, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def calibrate_confidence(raw_scores, labels):
    """Calibrate confidence scores using Logistic Regression.

    Args:
        raw_scores: Raw confidence scores from model
        labels: Ground truth labels

    Returns:
        Calibrated probabilities
    """
    lr = LogisticRegression()
    raw_scores = np.array(raw_scores).reshape(-1, 1)
    labels = np.array(labels)
    lr.fit(raw_scores, labels)
    calibrated = lr.predict_proba(raw_scores)[:, 1]
    return calibrated


def risk_coverage_peel_off_low(conf, correct):
    """Calculate risk-coverage curve by peeling off low-confidence predictions.

    Args:
        conf: Confidence scores
        correct: Binary correctness indicators

    Returns:
        coverage, risk, accuracy, aurc (Area Under Risk-Coverage curve)
    """
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, int)
    N = len(conf)
    order_low_first = np.argsort(conf, kind="mergesort")
    corr_low_first = correct[order_low_first]
    cum_correct_low = np.concatenate(([0], np.cumsum(corr_low_first)))
    total_correct = correct.sum()
    k = np.arange(N + 1)
    kept = N - k
    correct_kept = total_correct - cum_correct_low
    with np.errstate(invalid="ignore", divide="ignore"):
        accuracy = correct_kept / np.maximum(kept, 1)
        risk = 1.0 - accuracy
        coverage = kept / N
    mask = kept > 0
    aurc = np.trapz(risk[mask][::-1], coverage[mask][::-1])
    return coverage, risk, accuracy, aurc


def abstention_table(confidence, correct, step=0.1):
    """Print abstention analysis table.

    Args:
        confidence: Confidence scores
        correct: Binary correctness indicators
        step: Abstention fraction step size
    """
    N = len(confidence)
    order = np.argsort(-confidence, kind="mergesort")
    corr_sorted = correct[order]
    print(f"{'Abstain':<10} {'Coverage':<10} {'Risk':<10} {'Accuracy':<10}")
    print("-" * 49)
    for abstain_frac in np.arange(0.0, 1.01, step):
        k = int(round((1 - abstain_frac) * N))
        if k <= 0:
            cov, rk, acc = 0.0, 0.0, 0.0
        else:
            cov = k / N
            acc = corr_sorted[:k].mean()
            rk = 1.0 - acc
        print(f"{abstain_frac:>9.0%} {cov:<10.3f} {rk:<10.3f} {acc:<10.3f}")


def plot_risk_coverage(coverage, risk, title="Risk-Coverage Curve", output_path=None, label=None):
    """Plot risk-coverage curve.

    Args:
        coverage: Coverage values
        risk: Risk values
        title: Plot title
        output_path: Path to save plot (optional)
        label: Legend label (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(coverage, risk, label=label or 'Risk', linewidth=2.5, marker='o', markersize=4)
    plt.xlabel('Coverage', fontsize=13, fontweight='bold')
    plt.ylabel('Risk (Error Rate)', fontsize=13, fontweight='bold')
    plt.title(title, fontsize=15, fontweight='bold')
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"✓ Saved risk-coverage plot: {output_path}")
        plt.close()
    else:
        plt.show()


def plot_accuracy_coverage(coverage, accuracy, title="Accuracy-Coverage Curve", output_path=None, label=None):
    """Plot accuracy-coverage curve.

    Args:
        coverage: Coverage values
        accuracy: Accuracy values
        title: Plot title
        output_path: Path to save plot (optional)
        label: Legend label (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(coverage, accuracy, label=label or 'Accuracy', linewidth=2.5, marker='o', markersize=4)
    plt.xlabel('Coverage', fontsize=13, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=13, fontweight='bold')
    plt.title(title, fontsize=15, fontweight='bold')
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"✓ Saved accuracy-coverage plot: {output_path}")
        plt.close()
    else:
        plt.show()


def plot_multiple_risk_coverage(data_dict, title="Risk-Coverage Curves", output_path=None):
    """Plot multiple risk-coverage curves on same plot.

    Args:
        data_dict: Dictionary with keys as labels and values as (coverage, risk) tuples
        title: Plot title
        output_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx, (label, (coverage, risk)) in enumerate(data_dict.items()):
        plt.plot(coverage, risk,
                label=label,
                linewidth=2.5,
                marker='o',
                markersize=4,
                color=colors[idx % len(colors)])

    plt.xlabel('Coverage', fontsize=13, fontweight='bold')
    plt.ylabel('Risk (Error Rate)', fontsize=13, fontweight='bold')
    plt.title(title, fontsize=15, fontweight='bold')
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"✓ Saved multi risk-coverage plot: {output_path}")
        plt.close()
    else:
        plt.show()


def plot_multiple_accuracy_coverage(data_dict, title="Accuracy-Coverage Curves", output_path=None):
    """Plot multiple accuracy-coverage curves on same plot.

    Args:
        data_dict: Dictionary with keys as labels and values as (coverage, accuracy) tuples
        title: Plot title
        output_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx, (label, (coverage, accuracy)) in enumerate(data_dict.items()):
        plt.plot(coverage, accuracy,
                label=label,
                linewidth=2.5,
                marker='o',
                markersize=4,
                color=colors[idx % len(colors)])

    plt.xlabel('Coverage', fontsize=13, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=13, fontweight='bold')
    plt.title(title, fontsize=15, fontweight='bold')
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"✓ Saved multi accuracy-coverage plot: {output_path}")
        plt.close()
    else:
        plt.show()


def generate_calibrated_metrics(confidence, correct, predictions, ground_truth, output_dir=None):
    """Generate comprehensive calibrated metrics and plots.

    Args:
        confidence: Array of confidence scores
        correct: Array of binary correctness indicators
        predictions: Array of predictions
        ground_truth: Array of ground truth labels
        output_dir: Directory to save outputs

    Returns:
        Dictionary with metrics and curve data
    """
    output_path = Path(output_dir) if output_dir else Path('output/calibration')
    output_path.mkdir(parents=True, exist_ok=True)

    confidence = np.array(confidence)
    correct = np.array(correct)

    # Calculate risk-coverage curve
    coverage, risk, accuracy, aurc = risk_coverage_peel_off_low(confidence, correct)

    # Calculate AUROC if possible
    auroc = None
    if len(np.unique(ground_truth)) > 1:
        try:
            auroc = roc_auc_score(ground_truth, confidence)
        except Exception as e:
            logger.warning(f"Could not calculate AUROC: {e}")

    # Calibrate confidence scores
    calibrated_conf = None
    try:
        if len(np.unique(correct)) > 1:  # Need both classes for calibration
            calibrated_conf = calibrate_confidence(confidence, correct)
            coverage_cal, risk_cal, accuracy_cal, aurc_cal = risk_coverage_peel_off_low(calibrated_conf, correct)
        else:
            calibrated_conf = confidence
            coverage_cal, risk_cal, accuracy_cal, aurc_cal = coverage, risk, accuracy, aurc
    except Exception as e:
        logger.warning(f"Calibration failed: {e}")
        calibrated_conf = confidence
        coverage_cal, risk_cal, accuracy_cal, aurc_cal = coverage, risk, accuracy, aurc

    # Generate plots
    plot_risk_coverage(coverage, risk,
                      title="Risk-Coverage Curve (Uncalibrated)",
                      output_path=output_path / "risk_coverage_uncalibrated.png",
                      label="Uncalibrated")

    plot_accuracy_coverage(coverage, accuracy,
                          title="Accuracy-Coverage Curve (Uncalibrated)",
                          output_path=output_path / "accuracy_coverage_uncalibrated.png",
                          label="Uncalibrated")

    if calibrated_conf is not None:
        plot_risk_coverage(coverage_cal, risk_cal,
                          title="Risk-Coverage Curve (Calibrated)",
                          output_path=output_path / "risk_coverage_calibrated.png",
                          label="Calibrated")

        plot_accuracy_coverage(coverage_cal, accuracy_cal,
                              title="Accuracy-Coverage Curve (Calibrated)",
                              output_path=output_path / "accuracy_coverage_calibrated.png",
                              label="Calibrated")

    # Print abstention table
    logger.info("\nAbstention Analysis:")
    abstention_table(confidence, correct, step=0.1)

    return {
        'aurc': aurc,
        'aurc_calibrated': aurc_cal if calibrated_conf is not None else None,
        'auroc': auroc,
        'coverage': coverage.tolist(),
        'risk': risk.tolist(),
        'accuracy': accuracy.tolist(),
        'coverage_calibrated': coverage_cal.tolist() if calibrated_conf is not None else None,
        'risk_calibrated': risk_cal.tolist() if calibrated_conf is not None else None,
        'accuracy_calibrated': accuracy_cal.tolist() if calibrated_conf is not None else None,
    }

