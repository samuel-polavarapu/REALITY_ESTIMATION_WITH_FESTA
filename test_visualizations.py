#!/usr/bin/env python3
"""
Test script to debug visualization generation
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load metrics
with open('/data/sam/Kaggle/code/LLAVA-V5-2/output/api_run/comprehensive_metrics.json', 'r') as f:
    metrics = json.load(f)

print("Metrics loaded:")
print(json.dumps(metrics, indent=2))

# Create test data
np.random.seed(42)

# Simulate FES text data (high accuracy)
fes_y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1])
fes_y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1])
fes_y_probs = np.array([0.95, 0.98, 0.96, 0.99, 0.97, 0.98, 0.95, 0.99])

# Simulate FCS text data (moderate accuracy)
fcs_y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0])
fcs_y_pred = np.array([0, 0, 0, 1, 1, 0, 1, 0])
fcs_y_probs = np.array([0.85, 0.90, 0.88, 0.92, 0.95, 0.87, 0.91, 0.89])

def calculate_risk_coverage(y_true, y_pred, y_probs):
    """Calculate risk-coverage curve"""
    thresholds = np.linspace(0, 1, 100)
    coverage = []
    risk = []

    for thresh in thresholds:
        mask = y_probs >= thresh
        if np.sum(mask) == 0:
            coverage.append(0)
            risk.append(0)
        else:
            coverage.append(np.mean(mask))
            errors = (y_true[mask] != y_pred[mask]).astype(float)
            risk.append(np.mean(errors) if len(errors) > 0 else 0)

    return np.array(coverage), np.array(risk)

def calculate_accuracy_coverage(y_true, y_pred, y_probs):
    """Calculate accuracy-coverage curve"""
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

# Test plotting
output_dir = Path('/data/sam/Kaggle/code/LLAVA-V5-2/output/api_run/visualizations_test')
output_dir.mkdir(exist_ok=True)

print(f"\nGenerating test visualizations in {output_dir}")

# Generate FES risk-coverage
fig, ax = plt.subplots(figsize=(10, 6))
coverage, risk = calculate_risk_coverage(fes_y_true, fes_y_pred, fes_y_probs)
print(f"\nFES Risk-Coverage data:")
print(f"  Coverage range: [{coverage.min():.3f}, {coverage.max():.3f}]")
print(f"  Risk range: [{risk.min():.3f}, {risk.max():.3f}]")
print(f"  Non-zero risk points: {np.sum(risk > 0)}")

ax.plot(coverage, risk, 'b-', linewidth=2.5, marker='o', markersize=4, label='FES Text')
ax.set_xlabel('Coverage', fontsize=13, fontweight='bold')
ax.set_ylabel('Risk (Error Rate)', fontsize=13, fontweight='bold')
ax.set_title('FES Risk-Coverage Curve (TEST)', fontsize=15, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.4, linestyle='--')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(output_dir / 'test_fes_risk.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ Saved: test_fes_risk.png")

# Generate FCS risk-coverage
fig, ax = plt.subplots(figsize=(10, 6))
coverage, risk = calculate_risk_coverage(fcs_y_true, fcs_y_pred, fcs_y_probs)
print(f"\nFCS Risk-Coverage data:")
print(f"  Coverage range: [{coverage.min():.3f}, {coverage.max():.3f}]")
print(f"  Risk range: [{risk.min():.3f}, {risk.max():.3f}]")
print(f"  Non-zero risk points: {np.sum(risk > 0)}")

ax.plot(coverage, risk, 'r-', linewidth=2.5, marker='s', markersize=4, label='FCS Text')
ax.set_xlabel('Coverage', fontsize=13, fontweight='bold')
ax.set_ylabel('Risk (Error Rate)', fontsize=13, fontweight='bold')
ax.set_title('FCS Risk-Coverage Curve (TEST)', fontsize=15, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.4, linestyle='--')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(output_dir / 'test_fcs_risk.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ Saved: test_fcs_risk.png")

# Generate FES accuracy-coverage
fig, ax = plt.subplots(figsize=(10, 6))
coverage, accuracy = calculate_accuracy_coverage(fes_y_true, fes_y_pred, fes_y_probs)
print(f"\nFES Accuracy-Coverage data:")
print(f"  Coverage range: [{coverage.min():.3f}, {coverage.max():.3f}]")
print(f"  Accuracy range: [{accuracy.min():.3f}, {accuracy.max():.3f}]")
print(f"  Perfect accuracy points: {np.sum(accuracy == 1.0)}")

ax.plot(coverage, accuracy, 'g-', linewidth=2.5, marker='o', markersize=4, label='FES Text')
ax.set_xlabel('Coverage', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax.set_title('FES Accuracy-Coverage Curve (TEST)', fontsize=15, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.4, linestyle='--')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(output_dir / 'test_fes_accuracy.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ Saved: test_fes_accuracy.png")

print(f"\n✓ Test visualizations generated successfully!")
print(f"  Check: {output_dir}")

