"""Metrics computation for calibration and validation.

Computes standard classification metrics (sensitivity, specificity, precision,
recall, F1) at each severity threshold, plus ROC curve data for confidence
thresholds.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConfusionMatrix:
    """Standard binary confusion matrix."""

    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class ClassificationMetrics:
    """Classification metrics at a given threshold."""

    threshold: float
    sensitivity: float  # True positive rate (recall)
    specificity: float  # True negative rate (1 - FPR)
    precision: float
    recall: float
    f1: float
    confusion: ConfusionMatrix


@dataclass
class ROCPoint:
    """A single point on an ROC curve."""

    threshold: float
    tpr: float  # True positive rate
    fpr: float  # False positive rate


@dataclass
class MethodMetrics:
    """Metrics for a specific analysis method."""

    method_name: str
    classification_metrics: list[ClassificationMetrics] = field(default_factory=list)
    roc_curve: list[ROCPoint] = field(default_factory=list)
    auc: float = 0.0


@dataclass
class CalibrationReport:
    """Full calibration report across all methods."""

    per_method: list[MethodMetrics] = field(default_factory=list)
    overall: MethodMetrics | None = None
    total_samples: int = 0
    total_positive: int = 0
    total_negative: int = 0


def compute_confusion_matrix(
    predictions: list[bool],
    labels: list[bool],
) -> ConfusionMatrix:
    """Compute confusion matrix from predictions and ground truth labels."""
    cm = ConfusionMatrix()
    for pred, label in zip(predictions, labels):
        if label and pred:
            cm.true_positives += 1
        elif not label and not pred:
            cm.true_negatives += 1
        elif not label and pred:
            cm.false_positives += 1
        elif label and not pred:
            cm.false_negatives += 1
    return cm


def compute_metrics_at_threshold(
    confidences: list[float],
    labels: list[bool],
    threshold: float,
) -> ClassificationMetrics:
    """Compute classification metrics at a given confidence threshold.

    Args:
        confidences: Predicted confidence scores (0.0 to 1.0).
        labels: Ground truth labels (True = manipulated).
        threshold: Confidence threshold for positive prediction.

    Returns:
        ClassificationMetrics at the specified threshold.
    """
    predictions = [c >= threshold for c in confidences]
    cm = compute_confusion_matrix(predictions, labels)

    sensitivity = cm.true_positives / max(cm.true_positives + cm.false_negatives, 1)
    specificity = cm.true_negatives / max(cm.true_negatives + cm.false_positives, 1)
    precision = cm.true_positives / max(cm.true_positives + cm.false_positives, 1)
    recall = sensitivity
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return ClassificationMetrics(
        threshold=threshold,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion=cm,
    )


def compute_roc_curve(
    confidences: list[float],
    labels: list[bool],
    n_thresholds: int = 100,
) -> tuple[list[ROCPoint], float]:
    """Compute ROC curve and AUC.

    Args:
        confidences: Predicted confidence scores.
        labels: Ground truth labels.
        n_thresholds: Number of threshold points to evaluate.

    Returns:
        Tuple of (ROC points, AUC).
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds + 1)
    roc_points = []

    for threshold in thresholds:
        metrics = compute_metrics_at_threshold(confidences, labels, threshold)
        tpr = metrics.sensitivity
        fpr = 1.0 - metrics.specificity
        roc_points.append(ROCPoint(threshold=threshold, tpr=tpr, fpr=fpr))

    # Compute AUC using trapezoidal rule
    # Sort by FPR ascending
    sorted_points = sorted(roc_points, key=lambda p: p.fpr)
    auc = 0.0
    for i in range(1, len(sorted_points)):
        dx = sorted_points[i].fpr - sorted_points[i - 1].fpr
        avg_y = (sorted_points[i].tpr + sorted_points[i - 1].tpr) / 2.0
        auc += dx * avg_y

    return roc_points, auc


def compute_method_metrics(
    method_name: str,
    confidences: list[float],
    labels: list[bool],
    severity_thresholds: list[float] | None = None,
) -> MethodMetrics:
    """Compute full metrics for a single analysis method.

    Args:
        method_name: Name of the analysis method.
        confidences: Predicted confidence scores.
        labels: Ground truth labels.
        severity_thresholds: Thresholds to evaluate. Defaults to
            [0.3, 0.5, 0.6, 0.7, 0.8, 0.9].

    Returns:
        MethodMetrics with classification metrics and ROC curve.
    """
    if severity_thresholds is None:
        severity_thresholds = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]

    classification_metrics = [
        compute_metrics_at_threshold(confidences, labels, t) for t in severity_thresholds
    ]

    roc_curve, auc = compute_roc_curve(confidences, labels)

    return MethodMetrics(
        method_name=method_name,
        classification_metrics=classification_metrics,
        roc_curve=roc_curve,
        auc=auc,
    )


def export_metrics_csv(
    report: CalibrationReport,
    output_path: str | Path,
) -> Path:
    """Export calibration metrics to CSV.

    Args:
        report: CalibrationReport to export.
        output_path: Path for the output CSV file.

    Returns:
        Path to the written CSV file.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "threshold",
                "sensitivity",
                "specificity",
                "precision",
                "recall",
                "f1",
                "auc",
                "tp",
                "tn",
                "fp",
                "fn",
            ]
        )

        for method_metrics in report.per_method:
            for cm in method_metrics.classification_metrics:
                writer.writerow(
                    [
                        method_metrics.method_name,
                        f"{cm.threshold:.2f}",
                        f"{cm.sensitivity:.4f}",
                        f"{cm.specificity:.4f}",
                        f"{cm.precision:.4f}",
                        f"{cm.recall:.4f}",
                        f"{cm.f1:.4f}",
                        f"{method_metrics.auc:.4f}",
                        cm.confusion.true_positives,
                        cm.confusion.true_negatives,
                        cm.confusion.false_positives,
                        cm.confusion.false_negatives,
                    ]
                )

        if report.overall:
            for cm in report.overall.classification_metrics:
                writer.writerow(
                    [
                        "overall",
                        f"{cm.threshold:.2f}",
                        f"{cm.sensitivity:.4f}",
                        f"{cm.specificity:.4f}",
                        f"{cm.precision:.4f}",
                        f"{cm.recall:.4f}",
                        f"{cm.f1:.4f}",
                        f"{report.overall.auc:.4f}",
                        cm.confusion.true_positives,
                        cm.confusion.true_negatives,
                        cm.confusion.false_positives,
                        cm.confusion.false_negatives,
                    ]
                )

    logger.info("Metrics CSV written to %s", output)
    return output
