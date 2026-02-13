"""Tests for snoopy.calibration.metrics module."""

from __future__ import annotations

from snoopy.calibration.metrics import (
    CalibrationReport,
    ClassificationMetrics,
    ConfusionMatrix,
    MethodMetrics,
    compute_confusion_matrix,
    compute_method_metrics,
    compute_metrics_at_threshold,
    compute_roc_curve,
    export_metrics_csv,
)


class TestConfusionMatrix:
    def test_perfect_predictions(self):
        preds = [True, True, False, False]
        labels = [True, True, False, False]
        cm = compute_confusion_matrix(preds, labels)
        assert cm.true_positives == 2
        assert cm.true_negatives == 2
        assert cm.false_positives == 0
        assert cm.false_negatives == 0

    def test_all_wrong(self):
        preds = [True, True, False, False]
        labels = [False, False, True, True]
        cm = compute_confusion_matrix(preds, labels)
        assert cm.true_positives == 0
        assert cm.true_negatives == 0
        assert cm.false_positives == 2
        assert cm.false_negatives == 2

    def test_empty(self):
        cm = compute_confusion_matrix([], [])
        assert cm.true_positives == 0
        assert cm.true_negatives == 0


class TestMetricsAtThreshold:
    def test_perfect_scores(self):
        confidences = [0.9, 0.8, 0.1, 0.2]
        labels = [True, True, False, False]
        metrics = compute_metrics_at_threshold(confidences, labels, threshold=0.5)
        assert metrics.sensitivity == 1.0
        assert metrics.specificity == 1.0
        assert metrics.precision == 1.0
        assert metrics.f1 == 1.0

    def test_threshold_zero_all_positive(self):
        confidences = [0.5, 0.5, 0.5]
        labels = [True, False, False]
        metrics = compute_metrics_at_threshold(confidences, labels, threshold=0.0)
        assert metrics.sensitivity == 1.0
        # All predicted positive, so FP=2, TN=0
        assert metrics.specificity == 0.0

    def test_threshold_one_all_negative(self):
        confidences = [0.5, 0.5]
        labels = [True, False]
        metrics = compute_metrics_at_threshold(confidences, labels, threshold=1.1)
        assert metrics.sensitivity == 0.0
        assert metrics.specificity == 1.0


class TestROCCurve:
    def test_roc_returns_points(self):
        confidences = [0.9, 0.8, 0.3, 0.1]
        labels = [True, True, False, False]
        points, auc = compute_roc_curve(confidences, labels, n_thresholds=10)
        assert len(points) == 11  # n_thresholds + 1
        assert 0.0 <= auc <= 1.0

    def test_perfect_auc(self):
        confidences = [1.0, 1.0, 0.0, 0.0]
        labels = [True, True, False, False]
        _, auc = compute_roc_curve(confidences, labels, n_thresholds=100)
        assert auc > 0.9  # Should be near-perfect


class TestMethodMetrics:
    def test_compute_method_metrics(self):
        confidences = [0.9, 0.8, 0.3, 0.1]
        labels = [True, True, False, False]
        result = compute_method_metrics("test_method", confidences, labels)
        assert result.method_name == "test_method"
        assert len(result.classification_metrics) == 6  # default thresholds
        assert len(result.roc_curve) > 0
        assert 0.0 <= result.auc <= 1.0

    def test_custom_thresholds(self):
        confidences = [0.9, 0.1]
        labels = [True, False]
        result = compute_method_metrics("test", confidences, labels, severity_thresholds=[0.5])
        assert len(result.classification_metrics) == 1


class TestExportCSV:
    def test_export_creates_file(self, tmp_path):
        method = MethodMetrics(
            method_name="test",
            classification_metrics=[
                ClassificationMetrics(
                    threshold=0.5,
                    sensitivity=0.8,
                    specificity=0.9,
                    precision=0.85,
                    recall=0.8,
                    f1=0.82,
                    confusion=ConfusionMatrix(
                        true_positives=8,
                        true_negatives=9,
                        false_positives=1,
                        false_negatives=2,
                    ),
                )
            ],
            roc_curve=[],
            auc=0.9,
        )
        report = CalibrationReport(
            per_method=[method],
            total_samples=20,
            total_positive=10,
            total_negative=10,
        )
        path = export_metrics_csv(report, tmp_path / "metrics.csv")
        assert path.exists()
        content = path.read_text()
        assert "test" in content
        assert "0.50" in content
