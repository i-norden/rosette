"""Tests for rosette.calibration.benchmark module."""

from __future__ import annotations

from rosette.calibration.benchmark import (
    LabeledSample,
    auto_tune_thresholds,
    load_labeled_samples,
)
from rosette.calibration.metrics import (
    CalibrationReport,
    ClassificationMetrics,
    ConfusionMatrix,
    MethodMetrics,
)


class TestLabeledSample:
    def test_creation(self):
        sample = LabeledSample(path="/tmp/test.png", label=True, category="synthetic")
        assert sample.label is True
        assert sample.category == "synthetic"
        assert sample.expected_methods == []


class TestLoadLabeledSamples:
    def test_empty_dir(self, tmp_path):
        samples = load_labeled_samples(tmp_path)
        assert samples == []

    def test_clean_dir(self, tmp_path):
        clean_dir = tmp_path / "clean"
        clean_dir.mkdir()
        (clean_dir / "image1.png").write_bytes(b"fake png")
        (clean_dir / "image2.jpg").write_bytes(b"fake jpg")
        (clean_dir / "readme.txt").write_text("not an image")

        samples = load_labeled_samples(tmp_path)
        assert len(samples) == 2
        assert all(not s.label for s in samples)
        assert all(s.category == "clean" for s in samples)

    def test_synthetic_dir(self, tmp_path):
        synthetic_dir = tmp_path / "synthetic"
        synthetic_dir.mkdir()
        (synthetic_dir / "forged.png").write_bytes(b"fake")

        samples = load_labeled_samples(tmp_path)
        assert len(samples) == 1
        assert samples[0].label is True
        assert samples[0].category == "synthetic"

    def test_labels_file(self, tmp_path):
        import json

        labels = [
            {"path": "test.png", "label": True, "category": "custom"},
            {"path": "clean.png", "label": False, "category": "control"},
        ]
        labels_file = tmp_path / "labels.json"
        labels_file.write_text(json.dumps(labels))

        samples = load_labeled_samples(tmp_path, labels_file=labels_file)
        assert len(samples) == 2
        assert samples[0].label is True
        assert samples[1].label is False


class TestAutoTuneThresholds:
    def test_finds_threshold(self):
        method = MethodMetrics(
            method_name="test_method",
            classification_metrics=[
                ClassificationMetrics(
                    threshold=0.3,
                    sensitivity=0.9,
                    specificity=0.8,
                    precision=0.7,
                    recall=0.9,
                    f1=0.78,
                    confusion=ConfusionMatrix(),
                ),
                ClassificationMetrics(
                    threshold=0.7,
                    sensitivity=0.7,
                    specificity=0.96,
                    precision=0.9,
                    recall=0.7,
                    f1=0.79,
                    confusion=ConfusionMatrix(),
                ),
            ],
            roc_curve=[],
            auc=0.9,
        )
        report = CalibrationReport(per_method=[method], total_samples=100)
        recommended = auto_tune_thresholds(report, target_specificity=0.95)
        assert "test_method" in recommended
        assert recommended["test_method"] == 0.7  # Only one meets specificity >= 0.95

    def test_default_when_no_threshold_meets_target(self):
        method = MethodMetrics(
            method_name="low_spec",
            classification_metrics=[
                ClassificationMetrics(
                    threshold=0.5,
                    sensitivity=0.9,
                    specificity=0.5,
                    precision=0.5,
                    recall=0.9,
                    f1=0.64,
                    confusion=ConfusionMatrix(),
                ),
            ],
            roc_curve=[],
            auc=0.6,
        )
        report = CalibrationReport(per_method=[method], total_samples=50)
        recommended = auto_tune_thresholds(report, target_specificity=0.95)
        assert recommended["low_spec"] == 0.5  # Falls back to default
