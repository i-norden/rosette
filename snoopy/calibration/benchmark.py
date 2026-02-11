"""Benchmark calibration runner.

Runs the full analysis pipeline against labeled benchmark datasets (demo
fixtures or custom labeled data) and computes per-method and overall
classification metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from snoopy.analysis.evidence import aggregate_findings
from snoopy.analysis.run_analysis import run_image_forensics, run_intra_paper_cross_ref
from snoopy.calibration.metrics import (
    CalibrationReport,
    MethodMetrics,
    compute_method_metrics,
    export_metrics_csv,
)
from snoopy.config import AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class LabeledSample:
    """A labeled benchmark sample."""

    path: str
    label: bool  # True = manipulated/fraudulent
    category: str  # e.g. "rsiil_forgery", "clean", "synthetic"
    expected_methods: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Result from analyzing a single benchmark sample."""

    sample: LabeledSample
    findings: list[dict]
    detected: bool
    max_confidence: float
    methods_fired: list[str]
    risk_level: str


def load_labeled_samples(
    fixtures_dir: str | Path,
    labels_file: str | Path | None = None,
) -> list[LabeledSample]:
    """Load labeled samples from a fixtures directory.

    If a labels_file (JSON) is provided, it should contain a list of objects
    with 'path', 'label', and 'category' keys. Otherwise, labels are inferred
    from directory names:
    - synthetic/ -> manipulated (True)
    - rsiil/ -> depends on filename convention
    - clean/ -> not manipulated (False)
    - retracted/ -> manipulated (True)

    Args:
        fixtures_dir: Root directory containing fixture subdirectories.
        labels_file: Optional path to a JSON labels file.

    Returns:
        List of LabeledSample objects.
    """
    fixtures = Path(fixtures_dir)
    samples: list[LabeledSample] = []

    if labels_file:
        labels_path = Path(labels_file)
        if labels_path.exists():
            with open(labels_path) as f:
                raw = json.load(f)
            for entry in raw:
                samples.append(LabeledSample(
                    path=str(fixtures / entry["path"]),
                    label=entry["label"],
                    category=entry.get("category", "unknown"),
                    expected_methods=entry.get("expected_methods", []),
                ))
            return samples

    # Infer labels from directory structure
    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    # Synthetic forgeries — expected to be detected
    synthetic_dir = fixtures / "synthetic"
    if synthetic_dir.exists():
        for img in sorted(synthetic_dir.iterdir()):
            if img.suffix.lower() in image_extensions:
                samples.append(LabeledSample(
                    path=str(img),
                    label=True,
                    category="synthetic",
                ))

    # RSIIL benchmark
    rsiil_dir = fixtures / "rsiil"
    if rsiil_dir.exists():
        for img in sorted(rsiil_dir.iterdir()):
            if img.suffix.lower() in image_extensions:
                # Convention: filenames with "pristine" or "authentic" are clean
                is_clean = any(
                    kw in img.stem.lower()
                    for kw in ("pristine", "authentic", "original", "clean")
                )
                samples.append(LabeledSample(
                    path=str(img),
                    label=not is_clean,
                    category="rsiil_clean" if is_clean else "rsiil_forgery",
                ))

    # Clean controls — should NOT be detected
    clean_dir = fixtures / "clean"
    if clean_dir.exists():
        for img in sorted(clean_dir.iterdir()):
            if img.suffix.lower() in image_extensions:
                samples.append(LabeledSample(
                    path=str(img),
                    label=False,
                    category="clean",
                ))

    return samples


def run_benchmark(
    samples: list[LabeledSample],
    config: AnalysisConfig | None = None,
) -> tuple[list[BenchmarkResult], CalibrationReport]:
    """Run the analysis pipeline on all labeled samples and compute metrics.

    Args:
        samples: Labeled benchmark samples.
        config: Analysis configuration. Uses defaults if None.

    Returns:
        Tuple of (per-sample results, calibration report).
    """
    if config is None:
        config = AnalysisConfig()

    results: list[BenchmarkResult] = []

    # Per-method data for metrics computation
    method_data: dict[str, tuple[list[float], list[bool]]] = {}
    overall_confidences: list[float] = []
    overall_labels: list[bool] = []

    for sample in samples:
        if not Path(sample.path).exists():
            logger.warning("Sample not found: %s", sample.path)
            continue

        # Run forensics
        findings = run_image_forensics(
            sample.path,
            figure_id=Path(sample.path).name,
            config=config,
        )

        # Collect results
        max_confidence = max((f.get("confidence", 0.0) for f in findings), default=0.0)
        methods_fired = list({f.get("method") or f.get("analysis_type", "") for f in findings})
        detected = len(findings) > 0

        evidence = aggregate_findings(findings)
        risk_level = evidence.paper_risk

        results.append(BenchmarkResult(
            sample=sample,
            findings=findings,
            detected=detected,
            max_confidence=max_confidence,
            methods_fired=methods_fired,
            risk_level=risk_level,
        ))

        # Collect per-method data
        for finding in findings:
            method = finding.get("method") or finding.get("analysis_type", "")
            conf = finding.get("confidence", 0.0)
            if method:
                if method not in method_data:
                    method_data[method] = ([], [])
                method_data[method][0].append(conf)
                method_data[method][1].append(sample.label)

        # Overall data
        overall_confidences.append(max_confidence)
        overall_labels.append(sample.label)

    # Compute per-method metrics
    per_method = []
    for method_name, (confs, labels) in method_data.items():
        if confs:
            metrics = compute_method_metrics(method_name, confs, labels)
            per_method.append(metrics)

    # Compute overall metrics
    overall = None
    if overall_confidences:
        overall = compute_method_metrics("overall", overall_confidences, overall_labels)

    total_positive = sum(1 for s in samples if s.label)
    total_negative = sum(1 for s in samples if not s.label)

    report = CalibrationReport(
        per_method=per_method,
        overall=overall,
        total_samples=len(samples),
        total_positive=total_positive,
        total_negative=total_negative,
    )

    return results, report


def auto_tune_thresholds(
    report: CalibrationReport,
    target_specificity: float = 0.95,
) -> dict[str, float]:
    """Auto-tune confidence thresholds to achieve target specificity.

    For each method, finds the lowest confidence threshold that achieves
    at least the target specificity (1 - false positive rate).

    Args:
        report: CalibrationReport from a benchmark run.
        target_specificity: Target specificity (default 0.95).

    Returns:
        Dict mapping method name to recommended threshold.
    """
    recommended: dict[str, float] = {}

    for method_metrics in report.per_method:
        best_threshold = 0.5  # Default
        best_f1 = 0.0

        for cm in method_metrics.classification_metrics:
            if cm.specificity >= target_specificity and cm.f1 > best_f1:
                best_f1 = cm.f1
                best_threshold = cm.threshold

        recommended[method_metrics.method_name] = best_threshold

    return recommended
